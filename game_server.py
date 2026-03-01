"""
Predictive Coding Gamestate Engine — Flask backend.

Uses the PCNetwork (local ODEs, lateral connections, no autograd) to
physically simulate a destructible 2D grid.  The network resolves
physical strikes by minimizing free energy.

World representation:
    A 10×10 dual-channel tensor of shape (1, 200).
    Indices 0–99:   Density (+1.0 = solid, −1.0 = air)
    Indices 100–199: Thermal (−1.0 = cold, +1.0 = hot)

Materials (Phase 8):
    The grid is split into two materials with per-cell property maps:
      Stone (left half, x < 5) — brittle and strong:
        precision = 1.5 (amplifies stress → brittle shattering)
        tensile   = 0.95 (passes stress easily → deep fractures)
        threshold = −0.1 (holds until severely damaged)
      Sand (right half, x ≥ 5) — malleable and weak:
        precision = 0.5 (dampens stress → soft absorption)
        tensile   = 0.4 (stress decays fast → localized crumbling)
        threshold = 0.4 (crumbles very easily)

Physics loop (on strike):
    1. Mark the cell as permanently destroyed (damage mask).
    2. Run hierarchical inference so the network "reacts" to the damage.
    3. Read the sensory prediction error at the struck cell — high error
       means the network is *surprised*, indicating structural stress.
    4. Max-propagate that stress spatially with a gravity-biased kernel
       (damage spreads downward more than sideways; max prevents
       exponential blow-up).
    5. Weaken cells proportionally to the propagated stress.
    6. Force all destroyed cells to −1 (damage is permanent).
    7. Auto-collapse settle loop: any cell below COLLAPSE_THRESHOLD is
       added to the destroyed set, triggering fresh stress propagation
       from the newly collapsed cell.  Repeat until no new collapses
       (up to MAX_SETTLE_ROUNDS).

The network provides the *structural assessment* (where is stress?),
and the spatial kernel provides *physical locality* (stress spreads
to neighbours, with gravity).

No autograd, no backpropagation — just local ODEs and Hebbian learning.

Run:
    pip install flask
    python game_server.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory

from pc_layer import _get_device
from pc_network import PCNetwork


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_W: int = 10
GRID_H: int = 10
GRID_CELLS: int = GRID_W * GRID_H          # 100 (per channel)
SENSORY_DIM: int = GRID_CELLS * 2            # 200 (density + thermal)

LAYER_DIMS: list[int] = [32, SENSORY_DIM]    # [32, 200]
ACTIVATION: str = "tanh"

# Pre-training
PRETRAIN_STEPS: int = 200
PRETRAIN_INFER_STEPS: int = 10
PRETRAIN_ETA_X: float = 0.1
PRETRAIN_ETA_W: float = 0.01
PRETRAIN_ETA_V: float = 0.01

# Strike resolution
STRIKE_INFER_STEPS: int = 20
STRIKE_ETA_X: float = 0.2

# Damage propagation
STRESS_SCALE: float = 0.8          # scalar applied to propagated stress
PROPAGATION_PASSES: int = 5        # convolution passes (reach = 5 cells)
COLLAPSE_THRESHOLD: float = 0.3    # weakened cells auto-collapse
MAX_SETTLE_ROUNDS: int = 10        # max cascade iterations per strike


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

app = Flask(__name__)

device = _get_device()
print(f"Device: {device}")

net = PCNetwork(layer_dims=LAYER_DIMS, activation_fn_name=ACTIVATION).to(device)
print(net)
print(f"Grid : {GRID_W}×{GRID_H} = {GRID_CELLS} cells × 2 channels = {SENSORY_DIM} dims")

# World state: density = solid (+1.0), thermal = cold (−1.0).
current_world_state = torch.cat([
    torch.ones(1, GRID_CELLS, device=device),         # density: solid
    torch.full((1, GRID_CELLS), -1.0, device=device), # thermal: cold
], dim=1)

# Permanent damage tracking — struck cells never heal.
destroyed_cells: set[int] = set()

# Gravity-biased 3×3 propagation kernel.
# In F.conv2d (cross-correlation), kernel[0,1] weights the input from
# the cell ABOVE (y−1) into the current cell.  For gravity (damage
# propagates downward), we want the cell BELOW to *receive* the most
# stress from the cell ABOVE — so kernel[0,1] must be the heaviest.
#
#   k[0,1] = 0.30  →  strong downward propagation  (gravity)
#   k[1,0] = k[1,2] = 0.15  →  moderate lateral spread
#   k[2,1] = 0.05  →  weak upward propagation
#
_GRAVITY_KERNEL = torch.tensor(
    [[0.05, 0.30, 0.05],
     [0.15, 0.00, 0.15],
     [0.02, 0.05, 0.02]],
    device=device,
).view(1, 1, 3, 3)

# Thermal radiance kernel — uniform spread in all directions.
# Heat has no directional bias (unlike gravity-biased structural stress).
_RADIANCE_KERNEL = torch.tensor(
    [[0.15, 0.15, 0.15],
     [0.15, 0.00, 0.15],
     [0.15, 0.15, 0.15]],
    device=device,
).view(1, 1, 3, 3)

# ---------------------------------------------------------------------------
# Material properties — heterogeneous 2D maps
# ---------------------------------------------------------------------------
#
# precision_map : How much a cell amplifies stress on destruction
#                 (brittleness).  High → sharp fracture; low → soft crumble.
# tensile_map   : How efficiently a cell transfers stress to neighbours
#                 during propagation (0 = absorbs all, 1 = passes all).
# threshold_map : The value below which a cell auto-collapses.
#

precision_map = torch.ones(GRID_H, GRID_W, device=device)
tensile_map   = torch.ones(GRID_H, GRID_W, device=device)
threshold_map = torch.full((GRID_H, GRID_W), COLLAPSE_THRESHOLD, device=device)

# Stone — left half (x < 5): brittle and strong.
precision_map[:, :5] = 1.5
tensile_map[:, :5]   = 0.95
threshold_map[:, :5] = -0.1

# Sand — right half (x ≥ 5): malleable and weak.
precision_map[:, 5:] = 0.5
tensile_map[:, 5:]   = 0.4
threshold_map[:, 5:] = 0.4

# Pre-shaped for element-wise use inside _propagate_stress.
_tensile_map_2d = tensile_map.view(1, 1, GRID_H, GRID_W)

# Flat material-label array returned by the API.
_material_labels: list[str] = [
    "stone" if x < 5 else "sand"
    for y in range(GRID_H)
    for x in range(GRID_W)
]

print(f"Materials: stone (x<5) | sand (x≥5)")


# ---------------------------------------------------------------------------
# Pre-training — physics prior
# ---------------------------------------------------------------------------

def pre_train_physics() -> None:
    """Teach the network what "solid ground" looks like.

    Phase 1 (steps 0–149): present near-solid targets so the network
    learns the baseline "everything is solid" prior.

    Phase 2 (steps 150–199): present targets with 1–5 random holes so
    the network learns what *damage* looks like and how it contrasts
    with the solid prior.  This makes the sensory prediction error
    a meaningful structural-stress signal.
    """
    print(f"\nPre-training physics prior ({PRETRAIN_STEPS} steps) ...")

    with torch.no_grad():
        for step in range(PRETRAIN_STEPS):
            # Density channel: solid baseline.
            density = torch.ones(1, GRID_CELLS, device=device)
            # Thermal channel: cold baseline.
            thermal = torch.full((1, GRID_CELLS), -1.0, device=device)

            if step < 150:
                # Phase 1 — solid with slight noise (density only).
                density -= torch.rand(1, GRID_CELLS, device=device) * 0.1
            else:
                # Phase 2 — random holes (density only).
                n_holes = torch.randint(1, 6, (1,)).item()
                hole_idxs = torch.randint(0, GRID_CELLS, (n_holes,))
                density[0, hole_idxs] = -1.0

            target = torch.cat([density, thermal], dim=1)  # (1, 200)

            net.infer(target, steps=PRETRAIN_INFER_STEPS, eta_x=PRETRAIN_ETA_X)
            net.learn(eta_w=PRETRAIN_ETA_W, eta_v=PRETRAIN_ETA_V)

            if step % 40 == 0:
                energy = net.get_total_energy()
                print(f"  step {step:3d} | energy {energy:.1f}")

    print("Pre-training complete.\n")


# ---------------------------------------------------------------------------
# Physics resolution
# ---------------------------------------------------------------------------

def _force_destroyed(state: torch.Tensor) -> None:
    """Stamp all destroyed cells to −1 in-place."""
    for d in destroyed_cells:
        state[0, d] = -1.0


def _propagate_stress(
    state: torch.Tensor,
    stress: torch.Tensor,
) -> torch.Tensor:
    """Spatially propagate stress with a gravity-biased kernel.

    Uses max-propagation (not accumulative sum) to prevent exponential
    blow-up.  Each pass extends the stress frontier by one cell; the
    ``max`` ensures that stress decays monotonically from the source
    (like a distance transform, not a diffusion).

    Before the ``max``, each convolution result is element-wise
    multiplied by ``_tensile_map_2d`` — the material's tensile
    property.  Stone (0.95) passes stress almost undiminished
    (deep fractures); sand (0.40) attenuates rapidly (local crumble).

    Effective decay per hop (kernel weight × tensile):
        Stone downward: 0.30 × 0.95 = 0.285
        Sand  downward: 0.30 × 0.40 = 0.120
        Stone lateral:  0.15 × 0.95 = 0.143
        Sand  lateral:  0.15 × 0.40 = 0.060

    Args:
        state:  (1, 200) full world state (density + thermal).
        stress: (1, 100) stress seeded at the struck cell only
                (density-grid indices).

    Returns:
        (1, 200) world state with density slice weakened.
    """
    result = state.clone()

    # Max-propagation: each pass extends the stress field by one cell.
    # torch.max prevents accumulation — stress decays from the source.
    spread = stress.view(1, 1, GRID_H, GRID_W)
    for _ in range(PROPAGATION_PASSES):
        padded = F.pad(spread, [1, 1, 1, 1], mode="constant", value=0.0)
        conv_result = F.conv2d(padded, _GRAVITY_KERNEL)
        # Material dampening: tensile_map modulates how much stress
        # each *receiving* cell accepts from its neighbours.
        conv_result = conv_result * _tensile_map_2d
        spread = torch.max(spread, conv_result)

    # Flatten and apply weakening to density slice only (0–99).
    weakening = spread.view(1, GRID_CELLS) * STRESS_SCALE
    result[:, :GRID_CELLS] = (
        result[:, :GRID_CELLS] - weakening
    ).clamp(-1.0, 1.0)

    return result


def _propagate_heat(
    state: torch.Tensor,
    stress: torch.Tensor,
) -> torch.Tensor:
    """Propagate thermal stress with the uniform radiance kernel.

    Same max-propagation strategy as structural stress, but:
      - Uses _RADIANCE_KERNEL (uniform, no directional bias).
      - No tensile dampening — heat flows freely through all materials.
      - Operates on the thermal slice (indices 100–199).
      - ADDS heat energy (moves toward +1.0) instead of weakening.

    Args:
        state:  (1, 200) full world state.
        stress: (1, 100) thermal stress seeded at the heated cell.

    Returns:
        (1, 200) world state with thermal slice heated.
    """
    result = state.clone()

    spread = stress.view(1, 1, GRID_H, GRID_W)
    for _ in range(PROPAGATION_PASSES):
        padded = F.pad(spread, [1, 1, 1, 1], mode="constant", value=0.0)
        spread = torch.max(spread, F.conv2d(padded, _RADIANCE_KERNEL))

    # Add heat to the thermal slice (100–199).
    heating = spread.view(1, GRID_CELLS) * STRESS_SCALE
    result[:, GRID_CELLS:] = (
        result[:, GRID_CELLS:] + heating
    ).clamp(-1.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Serve the browser debug client."""
    return send_from_directory(
        Path(__file__).resolve().parent, "debug_client.html",
    )


@app.route("/get_state", methods=["GET"])
def get_state():
    """Return the world state and static material labels.

    Returns JSON::

        {"state": [200 floats], "materials": [100 strings]}

    ``state`` contains 200 values: indices 0–99 are density,
    indices 100–199 are thermal.  ``materials`` is a flat list of
    ``"stone"`` or ``"sand"`` for each cell (row-major, 100 entries).
    """
    state_list = current_world_state.squeeze(0).cpu().tolist()
    return jsonify({"state": state_list, "materials": _material_labels})


@app.route("/strike", methods=["POST"])
def strike():
    """Destroy a block and resolve the structural physics.

    1. Mark the cell as permanently destroyed.
    2. Run inference — the network's sensory prediction error reveals
       where it is structurally "surprised" by the damage.
    3. Propagate that stress spatially with a gravity-biased kernel.
    4. Weaken cells proportionally; destroyed cells stay at −1.

    Expects JSON: ``{"x": <int>, "y": <int>}``
    Returns: flat JSON list of 200 floats (the new world state).
    """
    global current_world_state

    data = request.get_json()
    x = int(data["x"])
    y = int(data["y"])

    if not (0 <= x < GRID_W and 0 <= y < GRID_H):
        return jsonify({"error": f"({x}, {y}) out of bounds"}), 400

    idx = y * GRID_W + x
    destroyed_cells.add(idx)

    with torch.no_grad():
        # 0. Snapshot the old cell value (used for stress magnitude).
        old_value = current_world_state[0, idx].item()

        # 1. Force all destroyed cells to −1 (damage is permanent).
        _force_destroyed(current_world_state)

        # 2. Run inference — the network settles its beliefs given
        #    the damaged sensory input.
        net.infer(
            current_world_state, steps=STRIKE_INFER_STEPS, eta_x=STRIKE_ETA_X,
        )

        # 3. Stress = load released × material precision.
        #    Habituation (load-based): |old − (−1)| = old + 1.
        #    Precision modulates brittleness:
        #      Stone (1.5) amplifies stress → sharp fractures.
        #      Sand  (0.5) dampens stress  → soft crumble.
        load_released = abs(old_value + 1.0)
        stress_magnitude = load_released * precision_map[y, x].item()
        mat_label = "stone" if x < 5 else "sand"
        print(
            f"  Strike ({x},{y}) idx={idx} [{mat_label}]  "
            f"old={old_value:.3f}  load={load_released:.3f}  "
            f"stress={stress_magnitude:.3f}",
            flush=True,
        )

        stress = torch.zeros(1, GRID_CELLS, device=device)
        stress[0, idx] = stress_magnitude

        # 4. Propagate stress spatially (gravity-biased max-propagation)
        #    and weaken cells proportionally (density slice only).
        current_world_state = _propagate_stress(current_world_state, stress)

        # 5. Destroyed cells are always −1.
        _force_destroyed(current_world_state)

        # 6. Auto-collapse settle loop — cells that drop below the
        #    collapse threshold are added to destroyed_cells.  Each
        #    newly collapsed cell seeds a fresh round of stress
        #    propagation (stress = load released), enabling cascading
        #    chain reactions.
        for _settle in range(MAX_SETTLE_ROUNDS):
            newly_collapsed: list[tuple[int, float]] = []
            for i in range(GRID_CELLS):
                val = current_world_state[0, i].item()
                y_i = i // GRID_W
                x_i = i % GRID_W
                # Per-material threshold: stone (−0.1) holds longer;
                # sand (0.4) crumbles at the first sign of weakness.
                if (
                    i not in destroyed_cells
                    and val < threshold_map[y_i, x_i].item()
                ):
                    newly_collapsed.append((i, val))

            if not newly_collapsed:
                break

            for i, _val in newly_collapsed:
                destroyed_cells.add(i)

            # Snapshot values BEFORE forcing to −1 (for stress calc).
            old_vals = {i: v for i, v in newly_collapsed}
            _force_destroyed(current_world_state)

            # Propagate stress from each newly collapsed cell,
            # modulated by that cell's material precision.
            for i, old_v in newly_collapsed:
                y_i = i // GRID_W
                x_i = i % GRID_W
                cascade_stress = (
                    abs(old_v + 1.0) * precision_map[y_i, x_i].item()
                )
                stress = torch.zeros(1, GRID_CELLS, device=device)
                stress[0, i] = cascade_stress
                current_world_state = _propagate_stress(
                    current_world_state, stress,
                )

            _force_destroyed(current_world_state)

            collapsed_ids = [i for i, _ in newly_collapsed]
            print(
                f"    cascade round {_settle + 1}: "
                f"{len(newly_collapsed)} cells collapsed {collapsed_ids}",
                flush=True,
            )

    state_list = current_world_state.squeeze(0).cpu().tolist()
    return jsonify(state_list)


@app.route("/heat", methods=["POST"])
def heat():
    """Apply heat to a cell and propagate thermal energy.

    1. Force the cell's thermal value to +1.0 (maximum heat).
    2. Run inference — the network reacts to the thermal anomaly.
    3. Propagate thermal stress with the radiance kernel (uniform,
       no tensile dampening — heat flows freely).

    Expects JSON: ``{"x": <int>, "y": <int>}``
    Returns: flat JSON list of 200 floats (the new world state).
    """
    global current_world_state

    data = request.get_json()
    x = int(data["x"])
    y = int(data["y"])

    if not (0 <= x < GRID_W and 0 <= y < GRID_H):
        return jsonify({"error": f"({x}, {y}) out of bounds"}), 400

    grid_idx = y * GRID_W + x
    thermal_idx = grid_idx + GRID_CELLS  # offset into thermal slice

    with torch.no_grad():
        old_thermal = current_world_state[0, thermal_idx].item()

        # 1. Force maximum heat at the target cell.
        current_world_state[0, thermal_idx] = 1.0

        # 2. Run inference — the network settles given the thermal spike.
        net.infer(
            current_world_state, steps=STRIKE_INFER_STEPS, eta_x=STRIKE_ETA_X,
        )

        # 3. Thermal stress = surprise at this cell.
        #    A cold cell (−1 → +1) produces maximum stress (2.0);
        #    an already-warm cell (0.5 → +1) produces less (0.5).
        thermal_stress = abs(old_thermal - 1.0)
        mat_label = "stone" if x < 5 else "sand"
        print(
            f"  Heat ({x},{y}) idx={grid_idx} [{mat_label}]  "
            f"old_t={old_thermal:.3f}  stress={thermal_stress:.3f}",
            flush=True,
        )

        # 4. Propagate thermal stress with the radiance kernel.
        stress = torch.zeros(1, GRID_CELLS, device=device)
        stress[0, grid_idx] = thermal_stress
        current_world_state = _propagate_heat(current_world_state, stress)

        # 5. Ensure destroyed density cells stay destroyed.
        _force_destroyed(current_world_state)

    state_list = current_world_state.squeeze(0).cpu().tolist()
    return jsonify(state_list)


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the world: density → solid, thermal → cold."""
    global current_world_state
    destroyed_cells.clear()
    current_world_state = torch.cat([
        torch.ones(1, GRID_CELLS, device=device),
        torch.full((1, GRID_CELLS), -1.0, device=device),
    ], dim=1)
    return jsonify(current_world_state.squeeze(0).cpu().tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pre_train_physics()
    print(f"Server ready → http://127.0.0.1:5001")
    print(f"  GET  /get_state   — fetch 200-dim state (density + thermal)")
    print(f"  POST /strike      — destroy a block   (JSON: {{x, y}})")
    print(f"  POST /heat        — heat a block      (JSON: {{x, y}})")
    print(f"  POST /reset       — restore the grid\n")
    app.run(host="127.0.0.1", port=5001, debug=False)
