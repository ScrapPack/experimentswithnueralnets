"""
Predictive Coding Gamestate Engine — Flask backend.

Uses the PCNetwork (local ODEs, lateral connections, no autograd) to
physically simulate a destructible 2D grid.  The network resolves
physical strikes by minimizing free energy.

World representation:
    A 10×10 flat tensor of shape (1, 100).  Each cell is in [−1, 1]:
        +1.0 = solid ground
        −1.0 = air / empty

Physics loop (on strike):
    1. Force the struck cell to −1 (destroy).
    2. Run hierarchical inference (20 ODE steps) — the lateral
       connections propagate structural expectations to neighbours,
       and free-energy minimization "settles" the world.
    3. Read the network's top-down prediction as the new world state.

No autograd, no backpropagation — just local ODEs and Hebbian learning.

Run:
    pip install flask
    python game_server.py
"""

from __future__ import annotations

import torch
from flask import Flask, jsonify, request

from pc_layer import _get_device
from pc_network import PCNetwork


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_W: int = 10
GRID_H: int = 10
SENSORY_DIM: int = GRID_W * GRID_H        # 100

LAYER_DIMS: list[int] = [32, SENSORY_DIM]
ACTIVATION: str = "tanh"

# Pre-training — teaches lateral connections to expect solid neighbours
PRETRAIN_STEPS: int = 50
PRETRAIN_INFER_STEPS: int = 10
PRETRAIN_ETA_X: float = 0.1
PRETRAIN_ETA_W: float = 0.01
PRETRAIN_ETA_V: float = 0.01

# Strike resolution
STRIKE_INFER_STEPS: int = 20
STRIKE_ETA_X: float = 0.2


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

app = Flask(__name__)

device = _get_device()
print(f"Device: {device}")

net = PCNetwork(layer_dims=LAYER_DIMS, activation_fn_name=ACTIVATION).to(device)
print(net)
print(f"Grid : {GRID_W}×{GRID_H} = {SENSORY_DIM} cells")

# World state: every cell starts as solid ground (+1.0).
current_world_state = torch.ones(1, SENSORY_DIM, device=device)


# ---------------------------------------------------------------------------
# Pre-training — physics prior
# ---------------------------------------------------------------------------

def pre_train_physics() -> None:
    """Teach the network what "solid ground" looks like.

    For 50 steps, present near-solid targets (all 1s minus slight
    uniform noise in [0, 0.1]).  This trains the weights *and*
    lateral connections to encode the structural expectation that
    neighbours of solid cells should also be solid.
    """
    print(f"\nPre-training physics prior ({PRETRAIN_STEPS} steps) ...")

    with torch.no_grad():
        for step in range(PRETRAIN_STEPS):
            # Near-solid target with slight random perturbation.
            noise = torch.rand(1, SENSORY_DIM, device=device) * 0.1
            target = torch.ones(1, SENSORY_DIM, device=device) - noise

            net.infer(target, steps=PRETRAIN_INFER_STEPS, eta_x=PRETRAIN_ETA_X)
            net.learn(eta_w=PRETRAIN_ETA_W, eta_v=PRETRAIN_ETA_V)

            if step % 10 == 0:
                energy = net.get_total_energy()
                print(f"  step {step:3d} | energy {energy:.1f}")

    print("Pre-training complete.\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/get_state", methods=["GET"])
def get_state():
    """Return the current world state as a flat JSON list of 100 floats."""
    state_list = current_world_state.squeeze(0).cpu().tolist()
    return jsonify(state_list)


@app.route("/strike", methods=["POST"])
def strike():
    """Destroy a block and let the network resolve the physics.

    Expects JSON: ``{"x": <int>, "y": <int>}``

    Returns: flat JSON list of 100 floats (the new world state).
    """
    global current_world_state

    data = request.get_json()
    x = int(data["x"])
    y = int(data["y"])

    # Validate coordinates.
    if not (0 <= x < GRID_W and 0 <= y < GRID_H):
        return jsonify({"error": f"({x}, {y}) out of bounds"}), 400

    idx = y * GRID_W + x

    with torch.no_grad():
        # 1. Force the struck cell to air/empty.
        current_world_state[0, idx] = -1.0

        # 2. Hierarchical inference — lateral connections propagate
        #    structural expectations; free-energy minimization settles
        #    the world into a physically consistent configuration.
        net.infer(current_world_state, steps=STRIKE_INFER_STEPS, eta_x=STRIKE_ETA_X)

        # 3. Read the network's settled top-down prediction as the
        #    new world state.
        current_world_state = (
            net.layers[-1]
            .predict_down(net.layers[-1].x)
            .clamp(-1.0, 1.0)
        )

    state_list = current_world_state.squeeze(0).cpu().tolist()
    return jsonify(state_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pre_train_physics()
    print(f"Server ready → http://127.0.0.1:5000")
    print(f"  GET  /get_state   — fetch the 10×10 world")
    print(f"  POST /strike      — destroy a block (JSON: {{x, y}})\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
