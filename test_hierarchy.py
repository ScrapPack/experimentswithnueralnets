"""
Test: Directed Saccade via Hierarchical Top-Down Attention

Validates the CorticalStack's ability to drive a fovea toward a target
using top-down hierarchical priors from a clamped Level 2 concept.

Setup:
    - 15×15 world with a Gaussian target (sigma=2.0).
    - 9×9 fovea (3×3 grid of 3×3 patches) sampled via bilinear interpolation.
    - CorticalStack: Level 1 = 3×3 CorticalGrid, Level 2 = 1 CorticalColumn.

Phase 1 — Training:
    Center the fovea on the target.  Train the stack so Level 1 learns
    the visual features and Level 2 learns the aggregated L1 state that
    represents "I am looking at the target".

Phase 2 — Directed Saccade:
    Displace the fovea from the target.  Clamp Level 2's x_obj to the
    trained "target concept".  This creates massive top-down prediction
    error in Level 1 (it is told to see a target, but sees blank space).
    Level 1's L5 motor outputs vote on a saccade direction.  The fovea
    shifts each step, guided purely by Free Energy minimisation.

What to look for:
    1. Fovea converges toward the target position.
    2. Energy decreases as the fovea approaches the target.
    3. Top-down error (err_td) decreases as Level 1 aligns with Level 2.
    4. No autograd — all dynamics are local ODEs + Hebbian learning.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from pc_layer import _get_device
from pc_network import CorticalStack


# -------------------------------------------------------------------
# World construction
# -------------------------------------------------------------------

def make_gaussian_world(
    world_h: int, world_w: int,
    target_y: float, target_x: float,
    sigma: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Create a Gaussian blob centered at (target_y, target_x).

    Returns: (world_h, world_w) tensor with peak intensity 1.0.
    """
    ys = torch.arange(world_h, dtype=torch.float32, device=device)
    xs = torch.arange(world_w, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    dist_sq = (grid_y - target_y) ** 2 + (grid_x - target_x) ** 2
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


# -------------------------------------------------------------------
# Bilinear fovea sampling (same pattern as test_column.py)
# -------------------------------------------------------------------

def sample_fovea(
    world_4d: Tensor,
    fy: float, fx: float,
    fovea_h: int, fovea_w: int,
    world_h: int, world_w: int,
    device: torch.device,
) -> Tensor:
    """Crop a fovea_h × fovea_w window centered at (fy, fx).

    Uses F.grid_sample with bilinear interpolation for sub-pixel
    positioning.  Out-of-bounds samples return 0 (padding_mode='zeros').

    Args:
        world_4d: (1, 1, world_h, world_w) — the world image.
        fy, fx:   Fovea centre in pixel coordinates (float).
        fovea_h, fovea_w: Size of the fovea window.

    Returns:
        (1, fovea_h * fovea_w) — flattened fovea observation.
    """
    half_h = (fovea_h - 1) / 2.0
    half_w = (fovea_w - 1) / 2.0

    gy = torch.linspace(fy - half_h, fy + half_h, fovea_h, device=device)
    gx = torch.linspace(fx - half_w, fx + half_w, fovea_w, device=device)

    # Normalise to [-1, 1] for grid_sample.
    gy_norm = gy / (world_h - 1) * 2.0 - 1.0
    gx_norm = gx / (world_w - 1) * 2.0 - 1.0

    grid_y, grid_x = torch.meshgrid(gy_norm, gx_norm, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    crop = F.grid_sample(
        world_4d, grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return crop.reshape(1, fovea_h * fovea_w)


# -------------------------------------------------------------------
# Finite-difference sensory gradient for the full fovea
# -------------------------------------------------------------------

def compute_fovea_gradient(
    world_4d: Tensor,
    fy: float, fx: float,
    fovea_h: int, fovea_w: int,
    world_h: int, world_w: int,
    device: torch.device,
    delta: float = 0.5,
) -> Tensor:
    """Finite-difference gradient of the full fovea w.r.t. (dy, dx).

    Returns:
        (1, fovea_h * fovea_w, 2) — [dy, dx] motor axes.
    """
    s0 = sample_fovea(world_4d, fy, fx, fovea_h, fovea_w, world_h, world_w, device)
    s_dy = sample_fovea(world_4d, fy + delta, fx, fovea_h, fovea_w, world_h, world_w, device)
    s_dx = sample_fovea(world_4d, fy, fx + delta, fovea_h, fovea_w, world_h, world_w, device)

    grad_dy = (s_dy - s0) / delta  # (1, fovea_h*fovea_w)
    grad_dx = (s_dx - s0) / delta  # (1, fovea_h*fovea_w)
    return torch.stack([grad_dy, grad_dx], dim=-1)  # (1, fovea_h*fovea_w, 2)


def split_gradient_to_columns(
    full_gradient: Tensor,
    grid_h: int, grid_w: int,
    patch_h: int, patch_w: int,
) -> list[Tensor]:
    """Split a full fovea gradient into per-column gradients.

    Args:
        full_gradient: (B, fovea_h*fovea_w, action_dim)
        grid_h, grid_w: grid layout.
        patch_h, patch_w: patch size per column.

    Returns:
        List of n_cols tensors, each (B, patch_h*patch_w, action_dim).
    """
    B = full_gradient.shape[0]
    action_dim = full_gradient.shape[2]
    fovea_h = grid_h * patch_h
    fovea_w = grid_w * patch_w

    grad_img = full_gradient.view(B, fovea_h, fovea_w, action_dim)

    per_col_grads: list[Tensor] = []
    for r in range(grid_h):
        for c in range(grid_w):
            y0 = r * patch_h
            x0 = c * patch_w
            patch_grad = grad_img[:, y0:y0 + patch_h, x0:x0 + patch_w, :]
            per_col_grads.append(
                patch_grad.reshape(B, patch_h * patch_w, action_dim)
            )
    return per_col_grads


# -------------------------------------------------------------------
# Main test
# -------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    device = _get_device()
    print(f"Device: {device}\n")

    # ---------------------------------------------------------------
    # 1. Build the world and stack
    # ---------------------------------------------------------------
    WORLD_H, WORLD_W = 15, 15
    FOVEA_H, FOVEA_W = 9, 9        # 3×3 grid of 3×3 patches
    GRID_H, GRID_W = 3, 3
    PATCH_H, PATCH_W = 3, 3
    L1_OBJ_DIM = 36   # overcomplete (4x for patch sensory_dim=9)
    L1_LOC_DIM = 4
    L2_OBJ_DIM = 64
    L2_LOC_DIM = 4

    TARGET_Y, TARGET_X = 7.0, 7.0  # centre of 15×15 world
    SIGMA = 2.0

    world = make_gaussian_world(
        WORLD_H, WORLD_W, TARGET_Y, TARGET_X, SIGMA, device,
    )
    world_4d = world.unsqueeze(0).unsqueeze(0)

    # Print the world.
    print(f"World ({WORLD_H}×{WORLD_W}), Gaussian target at "
          f"({TARGET_Y}, {TARGET_X}), sigma={SIGMA}:")
    chars = " ·░▒▓█"
    for r in range(WORLD_H):
        row_str = "  "
        for c in range(WORLD_W):
            v = world[r, c].item()
            idx = min(int(v * len(chars)), len(chars) - 1)
            row_str += chars[idx] * 2
        print(row_str)
    print()

    stack = CorticalStack(
        grid_h=GRID_H, grid_w=GRID_W,
        l1_obj_dim=L1_OBJ_DIM, l1_loc_dim=L1_LOC_DIM,
        l1_patch_h=PATCH_H, l1_patch_w=PATCH_W,
        l2_obj_dim=L2_OBJ_DIM, l2_loc_dim=L2_LOC_DIM,
        activation_fn_name="tanh",
    ).to(device)
    print(stack)
    print()

    # ---------------------------------------------------------------
    # 2. Training: centre fovea on target, learn the pattern
    # ---------------------------------------------------------------
    train_fovea = sample_fovea(
        world_4d, TARGET_Y, TARGET_X,
        FOVEA_H, FOVEA_W, WORLD_H, WORLD_W, device,
    )

    INFER_STEPS = 50
    ETA_X = 0.1
    ETA_W = 0.01

    # Phase 1a: Train L1 grid ALONE (no L2 top-down).
    # With multiplicative dendritic gating, x_obj starts near zero and
    # grows slowly.  The L2 top-down prior and lateral decay can prevent
    # x_obj from escaping the zero basin.  By training L1 independently
    # first, each column's sensory pathway bootstraps to a meaningful
    # representation before the hierarchy connects.
    N_GRID_EPOCHS = 200
    print(f"Phase 1a: Training L1 grid alone ({N_GRID_EPOCHS} epochs)...")

    with torch.no_grad():
        for epoch in range(N_GRID_EPOCHS):
            history = stack.level1.infer(
                train_fovea, steps=INFER_STEPS, eta_x=ETA_X,
            )
            stack.level1.learn(train_fovea, eta_w=ETA_W)

            if epoch % 40 == 0 or epoch == N_GRID_EPOCHS - 1:
                print(f"  epoch {epoch:3d}  "
                      f"start={history[0]:.4f}  end={history[-1]:.4f}")

    l1_avg_norm = sum(
        col.x_obj.norm().item() for col in stack.level1.columns
    ) / stack.n_l1_cols
    print(f"  L1 avg x_obj norm after Phase 1a: {l1_avg_norm:.4f}")

    # Phase 1b: Train full stack (L1 + L2 together).
    N_STACK_EPOCHS = 100
    print(f"\nPhase 1b: Training full stack ({N_STACK_EPOCHS} epochs)...")

    with torch.no_grad():
        for epoch in range(N_STACK_EPOCHS):
            history = stack.infer(
                train_fovea, steps=INFER_STEPS, eta_x=ETA_X,
            )
            stack.learn(train_fovea, eta_w=ETA_W)

            if epoch % 25 == 0 or epoch == N_STACK_EPOCHS - 1:
                print(f"  epoch {epoch:3d}  "
                      f"start={history[0]:.4f}  end={history[-1]:.4f}")

    # Final settle to capture trained representations.
    history = stack.infer(train_fovea, steps=INFER_STEPS, eta_x=ETA_X)
    trained_l2_x_obj = stack.level2.x_obj.clone()
    trained_l1_x_objs = [col.x_obj.clone() for col in stack.level1.columns]
    trained_energy = history[-1]

    print(f"\n  Trained L2 x_obj norm: {trained_l2_x_obj.norm().item():.4f}")
    print(f"  Trained L1 avg x_obj norm: "
          f"{sum(x.norm().item() for x in trained_l1_x_objs) / len(trained_l1_x_objs):.4f}")
    print(f"  Final training energy: {trained_energy:.4f}")

    # ---------------------------------------------------------------
    # 3. Directed Saccade: displace fovea, clamp L2, run motor loop
    # ---------------------------------------------------------------
    print("\nPhase 2: Directed Saccade (top-down driven motor tracking)")

    fy, fx = 5.0, 5.0  # displaced from target at (7, 7)
    print(f"  Fovea starts at ({fy:.1f}, {fx:.1f})")
    print(f"  Target is at ({TARGET_Y:.1f}, {TARGET_X:.1f})")

    dist_start = math.sqrt((fy - TARGET_Y) ** 2 + (fx - TARGET_X) ** 2)
    print(f"  Starting distance: {dist_start:.2f}\n")

    MOTOR_STEPS = 80
    SETTLE_STEPS = 30
    ETA_A = 1.0

    trajectory: list[tuple[float, float]] = [(fy, fx)]
    energies: list[float] = []

    with torch.no_grad():
        for t in range(MOTOR_STEPS):
            # Sample fovea at current position.
            fovea = sample_fovea(
                world_4d, fy, fx,
                FOVEA_H, FOVEA_W, WORLD_H, WORLD_W, device,
            )

            # Reset states and clamp L2 x_obj + L1 x_objs.
            # Freezing x_obj at trained values prevents the overcomplete
            # model from explaining away the sensory mismatch — the
            # residual prediction error persists and drives the motor.
            for i, col in enumerate(stack.level1.columns):
                col.reset_states(1, device)
                col.x_obj = trained_l1_x_objs[i].clone()
            stack.level2.reset_states(1, device)
            stack.level2.x_obj = trained_l2_x_obj.clone()

            # Settle with freeze_obj=True at both levels.
            # Only x_loc settles — "where am I looking?" updates
            # while "what am I looking for?" stays fixed.
            patches = stack.level1._slice_patches(fovea)
            n_cols = stack.n_l1_cols

            for _ in range(SETTLE_STEPS):
                # GATHER.
                l2_sensory = stack._gather_l1_states()
                td_priors = stack._split_td_priors(
                    stack.level2.predict_down()[0]
                )
                contexts = [
                    stack.level1._gather_neighbor_context(i, 1, device)
                    for i in range(n_cols)
                ]
                # UPDATE — Level 2 with freeze_obj=True.
                stack.level2.infer_step(
                    l2_sensory, eta_x=ETA_X, freeze_obj=True,
                )
                # UPDATE — Level 1 with freeze_obj=True.
                for i in range(n_cols):
                    stack.level1.columns[i].infer_step(
                        patches[i],
                        eta_x=ETA_X,
                        freeze_obj=True,
                        neighbor_context=contexts[i],
                        top_down_prior=td_priors[i],
                    )

            energy = stack.get_total_energy()
            energies.append(energy)

            # Compute motor action.
            full_grad = compute_fovea_gradient(
                world_4d, fy, fx,
                FOVEA_H, FOVEA_W, WORLD_H, WORLD_W, device,
            )
            per_col_grads = split_gradient_to_columns(
                full_grad, GRID_H, GRID_W, PATCH_H, PATCH_W,
            )
            action = stack.get_global_action(per_col_grads, eta_a=ETA_A)
            dy, dx = action[0, 0].item(), action[0, 1].item()

            if t % 10 == 0 or t == MOTOR_STEPS - 1:
                dist_now = math.sqrt(
                    (fy - TARGET_Y) ** 2 + (fx - TARGET_X) ** 2
                )
                print(f"  step {t:3d}  fy={fy:6.2f}  fx={fx:6.2f}  "
                      f"dist={dist_now:.2f}  energy={energy:.2f}  "
                      f"dy={dy:+.4f}  dx={dx:+.4f}")

            # Move fovea (clamped to world bounds).
            fy = max(0.0, min(float(WORLD_H) - 1.0, fy + dy))
            fx = max(0.0, min(float(WORLD_W) - 1.0, fx + dx))
            trajectory.append((fy, fx))

    # ---------------------------------------------------------------
    # 4. Analysis
    # ---------------------------------------------------------------
    final_pos = trajectory[-1]
    dist_end = math.sqrt(
        (final_pos[0] - TARGET_Y) ** 2 + (final_pos[1] - TARGET_X) ** 2
    )

    print(f"\n{'='*68}")
    print("ANALYSIS")
    print(f"{'='*68}")

    print(f"\n  Start position : ({trajectory[0][0]:.2f}, {trajectory[0][1]:.2f})")
    print(f"  Final position : ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Target position: ({TARGET_Y:.1f}, {TARGET_X:.1f})")
    print(f"  Distance start : {dist_start:.2f}")
    print(f"  Distance end   : {dist_end:.2f}")

    moved_closer = dist_end < dist_start
    energy_decreased = energies[-1] < energies[0] if len(energies) > 1 else False
    significant = dist_end < dist_start * 0.5  # at least 50% closer

    print(f"\n  Moved closer:          {'YES' if moved_closer else 'NO'}")
    print(f"  Distance reduced >50%: {'YES' if significant else 'NO'}")
    print(f"  Energy decreased:      {'YES' if energy_decreased else 'NO'}  "
          f"({energies[0]:.2f} → {energies[-1]:.2f})")

    # L2 + L1 x_obj should remain clamped (identical to trained values).
    l2_stable = torch.allclose(
        stack.level2.x_obj, trained_l2_x_obj, atol=1e-4,
    )
    l1_stable = all(
        torch.allclose(col.x_obj, trained_l1_x_objs[i], atol=1e-4)
        for i, col in enumerate(stack.level1.columns)
    )
    print(f"  L2 x_obj stable:       {'YES' if l2_stable else 'NO'}")
    print(f"  L1 x_obj stable:       {'YES' if l1_stable else 'NO'}")

    # ---------------------------------------------------------------
    # 5. Verdict
    # ---------------------------------------------------------------
    print(f"\n{'='*68}")

    passed = True
    checks = []

    checks.append(("Fovea moved closer to target", moved_closer))
    if not moved_closer:
        passed = False

    checks.append(("Distance reduced by >50%", significant))
    if not significant:
        passed = False

    checks.append(("Energy decreased during saccade", energy_decreased))
    if not energy_decreased:
        passed = False

    # No autograd.
    all_no_grad = all(
        not p.requires_grad for p in stack.parameters()
    )
    checks.append(("All requires_grad=False", all_no_grad))
    if not all_no_grad:
        passed = False

    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")

    print(f"\n{'='*68}")
    if passed:
        print("PASS: Directed saccade via hierarchical top-down attention.")
    else:
        print("INVESTIGATE: Saccade did not converge — tuning may be needed.")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
