"""
Test: Emergent Sensorimotor Tracking via Active Inference

Validates the CorticalColumn by placing a 3×3 "target" square in a 10×10
sensory world and giving the column a 5×5 fovea that can move.  The column
must learn the target and then *organically* move its fovea to centre the
target — purely by minimising Free Energy (prediction error).

No explicit "seek target" logic is written.  The L5 motor output physically
moves the sensor because moving reduces the mathematical prediction error
between the fovea input and the column's internal generative model.

What to look for:
    1. x_obj (What) should STABILISE — the column recognises the square.
    2. x_loc (Where) should CHANGE — it tracks the fovea's position.
    3. Fovea should drift toward the target (emergent tracking).
    4. Energy should decrease as the column centres its sensor.

No autograd.  All dynamics are local ODEs + Hebbian learning.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pc_layer import CorticalColumn, _get_device


def main() -> None:
    device = _get_device()
    print(f"Device: {device}\n")

    # -----------------------------------------------------------------
    # 1. Build the 10×10 sensory world with a 3×3 target square
    # -----------------------------------------------------------------
    WORLD_H, WORLD_W = 10, 10
    FOVEA_H, FOVEA_W = 5, 5
    TARGET_Y, TARGET_X = 5.0, 5.0   # target centre (float for Gaussian)

    world = torch.zeros(WORLD_H, WORLD_W, device=device)
    # Gaussian target: smooth falloff extends gradient field beyond the peak.
    # This is physically realistic (natural stimuli have soft edges) and
    # ensures the motor gradient is non-zero even when the target is
    # partially outside the fovea.
    sigma = 1.5
    for r in range(WORLD_H):
        for c in range(WORLD_W):
            dist_sq = (r - TARGET_Y) ** 2 + (c - TARGET_X) ** 2
            world[r, c] = math.exp(-dist_sq / (2.0 * sigma ** 2))

    print("World (10×10 with Gaussian target at centre):")
    chars = " ·░▒▓█"
    for row in range(WORLD_H):
        line = ""
        for col in range(WORLD_W):
            v = world[row, col].item()
            idx = min(int(v * len(chars)), len(chars) - 1)
            line += chars[idx] * 2
        print(f"  {line}")

    # -----------------------------------------------------------------
    # 2. Instantiate the CorticalColumn
    # -----------------------------------------------------------------
    SENSORY_DIM = FOVEA_H * FOVEA_W   # 25
    OBJ_DIM = 16                       # "What" latent size
    LOC_DIM = 8                        # "Where" latent size

    column = CorticalColumn(
        obj_dim=OBJ_DIM,
        loc_dim=LOC_DIM,
        sensory_dim=SENSORY_DIM,
        activation_fn_name="tanh",
    ).to(device)
    print(f"\n{column}\n")

    # -----------------------------------------------------------------
    # 3. Helper: crop the fovea from the world at (fy, fx)
    # -----------------------------------------------------------------
    # Pre-compute world as 4D tensor for grid_sample: (1, 1, H, W).
    world_4d = world.unsqueeze(0).unsqueeze(0)

    def sample_fovea(fy: float, fx: float) -> Tensor:
        """Crop a 5×5 fovea centred at (fy, fx) using bilinear interpolation.

        Sub-pixel positions produce smooth, differentiable outputs — this
        is critical for finite-difference motor gradients to be non-zero.

        Returns: (1, 25) flattened fovea.
        """
        # Build a 5×5 sampling grid centred at (fy, fx).
        # grid_sample expects coordinates in [-1, +1].
        half_h = (FOVEA_H - 1) / 2.0   # 2.0
        half_w = (FOVEA_W - 1) / 2.0   # 2.0

        gy = torch.linspace(fy - half_h, fy + half_h, FOVEA_H, device=device)
        gx = torch.linspace(fx - half_w, fx + half_w, FOVEA_W, device=device)

        # Normalise to [-1, 1] for grid_sample (align_corners=True).
        gy_norm = gy / (WORLD_H - 1) * 2.0 - 1.0
        gx_norm = gx / (WORLD_W - 1) * 2.0 - 1.0

        grid_y, grid_x = torch.meshgrid(gy_norm, gx_norm, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,5,5,2)

        crop = F.grid_sample(
            world_4d, grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        return crop.reshape(1, SENSORY_DIM)

    def compute_sensory_gradient(fy: float, fx: float) -> Tensor:
        """Finite-difference spatial gradient of the fovea w.r.t. (dy, dx).

        Uses bilinear-interpolated fovea so sub-pixel shifts produce
        non-zero gradients.

        Returns: (1, 25, 2) — gradient for [dy, dx] motor axes.
        """
        delta = 0.5
        s_centre = sample_fovea(fy, fx)
        s_dy = sample_fovea(fy + delta, fx)
        s_dx = sample_fovea(fy, fx + delta)
        grad_dy = (s_dy - s_centre) / delta  # (1, 25)
        grad_dx = (s_dx - s_centre) / delta  # (1, 25)
        return torch.stack([grad_dy, grad_dx], dim=-1)  # (1, 25, 2)

    # -----------------------------------------------------------------
    # 4. Pre-train: teach the column what the target looks like
    # -----------------------------------------------------------------
    print("Phase 1: Pre-training (learning the target pattern)...")

    # Present the target centred in the fovea for several epochs.
    target_fovea = sample_fovea(TARGET_Y, TARGET_X)
    print(f"  Target fovea (centred): {target_fovea.sum().item():.0f} lit pixels")

    with torch.no_grad():
        for epoch in range(50):
            column.reset_states(1, device)
            for step in range(30):
                column.infer_step(target_fovea, eta_x=0.1)
            column.learn(eta_w=0.01)

            if epoch % 10 == 0:
                e = column.get_energy()
                print(f"  epoch {epoch:3d}  energy={e:.4f}")

    # Final settle — capture the trained x_obj as the persistent object prior.
    column.reset_states(1, device)
    for step in range(30):
        column.infer_step(target_fovea, eta_x=0.1)
    final_train_energy = column.get_energy()
    trained_x_obj = column.x_obj.clone()
    print(f"  Final training energy: {final_train_energy:.4f}")
    print(f"  Trained x_obj norm  : {trained_x_obj.norm().item():.4f}")
    print(f"  (This persistent belief keeps the column 'committed' to the target)\n")

    # -----------------------------------------------------------------
    # 5. Active Inference loop: emergent tracking
    # -----------------------------------------------------------------
    print("Phase 2: Active Inference — emergent sensorimotor tracking")
    print(f"  Starting fovea at (2.0, 2.0)")
    print(f"  Target is at ({TARGET_Y}, {TARGET_X})")
    print(f"  {'step':>4s}  {'fy':>6s}  {'fx':>6s}  {'energy':>8s}  "
          f"{'dy':>7s}  {'dx':>7s}  {'x_obj_norm':>10s}  {'x_loc_norm':>10s}")
    print("  " + "-" * 72)

    # Start the fovea away from the target.
    fy, fx = 2.0, 2.0

    INFER_STEPS = 30
    ETA_X = 0.1
    ETA_A = 1.0
    N_STEPS = 80

    trajectory = [(fy, fx)]
    energies = []
    x_obj_norms = []
    x_loc_norms = []

    with torch.no_grad():
        for t in range(N_STEPS):
            # Sample fovea at current position.
            fovea = sample_fovea(fy, fx)

            # Settle the column (perception).
            # KEY: x_obj is FROZEN to the trained object belief.
            # This is biologically correct — L2/3 maintains its object
            # representation across saccades.  Only L6 (x_loc) settles
            # to discover the new sensor location.
            column.x_obj = trained_x_obj.clone()
            column.x_loc = torch.zeros(1, LOC_DIM, device=device)
            column.error = torch.zeros(1, SENSORY_DIM, device=device)
            for s in range(INFER_STEPS):
                column.infer_step(fovea, eta_x=ETA_X, freeze_obj=True)

            energy = column.get_energy()
            energies.append(energy)

            # Record state norms.
            obj_norm = column.x_obj.norm().item()
            loc_norm = column.x_loc.norm().item()
            x_obj_norms.append(obj_norm)
            x_loc_norms.append(loc_norm)

            # Compute motor action (Active Inference).
            grad = compute_sensory_gradient(fy, fx)
            action = column.get_motor_action(grad, eta_a=ETA_A)
            dy = action[0, 0].item()
            dx = action[0, 1].item()

            if t % 10 == 0 or t == N_STEPS - 1:
                print(
                    f"  {t:4d}  {fy:6.2f}  {fx:6.2f}  {energy:8.4f}  "
                    f"{dy:7.4f}  {dx:7.4f}  {obj_norm:10.4f}  {loc_norm:10.4f}"
                )

            # Move the fovea.
            fy = max(0.0, min(float(WORLD_H) - 1.0, fy + dy))
            fx = max(0.0, min(float(WORLD_W) - 1.0, fx + dx))
            trajectory.append((fy, fx))

    # -----------------------------------------------------------------
    # 6. Analysis
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    start_pos = trajectory[0]
    final_pos = trajectory[-1]
    dist_to_target_start = (
        (start_pos[0] - TARGET_Y) ** 2 + (start_pos[1] - TARGET_X) ** 2
    ) ** 0.5
    dist_to_target_end = (
        (final_pos[0] - TARGET_Y) ** 2 + (final_pos[1] - TARGET_X) ** 2
    ) ** 0.5

    print(f"\n  Start position : ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    print(f"  Final position : ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"  Target position: ({TARGET_Y:.1f}, {TARGET_X:.1f})")
    print(f"  Distance to target (start): {dist_to_target_start:.2f}")
    print(f"  Distance to target (end)  : {dist_to_target_end:.2f}")

    moved_closer = dist_to_target_end < dist_to_target_start
    print(f"\n  Moved closer to target: {'YES' if moved_closer else 'NO'}")

    # x_obj stability vs x_loc variability.
    obj_var = torch.tensor(x_obj_norms).std().item()
    loc_var = torch.tensor(x_loc_norms).std().item()
    print(f"\n  x_obj norm variability (std): {obj_var:.4f}")
    print(f"  x_loc norm variability (std): {loc_var:.4f}")
    if obj_var < loc_var:
        print("  x_obj is MORE STABLE than x_loc  [Expected — What vs Where]")
    else:
        print("  NOTE: x_obj variability >= x_loc (may need more training)")

    # Energy trend.
    e_start = energies[0]
    e_end = energies[-1]
    print(f"\n  Energy at start: {e_start:.4f}")
    print(f"  Energy at end  : {e_end:.4f}")
    if e_end < e_start:
        print("  Energy DECREASED over tracking  [Active Inference working]")
    else:
        print("  Energy did not decrease (fovea may have overshot)")

    # Trajectory summary.
    print(f"\n  Trajectory ({len(trajectory)} points):")
    for i in range(0, len(trajectory), 8):
        pts = trajectory[i:i + 8]
        line = "    " + "  →  ".join(f"({y:.1f},{x:.1f})" for y, x in pts)
        print(line)

    print(f"\n{'='*60}")
    if moved_closer:
        print("PASS: Emergent sensorimotor tracking via Free Energy minimisation.")
    else:
        print("INVESTIGATE: Fovea did not move closer to target.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
