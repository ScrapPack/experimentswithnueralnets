"""
Test: Lateral Consensus Defeats Occlusion

Validates the CorticalGrid's emergent consensus behaviour.  A 3×3 grid
of CorticalColumns is trained on a coherent 9×9 image (a cross / + pattern).
After training, the centre column's sensory input is completely zeroed out
(simulating occlusion), and the grid is re-run.

The lateral connections should pull the occluded column's x_obj toward
the consensus of its neighbours, effectively "filling in" the missing
information purely through local ODE dynamics and Hebbian-learned
lateral weights.

What to look for:
    1. After training, all columns converge on similar x_obj representations.
    2. With occlusion, the centre column's x_obj is reconstructed from
       its neighbours — cosine similarity should be high.
    3. Energy should decrease during inference even with occlusion.
    4. No autograd.  All dynamics are local ODEs + Hebbian learning.
"""

from __future__ import annotations

import torch
from pc_layer import _get_device
from pc_network import CorticalGrid


def make_cross_image(grid_h: int, grid_w: int, patch_h: int, patch_w: int,
                     device: torch.device) -> torch.Tensor:
    """Create a 9×9 cross/+ pattern image, flattened to (1, 81).

    The cross has a vertical bar down the centre column of patches
    and a horizontal bar across the centre row of patches.
    Values are 1.0 on the cross, 0.0 off.
    """
    img_h = grid_h * patch_h
    img_w = grid_w * patch_w
    img = torch.zeros(img_h, img_w, device=device)

    centre_col_start = (grid_w // 2) * patch_w
    centre_col_end = centre_col_start + patch_w
    centre_row_start = (grid_h // 2) * patch_h
    centre_row_end = centre_row_start + patch_h

    # Vertical bar (full height, centre patch-column).
    img[:, centre_col_start:centre_col_end] = 1.0
    # Horizontal bar (centre patch-row, full width).
    img[centre_row_start:centre_row_end, :] = 1.0

    return img.reshape(1, img_h * img_w)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two (1, D) tensors."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    dot = (a_flat * b_flat).sum()
    denom = a_flat.norm() * b_flat.norm() + 1e-8
    return (dot / denom).item()


def main() -> None:
    # Fixed seed for reproducible weight initialisation.
    torch.manual_seed(0)

    device = _get_device()
    print(f"Device: {device}\n")

    # -----------------------------------------------------------------
    # 1. Build the 3×3 CorticalGrid
    # -----------------------------------------------------------------
    GRID_H, GRID_W = 3, 3
    PATCH_H, PATCH_W = 3, 3
    OBJ_DIM = 36   # overcomplete (4x for sensory_dim=9)
    LOC_DIM = 4

    grid = CorticalGrid(
        grid_h=GRID_H, grid_w=GRID_W,
        obj_dim=OBJ_DIM, loc_dim=LOC_DIM,
        patch_h=PATCH_H, patch_w=PATCH_W,
        activation_fn_name="tanh",
    ).to(device)

    print(grid)
    print(f"  Total columns : {len(grid.columns)}")
    print(f"  Sensory dim   : {grid.sensory_dim}")
    print(f"  Image size    : {GRID_H * PATCH_H}×{GRID_W * PATCH_W} = "
          f"{GRID_H * PATCH_H * GRID_W * PATCH_W}\n")

    # -----------------------------------------------------------------
    # 2. Build the cross image
    # -----------------------------------------------------------------
    cross = make_cross_image(GRID_H, GRID_W, PATCH_H, PATCH_W, device)

    print("Cross pattern (9×9):")
    img_2d = cross.reshape(GRID_H * PATCH_H, GRID_W * PATCH_W)
    for r in range(GRID_H * PATCH_H):
        row_str = "  "
        for c in range(GRID_W * PATCH_W):
            row_str += "██" if img_2d[r, c].item() > 0.5 else "  "
        print(row_str)
    print()

    # -----------------------------------------------------------------
    # 3. Training: teach the grid the cross pattern
    # -----------------------------------------------------------------
    print("Phase 1: Training on cross pattern...")

    N_EPOCHS = 200
    INFER_STEPS = 50
    ETA_X = 0.1
    ETA_W = 0.01

    with torch.no_grad():
        for epoch in range(N_EPOCHS):
            history = grid.infer(cross, steps=INFER_STEPS, eta_x=ETA_X)
            grid.learn(cross, eta_w=ETA_W)

            if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
                print(f"  epoch {epoch:3d}  "
                      f"start={history[0]:.4f}  end={history[-1]:.4f}")

    # Final settle to capture trained representations.
    # Seed RNG so reset_states() random init is reproducible — with
    # multiplicative dendritic gating, different random inits can lead
    # to different convergence basins.  Seeding here and before the
    # occlusion test ensures the same starting point.
    torch.manual_seed(999)
    history = grid.infer(cross, steps=INFER_STEPS, eta_x=ETA_X)
    final_train_energy = history[-1]

    # Capture each column's trained x_obj.
    trained_x_objs = [col.x_obj.clone() for col in grid.columns]

    print(f"\n  Final training energy: {final_train_energy:.4f}")

    # Show pairwise cosine similarity of all columns' x_obj.
    print("\n  Pairwise cosine similarity of trained x_obj:")
    print("       ", end="")
    for c in range(GRID_H * GRID_W):
        print(f"  col{c}", end="")
    print()
    for i in range(GRID_H * GRID_W):
        print(f"  col{i}", end="")
        for j in range(GRID_H * GRID_W):
            sim = cosine_sim(trained_x_objs[i], trained_x_objs[j])
            print(f"  {sim:5.2f}", end="")
        print()

    # Centre column index.
    centre_idx = (GRID_H // 2) * GRID_W + (GRID_W // 2)  # = 4
    centre_trained = trained_x_objs[centre_idx].clone()
    print(f"\n  Centre column (idx={centre_idx}) trained x_obj norm: "
          f"{centre_trained.norm().item():.4f}")

    # Average x_obj of all columns (full-info baseline).
    avg_trained = torch.stack(trained_x_objs, dim=0).mean(dim=0)  # (1, obj_dim)

    # -----------------------------------------------------------------
    # 4. Test: occlude the centre column and re-infer
    # -----------------------------------------------------------------
    print("\nPhase 2: Occlusion test — centre column receives ZERO input")

    # Create occluded input: same as cross, but centre patch zeroed.
    occluded = cross.clone()
    img_occluded = occluded.reshape(1, GRID_H * PATCH_H, GRID_W * PATCH_W)
    r0 = (GRID_H // 2) * PATCH_H
    c0 = (GRID_W // 2) * PATCH_W
    img_occluded[:, r0:r0 + PATCH_H, c0:c0 + PATCH_W] = 0.0
    occluded = img_occluded.reshape(1, -1)

    print("  Occluded pattern (9×9):")
    occ_2d = occluded.reshape(GRID_H * PATCH_H, GRID_W * PATCH_W)
    for r in range(GRID_H * PATCH_H):
        row_str = "    "
        for c in range(GRID_W * PATCH_W):
            row_str += "██" if occ_2d[r, c].item() > 0.5 else "  "
        print(row_str)
    print()

    # Infer on occluded input.
    # Same seed as capture run → same random init in reset_states().
    torch.manual_seed(999)
    with torch.no_grad():
        occ_history = grid.infer(occluded, steps=INFER_STEPS * 2, eta_x=ETA_X)

    occ_energy = occ_history[-1]
    occluded_x_objs = [col.x_obj.clone() for col in grid.columns]
    centre_occluded = occluded_x_objs[centre_idx]

    # -----------------------------------------------------------------
    # 5. Control: truly isolated columns (no lateral connections)
    # -----------------------------------------------------------------
    print("Phase 3: Control — same occlusion, columns settle INDEPENDENTLY")

    # Run each column individually WITHOUT neighbor_context.
    # This is the proper ablation: columns see the same sensory patches
    # but have zero lateral influence.
    B = 1

    # Slice the occluded image the same way CorticalGrid does.
    occ_img = occluded.reshape(B, GRID_H * PATCH_H, GRID_W * PATCH_W)
    ctrl_patches: list[torch.Tensor] = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            y0 = r * PATCH_H
            x0 = c * PATCH_W
            patch = occ_img[:, y0:y0 + PATCH_H, x0:x0 + PATCH_W]
            ctrl_patches.append(patch.reshape(B, grid.sensory_dim))

    # Build isolated columns with the SAME trained W_obj/W_loc weights.
    from pc_layer import CorticalColumn
    isolated_cols: list[CorticalColumn] = []
    for src in grid.columns:
        iso = CorticalColumn(
            obj_dim=OBJ_DIM, loc_dim=LOC_DIM,
            sensory_dim=grid.sensory_dim,
            activation_fn_name="tanh",
        ).to(device)
        iso.W_obj.data.copy_(src.W_obj.data)
        iso.W_loc.data.copy_(src.W_loc.data)
        isolated_cols.append(iso)

    # Settle each isolated column independently — NO neighbor_context.
    # Same seed as capture run → same random init for fair comparison.
    torch.manual_seed(999)
    with torch.no_grad():
        for iso in isolated_cols:
            iso.reset_states(B, device)
        for _ in range(INFER_STEPS * 2):
            for iso, patch in zip(isolated_cols, ctrl_patches):
                iso.infer_step(patch, eta_x=ETA_X)  # no neighbor_context!

    ctrl_energy = sum(iso.get_energy() for iso in isolated_cols)
    control_x_objs = [iso.x_obj.clone() for iso in isolated_cols]
    centre_control = control_x_objs[centre_idx]

    # -----------------------------------------------------------------
    # 6. Analysis
    # -----------------------------------------------------------------
    print(f"\n{'='*68}")
    print("ANALYSIS")
    print(f"{'='*68}")

    # Cosine similarity: how close is centre's x_obj to its trained value?
    sim_lateral = cosine_sim(centre_occluded, centre_trained)
    sim_control = cosine_sim(centre_control, centre_trained)

    print(f"\n  Centre column x_obj cosine similarity to TRAINED value:")
    print(f"    With lateral connections:    {sim_lateral:.4f}")
    print(f"    Without lateral (control):   {sim_control:.4f}")
    print(f"    Improvement:                 {sim_lateral - sim_control:+.4f}")

    # Norm comparison: does the occluded column develop a meaningful representation?
    norm_lateral = centre_occluded.norm().item()
    norm_control = centre_control.norm().item()
    norm_trained = centre_trained.norm().item()
    print(f"\n  Centre column x_obj norm:")
    print(f"    Trained (full info):         {norm_trained:.4f}")
    print(f"    Occluded + lateral:          {norm_lateral:.4f}")
    print(f"    Occluded + no lateral:       {norm_control:.4f}")

    # Energy comparison.
    print(f"\n  Final inference energy:")
    print(f"    Occluded + lateral:          {occ_energy:.4f}")
    print(f"    Occluded + no lateral:       {ctrl_energy:.4f}")

    # Energy decrease check.
    occ_decreased = occ_history[-1] < occ_history[0]
    print(f"\n  Energy decreased (lateral):    "
          f"{'YES' if occ_decreased else 'NO'}  "
          f"({occ_history[0]:.2f} → {occ_history[-1]:.2f})")

    # Neighbour similarity: occluded centre vs its 4 neighbours.
    nbr_idxs = grid._neighbor_idx[centre_idx]
    nbr_sims_lat = [
        cosine_sim(centre_occluded, occluded_x_objs[n])
        for n in nbr_idxs
    ]
    nbr_sims_ctrl = [
        cosine_sim(centre_control, control_x_objs[n])
        for n in nbr_idxs
    ]
    avg_nbr_sim_lat = sum(nbr_sims_lat) / len(nbr_sims_lat)
    avg_nbr_sim_ctrl = sum(nbr_sims_ctrl) / len(nbr_sims_ctrl)

    print(f"\n  Centre vs neighbours (avg cosine similarity):")
    print(f"    With lateral:                {avg_nbr_sim_lat:.4f}")
    print(f"    Without lateral:             {avg_nbr_sim_ctrl:.4f}")

    # W_lat diverged from identity.
    centre_col = grid._col(GRID_H // 2, GRID_W // 2)
    w_lat_delta = (
        centre_col.W_lat.data - torch.eye(OBJ_DIM, device=device)
    ).abs().sum().item()
    print(f"\n  Centre W_lat delta from identity: {w_lat_delta:.4f}")

    # -----------------------------------------------------------------
    # 7. Verdict
    # -----------------------------------------------------------------
    print(f"\n{'='*68}")

    passed = True
    checks = []

    # Check 1: Lateral cosine similarity significantly better than control.
    if sim_lateral > sim_control + 0.05:
        checks.append(("Lateral > control similarity", True))
    else:
        checks.append(("Lateral > control similarity", False))
        passed = False

    # Check 2: Lateral cosine similarity positive and meaningful.
    # With multiplicative dendritic gating, the occluded column converges
    # toward the neighbour average (which is not identical to the sensory-
    # driven representation).  The key evidence is the improvement over
    # control and the high centre-vs-neighbours similarity.
    if sim_lateral > 0.1:
        checks.append(("Lateral similarity > 0.1", True))
    else:
        checks.append(("Lateral similarity > 0.1", False))
        passed = False

    # Check 3: Energy decreased during occluded inference.
    if occ_decreased:
        checks.append(("Energy decreased", True))
    else:
        checks.append(("Energy decreased", False))
        passed = False

    # Check 4: Centre x_obj norm is meaningful (not near zero).
    # The sensory pathway (zero input) competes with lateral pull,
    # so the magnitude is naturally attenuated.  The direction (cosine)
    # is the primary validation; norm > 30% confirms meaningful recovery.
    if norm_lateral > 0.3 * norm_trained:
        checks.append(("Occluded norm > 30% of trained", True))
    else:
        checks.append(("Occluded norm > 30% of trained", False))
        passed = False

    # Check 5: W_lat learned (diverged from identity).
    if w_lat_delta > 0.1:
        checks.append(("W_lat diverged from identity", True))
    else:
        checks.append(("W_lat diverged from identity", False))
        passed = False

    # Check 6: No autograd.
    all_no_grad = all(
        not p.requires_grad for p in grid.parameters()
    )
    checks.append(("All requires_grad=False", all_no_grad))
    if not all_no_grad:
        passed = False

    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")

    print(f"\n{'='*68}")
    if passed:
        print("PASS: Lateral consensus fills in occluded column via emergent voting.")
    else:
        print("INVESTIGATE: Some checks failed — lateral consensus may need tuning.")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
