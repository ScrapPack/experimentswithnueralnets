"""
Test: Lateral Consensus Defeats Occlusion (Sensory Attenuation)

Validates the CorticalGrid's emergent consensus behaviour.  A 3×3 grid
of CorticalColumns is trained on a coherent 9×9 image (a cross / + pattern).
After training, the centre column's sensory precision (pi_sensory) is dropped
to zero — the biological mechanism for occlusion / unreliable input — and
the grid is re-run on the *unmodified* image.

Because pi_sensory = 0, the centre column's sensory error term vanishes
from both the ODE and the energy, leaving *only* lateral consensus (err_lat)
and state decay.  The lateral connections should pull the occluded column's
x_obj toward the consensus of its neighbours, effectively "filling in" the
missing information purely through local ODE dynamics and Hebbian-learned
lateral weights.

This is the biologically correct mechanism: sensory attenuation (precision
weighting), not data manipulation (zeroing the patch).

Training uses strong lateral precision (pi_lat = 6.0) to enforce Thousand
Brains consensus: columns seeing identical patches MUST converge to the
same representation.  Without this, dendritic gating's multiplicative
interaction creates spurious local minima where columns with identical
sensory input diverge based on random initialisation.

Anti-cheat guarantees:
    1. No manual x_obj assignment during inference — states settle via FEP ODEs.
    2. The global grid.infer() integration loop is used, not manual column stepping.
    3. Because pi_sensory = 0.0 the centre column has zero mathematical resistance
       to its neighbours, so cosine similarity to the trained state should be > 0.95.

What to look for:
    1. After training, columns seeing the same patch converge to similar x_obj.
    2. With sensory attenuation, the centre column's x_obj is reconstructed
       from its neighbours — cosine similarity to trained state should be > 0.95.
    3. Energy should decrease during inference even with attenuation.
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
    print("  (pi_lat = 6.0: strong lateral consensus enforces Thousand Brains voting)")

    N_EPOCHS = 400
    INFER_STEPS = 50
    ETA_X = 0.1
    ETA_W = 0.01

    # Strong lateral precision during training.
    # Biologically justified: the Thousand Brains theory requires columns
    # seeing the same object to converge to the same representation.
    # In the cross pattern, the centre and all 4 neighbours see identical
    # all-ones patches.  Without strong lateral coupling, dendritic gating's
    # multiplicative interaction creates spurious local minima where columns
    # with identical input diverge based on random init.
    PI_LAT_TRAIN = 6.0
    for col in grid.columns:
        col.pi_lat = PI_LAT_TRAIN

    with torch.no_grad():
        for epoch in range(N_EPOCHS):
            history = grid.infer(cross, steps=INFER_STEPS, eta_x=ETA_X)
            grid.learn(cross, eta_w=ETA_W)

            if epoch % 30 == 0 or epoch == N_EPOCHS - 1:
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

    # Show similarity of centre to each neighbour.
    nbr_idxs = grid._neighbor_idx[centre_idx]
    for n in nbr_idxs:
        sim = cosine_sim(centre_trained, trained_x_objs[n])
        print(f"    vs col{n} (neighbour): {sim:.4f}")

    # -----------------------------------------------------------------
    # 4. Test: sensory attenuation on centre column and re-infer
    # -----------------------------------------------------------------
    print("\nPhase 2: Sensory attenuation — centre column pi_sensory = 0.0")
    print("  (Full cross image passed; centre column ignores sensory input)")

    # Biological occlusion: drop sensory precision to zero.
    # The column's ODE term -(pi_sensory * grad) becomes zero, leaving
    # only lateral consensus (err_lat) and state decay (-alpha * x) active.
    # No data manipulation — the raw cross image is passed unchanged.
    #
    # ODE equilibrium for the attenuated column:
    #   0 = -pi_lat * (x - W_lat @ nbr_avg) - alpha * x
    #   x* = (pi_lat / (pi_lat + alpha)) * W_lat @ nbr_avg
    grid.columns[centre_idx].pi_sensory = 0.0

    # Same seed as capture run → same random init in reset_states().
    torch.manual_seed(999)
    with torch.no_grad():
        occ_history = grid.infer(cross, steps=INFER_STEPS * 4, eta_x=ETA_X)

    # Restore sensory precision for future use.
    grid.columns[centre_idx].pi_sensory = 1.0

    occ_energy = occ_history[-1]
    occluded_x_objs = [col.x_obj.clone() for col in grid.columns]
    centre_occluded = occluded_x_objs[centre_idx]

    # -----------------------------------------------------------------
    # 5. Control: truly isolated columns (no lateral connections)
    # -----------------------------------------------------------------
    print("\nPhase 3: Control — same attenuation, columns settle INDEPENDENTLY")

    # Run each column individually WITHOUT neighbor_context.
    # This is the proper ablation: the centre column has pi_sensory=0.0
    # and NO lateral connections, so it receives zero information and
    # should decay to near-zero via the state decay prior.
    B = 1

    # Slice the FULL cross image the same way CorticalGrid does.
    full_img = cross.reshape(B, GRID_H * PATCH_H, GRID_W * PATCH_W)
    ctrl_patches: list[torch.Tensor] = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            y0 = r * PATCH_H
            x0 = c * PATCH_W
            patch = full_img[:, y0:y0 + PATCH_H, x0:x0 + PATCH_W]
            ctrl_patches.append(patch.reshape(B, grid.sensory_dim))

    # Build isolated columns with the SAME trained weights.
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
        # Match training lateral precision.
        iso.pi_lat = PI_LAT_TRAIN
        isolated_cols.append(iso)

    # Match the occlusion condition: centre column has pi_sensory = 0.0.
    isolated_cols[centre_idx].pi_sensory = 0.0

    # Settle each isolated column independently — NO neighbor_context.
    # Same seed as capture run → same random init for fair comparison.
    torch.manual_seed(999)
    with torch.no_grad():
        for iso in isolated_cols:
            iso.reset_states(B, device)
        for _ in range(INFER_STEPS * 4):
            for iso, patch in zip(isolated_cols, ctrl_patches):
                iso.infer_step(patch, eta_x=ETA_X)  # no neighbor_context!

    ctrl_energy = sum(iso.get_energy() for iso in isolated_cols)
    control_x_objs = [iso.x_obj.clone() for iso in isolated_cols]
    centre_control = control_x_objs[centre_idx]

    # Restore for cleanliness.
    isolated_cols[centre_idx].pi_sensory = 1.0

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

    # Lateral equilibrium target: theoretical convergence point.
    # At equilibrium: x* = (pi_lat / (pi_lat + alpha)) * W_lat @ nbr_avg.
    centre_col = grid._col(GRID_H // 2, GRID_W // 2)
    nbr_avg_settled = torch.stack(
        [occluded_x_objs[n] for n in nbr_idxs], dim=0
    ).mean(dim=0)  # (1, obj_dim)
    lateral_target = (
        PI_LAT_TRAIN / (PI_LAT_TRAIN + centre_col.alpha)
    ) * (nbr_avg_settled @ centre_col.W_lat.t())
    sim_to_target = cosine_sim(centre_occluded, lateral_target)
    print(f"\n  Centre cosine to lateral equilibrium target: {sim_to_target:.4f}")
    print(f"    (validates ODE convergence — should be near 1.0)")

    # Norm comparison: does the occluded column develop a meaningful representation?
    norm_lateral = centre_occluded.norm().item()
    norm_control = centre_control.norm().item()
    norm_trained = centre_trained.norm().item()
    print(f"\n  Centre column x_obj norm:")
    print(f"    Trained (full info):         {norm_trained:.4f}")
    print(f"    Attenuated + lateral:        {norm_lateral:.4f}")
    print(f"    Attenuated + no lateral:     {norm_control:.4f}")

    # Energy comparison.
    print(f"\n  Final inference energy:")
    print(f"    Attenuated + lateral:        {occ_energy:.4f}")
    print(f"    Attenuated + no lateral:     {ctrl_energy:.4f}")

    # Energy decrease check.
    occ_decreased = occ_history[-1] < occ_history[0]
    print(f"\n  Energy decreased (lateral):    "
          f"{'YES' if occ_decreased else 'NO'}  "
          f"({occ_history[0]:.2f} → {occ_history[-1]:.2f})")

    # Neighbour similarity: occluded centre vs its 4 neighbours.
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

    # Check 2: Lateral cosine similarity near-perfect.
    # With pi_sensory=0 and strong lateral coupling, the centre column has
    # zero resistance to its neighbours.  Because the centre and all 4
    # neighbours see identical all-ones patches, the trained representations
    # should be similar, and lateral consensus should reconstruct > 0.95.
    if sim_lateral > 0.95:
        checks.append(("Lateral similarity > 0.95 (near-perfect reconstruction)", True))
    else:
        checks.append(("Lateral similarity > 0.95 (near-perfect reconstruction)", False))
        passed = False

    # Check 3: ODE convergence — centre reached lateral equilibrium target.
    if sim_to_target > 0.95:
        checks.append(("ODE converged to lateral equilibrium (> 0.95)", True))
    else:
        checks.append(("ODE converged to lateral equilibrium (> 0.95)", False))
        passed = False

    # Check 4: Energy decreased during attenuated inference.
    if occ_decreased:
        checks.append(("Energy decreased", True))
    else:
        checks.append(("Energy decreased", False))
        passed = False

    # Check 5: Centre x_obj norm is meaningful (not near zero).
    # With pi_sensory=0, the column gets all signal from lateral consensus.
    # Norm should be close to the trained norm.
    if norm_lateral > 0.5 * norm_trained:
        checks.append(("Attenuated norm > 50% of trained", True))
    else:
        checks.append(("Attenuated norm > 50% of trained", False))
        passed = False

    # Check 6: W_lat learned (diverged from identity).
    if w_lat_delta > 0.1:
        checks.append(("W_lat diverged from identity", True))
    else:
        checks.append(("W_lat diverged from identity", False))
        passed = False

    # Check 7: Control centre column decayed (near-zero norm) — proves
    # lateral consensus is the sole source of reconstruction.
    if norm_control < 0.3 * norm_trained:
        checks.append(("Control centre decayed (no info without lateral)", True))
    else:
        checks.append(("Control centre decayed (no info without lateral)", False))
        passed = False

    # Check 8: No autograd.
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
        print("PASS: Lateral consensus fills in attenuated column via sensory precision.")
    else:
        print("INVESTIGATE: Some checks failed — lateral consensus may need tuning.")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
