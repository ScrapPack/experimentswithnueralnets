"""
Test: Temporal Predictive Coding — The Falling Block

Validates Phase 4's temporal prediction by training a single CorticalColumn
on a three-frame sequence of a block falling down a 5×5 grid.  After
training, the column is asked to "dream" the next two frames given only
the first frame's settled belief — with ZERO sensory input.

The column remembers its previous belief (x_obj_prev), and its learned
transition matrix (W_trans) predicts how beliefs evolve over time.
This enables generative "physics" — the column hallucinates the block
falling purely through its internal temporal dynamics.

What to look for:
    1. After training, each frame produces a distinct x_obj representation.
    2. Dream 1 (from Frame 1 memory): x_obj settles near Frame 2's concept.
    3. Dream 2 (from Dream 1 memory): x_obj settles near Frame 3's concept.
    4. Dreams are more similar to the correct next frame than to other frames.
    5. W_trans diverged from identity (learned transition dynamics).
    6. Energy decreases during inference.
    7. No autograd.  All dynamics are local ODEs + Hebbian learning.
"""

from __future__ import annotations

import torch
from pc_layer import CorticalColumn, _get_device


def make_falling_frames(device: torch.device) -> list[torch.Tensor]:
    """Create 3 frames of a block falling in a 5×5 grid.

    Each frame is (1, 25) — a flattened 5×5 image.
    The block is a horizontal bar (full row) that falls:
        Frame 0: bar at row 0 (top)
        Frame 1: bar at row 2 (middle)
        Frame 2: bar at row 4 (bottom)
    """
    frames = []
    for row_idx in [0, 2, 4]:
        img = torch.zeros(5, 5, device=device)
        img[row_idx, :] = 1.0
        frames.append(img.reshape(1, 25))
    return frames


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two (1, D) tensors."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    dot = (a_flat * b_flat).sum()
    denom = a_flat.norm() * b_flat.norm() + 1e-8
    return (dot / denom).item()


def main() -> None:
    torch.manual_seed(42)

    device = _get_device()
    print(f"Device: {device}\n")

    # -----------------------------------------------------------------
    # 1. Build the CorticalColumn
    # -----------------------------------------------------------------
    OBJ_DIM = 100   # overcomplete (4x for sensory_dim=25)
    LOC_DIM = 4
    SENSORY_DIM = 25   # 5×5 grid

    col = CorticalColumn(
        obj_dim=OBJ_DIM, loc_dim=LOC_DIM,
        sensory_dim=SENSORY_DIM,
        activation_fn_name="tanh",
    ).to(device)

    print(col)
    print()

    # -----------------------------------------------------------------
    # 2. Build the falling-block frames
    # -----------------------------------------------------------------
    frames = make_falling_frames(device)

    print("Falling block sequence (5×5 each):")
    for i, frame in enumerate(frames):
        img = frame.reshape(5, 5)
        row_str = "  "
        for r in range(5):
            for c in range(5):
                row_str += "██" if img[r, c].item() > 0.5 else "  "
            if r < 4:
                row_str += "\n  "
        print(f"  Frame {i}:")
        print(row_str)
        print()

    # -----------------------------------------------------------------
    # 3. Training: teach the column the falling sequence
    # -----------------------------------------------------------------
    print("Phase 1: Training on falling block sequence...")

    N_EPOCHS = 300
    SETTLE_STEPS = 50
    ETA_X = 0.1
    ETA_W = 0.01
    B = 1

    with torch.no_grad():
        for epoch in range(N_EPOCHS):
            # Fresh sequence start.
            col.x_obj_prev = None

            for frame_idx, frame in enumerate(frames):
                # Reset fast states for this frame (x_obj_prev persists).
                col.reset_states(B, device)

                # Settle on this frame.
                for step in range(SETTLE_STEPS):
                    col.infer_step(frame, eta_x=ETA_X)

                # Learn from settled state.
                col.learn(eta_w=ETA_W)

                # Tick the clock: x_obj → x_obj_prev.
                col.step_time()

            if epoch % 40 == 0 or epoch == N_EPOCHS - 1:
                # Report energy on last frame.
                print(f"  epoch {epoch:3d}  energy={col.get_energy():.4f}")

    # -----------------------------------------------------------------
    # 4. Capture trained representations for each frame
    # -----------------------------------------------------------------
    print("\nCapturing trained frame representations...")

    trained_x_objs: list[torch.Tensor] = []

    with torch.no_grad():
        col.x_obj_prev = None

        for frame_idx, frame in enumerate(frames):
            col.reset_states(B, device)

            for step in range(SETTLE_STEPS):
                col.infer_step(frame, eta_x=ETA_X)

            trained_x_objs.append(col.x_obj.clone())
            col.step_time()

    print("  Frame x_obj norms:")
    for i, x in enumerate(trained_x_objs):
        print(f"    Frame {i}: {x.norm().item():.4f}")

    # Show pairwise similarity of trained frame representations.
    print("\n  Pairwise cosine similarity of trained frame x_obj:")
    print("          ", end="")
    for i in range(3):
        print(f"  F{i}   ", end="")
    print()
    for i in range(3):
        print(f"    F{i}  ", end="")
        for j in range(3):
            sim = cosine_sim(trained_x_objs[i], trained_x_objs[j])
            print(f"  {sim:5.2f}", end="")
        print()

    # -----------------------------------------------------------------
    # 5. Dreaming: generative physics with ZERO sensory input
    # -----------------------------------------------------------------
    print("\nPhase 2: Dreaming — generating future from memory alone")

    blank = torch.zeros(1, SENSORY_DIM, device=device)
    DREAM_SETTLE = 150  # more steps for dreaming (weaker signal)

    dream_x_objs: list[torch.Tensor] = []
    dream_predictions: list[torch.Tensor] = []

    with torch.no_grad():
        # Inject Frame 0's trained belief into temporal memory.
        col.x_obj_prev = trained_x_objs[0].clone()

        for dream_idx in range(2):
            # Reset fast states (x_obj_prev persists).
            col.reset_states(B, device)

            # Settle with BLANK input — only temporal pressure drives x_obj.
            energies = []
            for step in range(DREAM_SETTLE):
                col.infer_step(blank, eta_x=ETA_X)
                energies.append(col.get_energy())

            dream_x_objs.append(col.x_obj.clone())
            dream_predictions.append(col.predict_down()[0].clone())

            print(f"\n  Dream {dream_idx + 1} (from Frame {dream_idx} memory):")
            print(f"    Energy: {energies[0]:.4f} → {energies[-1]:.4f}")
            print(f"    x_obj norm: {col.x_obj.norm().item():.4f}")

            # Show the dreamed prediction as a 5×5 image.
            pred_img = col.predict_down()[0].reshape(5, 5)
            print("    Predicted image (5×5):")
            for r in range(5):
                row_str = "      "
                for c in range(5):
                    val = pred_img[r, c].item()
                    if val > 0.5:
                        row_str += "██"
                    elif val > 0.2:
                        row_str += "▒▒"
                    elif val > 0.05:
                        row_str += "░░"
                    else:
                        row_str += "  "
                print(row_str)

            # Tick the clock: dreamed x_obj becomes memory for next dream.
            col.step_time()

    # -----------------------------------------------------------------
    # 6. Analysis
    # -----------------------------------------------------------------
    print(f"\n{'='*68}")
    print("ANALYSIS")
    print(f"{'='*68}")

    # Dream 1 should be closest to Frame 1, Dream 2 to Frame 2.
    print("\n  Dream 1 cosine similarity to each trained frame:")
    dream1_sims = []
    for i in range(3):
        sim = cosine_sim(dream_x_objs[0], trained_x_objs[i])
        dream1_sims.append(sim)
        marker = " ← target" if i == 1 else ""
        print(f"    vs Frame {i}: {sim:.4f}{marker}")

    print("\n  Dream 2 cosine similarity to each trained frame:")
    dream2_sims = []
    for i in range(3):
        sim = cosine_sim(dream_x_objs[1], trained_x_objs[i])
        dream2_sims.append(sim)
        marker = " ← target" if i == 2 else ""
        print(f"    vs Frame {i}: {sim:.4f}{marker}")

    # W_trans divergence from identity.
    w_trans_delta = (
        col.W_trans.data - torch.eye(OBJ_DIM, device=device)
    ).abs().sum().item()
    print(f"\n  W_trans delta from identity: {w_trans_delta:.4f}")

    # -----------------------------------------------------------------
    # 7. Verdict
    # -----------------------------------------------------------------
    print(f"\n{'='*68}")

    passed = True
    checks = []

    # Check 1: Dream 1 is more similar to Frame 1 than to Frame 0.
    if dream1_sims[1] > dream1_sims[0]:
        checks.append(("Dream 1 closer to Frame 1 than Frame 0", True))
    else:
        checks.append(("Dream 1 closer to Frame 1 than Frame 0", False))
        passed = False

    # Check 2: Dream 2 is more similar to Frame 2 than to Frame 0.
    if dream2_sims[2] > dream2_sims[0]:
        checks.append(("Dream 2 closer to Frame 2 than Frame 0", True))
    else:
        checks.append(("Dream 2 closer to Frame 2 than Frame 0", False))
        passed = False

    # Check 3: Dream 1 cosine similarity to Frame 1 is reasonably high.
    if dream1_sims[1] > 0.3:
        checks.append(("Dream 1 similarity to Frame 1 > 0.3", True))
    else:
        checks.append(("Dream 1 similarity to Frame 1 > 0.3", False))
        passed = False

    # Check 4: Dream 2 cosine similarity to Frame 2 is reasonably high.
    if dream2_sims[2] > 0.3:
        checks.append(("Dream 2 similarity to Frame 2 > 0.3", True))
    else:
        checks.append(("Dream 2 similarity to Frame 2 > 0.3", False))
        passed = False

    # Check 5: W_trans learned (diverged from identity).
    if w_trans_delta > 0.1:
        checks.append(("W_trans diverged from identity", True))
    else:
        checks.append(("W_trans diverged from identity", False))
        passed = False

    # Check 6: Dream x_obj norms are meaningful (not near zero).
    dream_norms_ok = all(
        d.norm().item() > 0.1 for d in dream_x_objs
    )
    checks.append(("Dream x_obj norms > 0.1", dream_norms_ok))
    if not dream_norms_ok:
        passed = False

    # Check 7: No autograd.
    all_no_grad = all(
        not p.requires_grad for p in col.parameters()
    )
    checks.append(("All requires_grad=False", all_no_grad))
    if not all_no_grad:
        passed = False

    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")

    print(f"\n{'='*68}")
    if passed:
        print("PASS: Column dreams the falling block — temporal prediction works.")
    else:
        print("INVESTIGATE: Some checks failed — temporal prediction may need tuning.")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
