"""
Test: Hippocampal Replay prevents Catastrophic Interference (Phase 5)

Demonstrates that continuous online Hebbian updates cause sequential
bias (the network reconstructs the most recent data much better than
earlier data), while episodic replay (consolidation on shuffled
memories) achieves balanced learning across all experiences.

Biological predictive coding models are more resistant to catastrophic
forgetting than standard neural networks because latent states
(x_obj, x_loc) re-settle per input.  However, without replay, the
weights still accumulate sequential bias — the generative model tilts
toward whatever was trained last.  Hippocampal replay corrects this
by replaying shuffled memories during consolidation, distributing
weight updates across the full experiential history.

Protocol:
  1. Create two multi-pattern datasets of random images.
     With a bottleneck latent (obj_dim << sensory_dim), the generative
     weights must compress, creating genuine competition for capacity.

  2. Online Bias Test:
       - Train on A briefly, then train on B extensively.
       - Show B reconstruction far exceeds A (sequential recency bias).

  3. Consolidation Balance Test:
       - Store all A and B patterns in the episodic buffer.
       - Consolidate (shuffled replay) for equivalent gradient steps.
       - Show replay achieves balanced A/B performance that exceeds
         online's worst-case (A) performance.

No autograd.  No backpropagation.  All dynamics are local FEP ODEs.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pc_layer import _get_device
from pc_network import CorticalGrid

# ======================================================================
# Constants
# ======================================================================

GRID_H, GRID_W = 3, 3
PATCH_H, PATCH_W = 3, 3
OBJ_DIM = 3              # severe bottleneck (sensory_dim=9, 3x compression)
LOC_DIM = 2
IMG_H = GRID_H * PATCH_H   # 9
IMG_W = GRID_W * PATCH_W    # 9
IMG_DIM = IMG_H * IMG_W     # 81

N_PATTERNS = 10             # patterns per class
N_A_EPOCHS = 10             # brief training on A
N_B_EPOCHS = 60             # extensive training on B (creates bias)
INFER_STEPS = 50            # ODE settling steps
ETA_X = 0.05                # belief update rate
ETA_W = 0.02                # aggressive weight learning rate

# Consolidation: total learn calls ≈ online total
# Online: (10 + 60) * 10 = 700 learn calls
CONSOL_ROUNDS = 700         # same total learn calls as online
CONSOL_BATCH = 10           # sample from 20 unique patterns

SEED = 42                   # reproducible weight init
DATA_SEED = 100             # reproducible dataset generation


# ======================================================================
# Dataset creation
# ======================================================================

def make_random_dataset(
    n_patterns: int,
    device: torch.device,
    seed: int,
) -> list[Tensor]:
    """Create a dataset of n_patterns random images.

    Random images have maximum entropy and no shared structure between
    classes, forcing the generative weights to specialise.  With a
    bottleneck latent (obj_dim << sensory_dim), the weights must
    compress each class into its own low-rank subspace.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    patterns = []
    for _ in range(n_patterns):
        img = torch.randn(1, IMG_DIM, generator=rng)
        patterns.append(img.to(device))
    return patterns


# ======================================================================
# Measurement helper
# ======================================================================

def measure_dataset_reconstruction(
    grid: CorticalGrid,
    dataset: list[Tensor],
) -> float:
    """Average cosine similarity across all patterns in a dataset."""
    total_sim = 0.0
    for stimulus in dataset:
        grid.infer(stimulus, steps=INFER_STEPS, eta_x=ETA_X)
        patches = grid._slice_patches(stimulus)
        col_sims = []
        for i, col in enumerate(grid.columns):
            pred = col.predict_down()[0]
            cos = torch.nn.functional.cosine_similarity(
                pred, patches[i], dim=-1,
            )
            col_sims.append(cos.mean().item())
        total_sim += sum(col_sims) / len(col_sims)
    return total_sim / len(dataset)


# ======================================================================
# Grid factory (deterministic init)
# ======================================================================

def make_grid(device: torch.device) -> CorticalGrid:
    """Create a fresh CorticalGrid with deterministic weight init."""
    torch.manual_seed(SEED)
    grid = CorticalGrid(
        grid_h=GRID_H, grid_w=GRID_W,
        obj_dim=OBJ_DIM, loc_dim=LOC_DIM,
        patch_h=PATCH_H, patch_w=PATCH_W,
        activation_fn_name="tanh",
        buffer_capacity=10_000,
    ).to(device)
    return grid


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    device = _get_device()
    print(f"Using device: {device}\n")

    dataset_A = make_random_dataset(N_PATTERNS, device, DATA_SEED)
    dataset_B = make_random_dataset(N_PATTERNS, device, DATA_SEED + 1000)

    # Measure untrained baseline.
    grid_baseline = make_grid(device)
    baseline_sim = measure_dataset_reconstruction(grid_baseline, dataset_A)
    del grid_baseline

    print(f"  Patterns per class: {N_PATTERNS}")
    print(f"  Bottleneck: obj={OBJ_DIM}, loc={LOC_DIM}, sensory={PATCH_H*PATCH_W}")
    print(f"  Untrained baseline: {baseline_sim:.4f}")
    print(f"  Online: {N_A_EPOCHS} A epochs + {N_B_EPOCHS} B epochs "
          f"= {(N_A_EPOCHS + N_B_EPOCHS) * N_PATTERNS} total steps")
    print(f"  Replay: {CONSOL_ROUNDS} rounds x batch {CONSOL_BATCH}")
    print()

    # ==================================================================
    # Test 1: Online Sequential Bias
    # ==================================================================
    print("=" * 60)
    print("Test 1: Online Sequential Bias")
    print("=" * 60)

    grid_online = make_grid(device)

    # Brief training on A.
    print(f"\n  Training on A ({N_A_EPOCHS} epochs)...")
    for epoch in range(N_A_EPOCHS):
        for frame in dataset_A:
            grid_online.infer(frame, steps=INFER_STEPS, eta_x=ETA_X,
                              online_learning=True, eta_w=ETA_W)

    sim_A_after_A = measure_dataset_reconstruction(grid_online, dataset_A)
    print(f"  A after training A: {sim_A_after_A:.4f}")

    # Extensive training on B (6x longer → strong recency bias).
    print(f"  Training on B ({N_B_EPOCHS} epochs)...")
    for epoch in range(N_B_EPOCHS):
        for frame in dataset_B:
            grid_online.infer(frame, steps=INFER_STEPS, eta_x=ETA_X,
                              online_learning=True, eta_w=ETA_W)

    sim_A_online = measure_dataset_reconstruction(grid_online, dataset_A)
    sim_B_online = measure_dataset_reconstruction(grid_online, dataset_B)
    recency_bias = sim_B_online - sim_A_online
    print(f"  A after training B: {sim_A_online:.4f}")
    print(f"  B after training B: {sim_B_online:.4f}")
    print(f"  Recency bias (B-A): {recency_bias:.4f}")

    # ==================================================================
    # Test 2: Consolidation (Balanced Replay)
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("Test 2: Consolidation (Hippocampal Replay)")
    print("=" * 60)

    grid_replay = make_grid(device)

    # Store all patterns (no learning).
    print(f"\n  Storing all {N_PATTERNS * 2} patterns...")
    for frame in dataset_A:
        grid_replay.infer(frame, steps=INFER_STEPS, eta_x=ETA_X,
                          online_learning=False)
    for frame in dataset_B:
        grid_replay.infer(frame, steps=INFER_STEPS, eta_x=ETA_X,
                          online_learning=False)

    buf_len = len(grid_replay.buffer)
    print(f"  Buffer: {buf_len} frames")

    # Consolidate with shuffled replay.
    print(f"  Consolidating ({CONSOL_ROUNDS} rounds)...")
    for r in range(CONSOL_ROUNDS):
        grid_replay.consolidate(
            batch_size=CONSOL_BATCH,
            steps=INFER_STEPS,
            eta_x=ETA_X,
            eta_w=ETA_W,
        )

    sim_A_replay = measure_dataset_reconstruction(grid_replay, dataset_A)
    sim_B_replay = measure_dataset_reconstruction(grid_replay, dataset_B)
    replay_bias = abs(sim_A_replay - sim_B_replay)
    print(f"  A after consolidation: {sim_A_replay:.4f}")
    print(f"  B after consolidation: {sim_B_replay:.4f}")
    print(f"  |A-B| imbalance:       {replay_bias:.4f}")

    # ==================================================================
    # Checks
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print("=" * 60)

    print(f"\n  Baseline (untrained): {baseline_sim:.4f}")
    print(f"  Online:  A={sim_A_online:.4f}  B={sim_B_online:.4f}  "
          f"bias={recency_bias:.4f}")
    print(f"  Replay:  A={sim_A_replay:.4f}  B={sim_B_replay:.4f}  "
          f"bias={replay_bias:.4f}")
    print()

    online_min = min(sim_A_online, sim_B_online)
    replay_min = min(sim_A_replay, sim_B_replay)

    checks = []

    # Check 1: Training improved from baseline.
    ok = sim_A_after_A > baseline_sim + 0.02
    checks.append((
        "Training A improved from baseline",
        ok,
        f"{sim_A_after_A:.4f} > {baseline_sim:.4f}+0.02",
    ))

    # Check 2: Online shows sequential bias (B > A).
    ok = recency_bias > 0.02
    checks.append((
        "Online recency bias (B > A)",
        ok,
        f"B-A = {recency_bias:.4f} > 0.02",
    ))

    # Check 3: Replay is more balanced than online.
    ok = replay_bias < recency_bias
    checks.append((
        "Replay more balanced than online",
        ok,
        f"|A-B|_replay={replay_bias:.4f} < bias_online={recency_bias:.4f}",
    ))

    # Check 4: Replay worst-case >= online worst-case.
    ok = replay_min >= online_min - 0.02
    checks.append((
        "Replay worst-case >= online worst-case",
        ok,
        f"min_replay={replay_min:.4f} >= min_online={online_min:.4f}-0.02",
    ))

    # Check 5: Both replay classes above baseline.
    ok = sim_A_replay > baseline_sim and sim_B_replay > baseline_sim
    checks.append((
        "Replay A and B both above baseline",
        ok,
        f"A={sim_A_replay:.4f}, B={sim_B_replay:.4f} > {baseline_sim:.4f}",
    ))

    # Check 6: Buffer size correct.
    ok = buf_len == N_PATTERNS * 2
    checks.append(("Buffer size correct", ok, buf_len))

    # Check 7: No autograd.
    no_grad = all(
        not p.requires_grad
        for p in list(grid_online.parameters()) + list(grid_replay.parameters())
    )
    checks.append(("All requires_grad=False", no_grad, ""))

    print("  " + "-" * 60)
    all_pass = True
    for label, passed, val in checks:
        tag = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{tag}] {label}")
        print(f"         ({val})")
    print("  " + "-" * 60)

    if all_pass:
        print(f"\n{'=' * 60}")
        print("PASS: Hippocampal replay prevents sequential bias.")
        print("=" * 60)
    else:
        print(f"\n{'=' * 60}")
        print("FAIL: Some checks did not pass — see above.")
        print("=" * 60)
        exit(1)
