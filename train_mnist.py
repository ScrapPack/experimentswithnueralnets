"""
MNIST Autoencoder — Phase 4

Trains the Predictive Coding network as a generative autoencoder on
MNIST.  There is no backpropagation anywhere — the network learns to
reconstruct handwritten digits using only:

    1. Local Euler-integration inference  (fast dynamics, ~20 steps)
    2. Local Hebbian weight updates       (slow dynamics, one step)

Architecture:
    [64, 256, 784]  — top → hidden → sensory
    layers[0]: PCLayer(64,  256)  — top, abstract causes
    layers[1]: PCLayer(256, 784)  — bottom, predicts pixel space

Reconstruction is the bottom PCLayer's top-down prediction:
    recon = net.layers[-1].predict_down(net.layers[-1].x)

Hardware: MPS (Apple Silicon) preferred.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — write PNGs only
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pc_layer import _get_device
from pc_network import PCNetwork


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Architecture
LAYER_DIMS: list[int] = [64, 256, 784]
ACTIVATION: str = "tanh"

# Training
BATCH_SIZE: int = 64
N_EPOCHS: int = 3
INFER_STEPS: int = 20
ETA_X: float = 0.1          # inference step size
ETA_W: float = 0.0005       # learning rate
PRINT_EVERY: int = 100      # print energy every N batches

# Visualization
N_VIS_IMAGES: int = 8       # images to show per reconstruction plot
VIS_BATCHES: list[int] = [0]  # extra batch indices to visualise (epoch 0)

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloader() -> DataLoader:
    """Load MNIST, normalise to [-1, 1], flatten to 784-d vectors."""
    transform = transforms.Compose([
        transforms.ToTensor(),                       # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),        # [-1, 1]
        transforms.Lambda(lambda x: x.view(-1)),     # (784,)
    ])
    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,   # keep batch size constant
    )
    return loader


# ---------------------------------------------------------------------------
# Reconstruction extraction
# ---------------------------------------------------------------------------

def get_reconstruction(net: PCNetwork) -> torch.Tensor:
    """Extract the bottom PCLayer's top-down prediction (the reconstruction).

    After inference has settled, the bottom layer's beliefs encode a
    compressed representation of the sensory input.  Its ``predict_down``
    produces the generative model's best guess for what the sensory data
    should look like.

    Returns:
        recon: (batch, 784) — raw reconstruction (in [-1, 1] range once
               the network has learned reasonable weights).
    """
    bottom_layer = net.layers[-1]
    return bottom_layer.predict_down(bottom_layer.x)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_reconstruction_plot(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    epoch: int,
    batch_idx: int,
    energy: float,
    output_dir: Path,
    n_images: int = N_VIS_IMAGES,
) -> Path:
    """Save a side-by-side plot of originals vs. reconstructions.

    Both tensors are (batch, 784) in the normalised [-1, 1] range.
    We denormalise to [0, 1] for display and reshape to 28×28.

    Returns:
        path: Path to the saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_images, originals.shape[0])

    # Denormalise: [-1, 1] → [0, 1] and move to CPU.
    orig = ((originals[:n].detach().cpu() + 1.0) / 2.0).clamp(0, 1)
    recon = ((reconstructions[:n].detach().cpu() + 1.0) / 2.0).clamp(0, 1)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    fig.suptitle(
        f"Epoch {epoch}  |  Batch {batch_idx}  |  Energy {energy:.1f}",
        fontsize=12,
        fontweight="bold",
    )

    for i in range(n):
        # Original
        axes[0, i].imshow(orig[i].view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=9)

        # Reconstruction
        axes[1, i].imshow(recon[i].view(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstruction", fontsize=9)

    plt.tight_layout()
    path = output_dir / f"mnist_reconstruction_epoch_{epoch}_batch_{batch_idx}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    """Full MNIST training loop: infer → learn → visualise."""
    device = _get_device()
    print(f"Device: {device}")

    # --- Build network ---
    net = PCNetwork(layer_dims=LAYER_DIMS, activation_fn_name=ACTIVATION).to(device)
    print(net)
    print(
        f"\nArchitecture : {LAYER_DIMS}"
        f"\nActivation   : {ACTIVATION}"
        f"\nBatch size   : {BATCH_SIZE}"
        f"\nInfer steps  : {INFER_STEPS}"
        f"\neta_x        : {ETA_X}"
        f"\neta_w        : {ETA_W}"
        f"\nEpochs       : {N_EPOCHS}"
    )

    # --- Data ---
    loader = build_dataloader()
    n_batches = len(loader)
    print(f"Batches/epoch: {n_batches}\n")

    # --- Training loop ---
    for epoch in range(N_EPOCHS):
        epoch_energy_sum = 0.0
        epoch_batches = 0
        t0 = time.perf_counter()

        for batch_idx, (images, _labels) in enumerate(loader):
            # images: (B, 784) already flattened and normalised to [-1, 1]
            sensory = images.to(device)

            # Fast dynamics: settle beliefs toward the sensory input.
            history = net.infer(sensory, steps=INFER_STEPS, eta_x=ETA_X)
            final_energy = history[-1]

            # Slow dynamics: update generative weights.
            net.learn(eta_w=ETA_W)

            # Accumulate energy for epoch average.
            epoch_energy_sum += final_energy
            epoch_batches += 1

            # --- Periodic logging ---
            if batch_idx % PRINT_EVERY == 0:
                avg_energy = epoch_energy_sum / epoch_batches
                elapsed = time.perf_counter() - t0
                print(
                    f"  Epoch {epoch} | Batch {batch_idx:4d}/{n_batches} | "
                    f"Energy {final_energy:10.1f} | "
                    f"Avg {avg_energy:10.1f} | "
                    f"{elapsed:5.1f}s"
                )

            # --- Save reconstruction at first batch and end of epoch ---
            is_first_batch_ever = (epoch == 0 and batch_idx == 0)
            is_epoch_end = (batch_idx == n_batches - 1)

            if is_first_batch_ever or is_epoch_end:
                recon = get_reconstruction(net)
                path = save_reconstruction_plot(
                    originals=sensory,
                    reconstructions=recon,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    energy=final_energy,
                    output_dir=OUTPUT_DIR,
                )
                print(f"    -> Saved: {path.name}")

        # --- Epoch summary ---
        epoch_avg = epoch_energy_sum / epoch_batches
        elapsed = time.perf_counter() - t0
        print(
            f"\n  Epoch {epoch} complete | "
            f"Avg energy: {epoch_avg:.1f} | "
            f"Time: {elapsed:.1f}s\n"
            f"  {'─' * 55}"
        )

    print("\nTraining complete.")
    print(f"Reconstruction PNGs saved to: {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
