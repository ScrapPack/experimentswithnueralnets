"""
Predictive Coding Network — Cortical Column Edition

A network of CorticalColumn units.  Each column separates object identity
("What" / L2/3) from sensor location ("Where" / L6) and produces motor
output ("Action" / L5) via Active Inference.

The network orchestrates inference across one or more columns:
    1. Reset states for a new observation.
    2. Euler-integrate the dual-state ODE (x_obj, x_loc) to settle.
    3. Optionally compute motor actions.
    4. Hebbian learning after settling.

No autograd. No backpropagation. All dynamics are local ODEs
minimising Free Energy (prediction error).

Hardware: defaults to MPS (Apple Silicon) when available.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from pc_layer import CorticalColumn, _get_device


class PCNetwork(nn.Module):
    """A network of CorticalColumn units.

    Currently supports a single column (Phase 1).  The architecture will
    scale to a grid of columns for Thousand Brains voting in later phases.

    Args:
        obj_dim:     Dimensionality of the "What" state per column.
        loc_dim:     Dimensionality of the "Where" state per column.
        sensory_dim: Dimensionality of the sensory input per column.
        activation_fn_name: Activation for the generative model.
    """

    def __init__(
        self,
        obj_dim: int = 16,
        loc_dim: int = 8,
        sensory_dim: int = 25,
        activation_fn_name: str = "tanh",
    ) -> None:
        super().__init__()

        self.obj_dim = obj_dim
        self.loc_dim = loc_dim
        self.sensory_dim = sensory_dim

        # Phase 1: single column.  Later phases will add a grid.
        self.column = CorticalColumn(
            obj_dim=obj_dim,
            loc_dim=loc_dim,
            sensory_dim=sensory_dim,
            activation_fn_name=activation_fn_name,
        )

        # Energy trace from the most recent infer() call.
        self.energy_history: list[float] = []

    # ------------------------------------------------------------------
    # Inference (settling / free-energy minimisation)
    # ------------------------------------------------------------------

    def infer(
        self,
        sensory_data: Tensor,
        steps: int = 50,
        eta_x: float = 0.05,
    ) -> list[float]:
        """Run the dual-state settling loop.

        Clamps sensory_data and lets x_obj and x_loc evolve to minimise
        the L4 prediction error via Euler integration.

        Args:
            sensory_data: (batch, sensory_dim) — observed data.
            steps:        Number of Euler integration steps.
            eta_x:        Step size for belief updates.

        Returns:
            energy_history: List of energy values per step.
        """
        batch_size = sensory_data.shape[0]
        device = sensory_data.device

        self.column.reset_states(batch_size, device)
        self.energy_history = []

        for _ in range(steps):
            self.column.infer_step(sensory_data, eta_x=eta_x)
            self.energy_history.append(self.column.get_energy())

        return self.energy_history

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_total_energy(self) -> float:
        """Return the current total free energy."""
        return self.column.get_energy()

    # ------------------------------------------------------------------
    # Learning (slow dynamics)
    # ------------------------------------------------------------------

    def learn(self, eta_w: float = 0.001) -> None:
        """Hebbian update for all column weights.

        Must be called after infer() has settled.

        Args:
            eta_w: Learning rate for weight updates.
        """
        self.column.learn(eta_w=eta_w)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"obj_dim={self.obj_dim}, loc_dim={self.loc_dim}, "
            f"sensory_dim={self.sensory_dim}"
        )


# ---------------------------------------------------------------------------
# Smoke test — infer + learn loop, verify energy drops
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = _get_device()
    print(f"Using device: {device}\n")

    net = PCNetwork(
        obj_dim=16, loc_dim=8, sensory_dim=25, activation_fn_name="tanh",
    ).to(device)
    print(net)
    print()

    B = 4
    sensory = torch.randn(B, 25, device=device)

    # --- Training loop ---
    n_epochs = 10
    infer_steps = 30
    eta_x = 0.1
    eta_w = 0.01

    w_obj_before = net.column.W_obj.data.clone()
    all_energies: list[float] = []

    print(f"  Config: {n_epochs} epochs x {infer_steps} infer steps")
    print(f"    eta_x={eta_x}, eta_w={eta_w}\n")

    for epoch in range(n_epochs):
        history = net.infer(sensory, steps=infer_steps, eta_x=eta_x)
        all_energies.extend(history)
        net.learn(eta_w=eta_w)

        if epoch == 0 or epoch == n_epochs - 1:
            print(
                f"  Epoch {epoch:2d}  "
                f"start={history[0]:.4f}  end={history[-1]:.4f}"
            )

    # --- Assertions ---
    print()

    start_e = all_energies[0]
    final_e = all_energies[-1]
    assert final_e < start_e, (
        f"Energy did not decrease! start={start_e:.4f} end={final_e:.4f}"
    )
    pct = (start_e - final_e) / abs(start_e) * 100
    print(f"  Start energy : {start_e:.4f}")
    print(f"  Final energy : {final_e:.4f}")
    print(f"  Reduction    : {pct:.1f}%")
    print("  Energy decreased  [OK]")

    w_obj_delta = (net.column.W_obj.data - w_obj_before).abs().sum().item()
    assert w_obj_delta > 0, "W_obj did not change!"
    print(f"  W_obj delta  : {w_obj_delta:.6f}  [OK]")

    for name, param in net.named_parameters():
        assert not param.requires_grad, f"{name} has requires_grad=True!"
    print("  All parameters requires_grad=False  [OK]")

    print("\nPCNetwork smoke test passed.")
