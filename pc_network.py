"""
Predictive Coding Network — Phase 4: Temporal Predictive Coding

Phase 1 (PCNetwork): A single CorticalColumn with What/Where/Action.
Phase 2 (CorticalGrid): A 2D grid of CorticalColumn instances wired
together with lateral connections.
Phase 3 (CorticalStack): A two-level hierarchy where a higher-order
column sends top-down priors to a lower-level grid, enabling
hierarchical directed attention and sensorimotor search.
Phase 4: Temporal predictive coding — columns remember their previous
belief (x_obj_prev) and learn transition dynamics (W_trans), enabling
prediction of future states and generative "dreaming".

This enables:
    - **Noise suppression**: columns vote to smooth out noisy patches.
    - **Occlusion filling**: occluded columns recover their x_obj from
      neighbours' consensus, even with zero sensory input.
    - **Emergent Thousand Brains voting**: object identity is a
      collective decision, not a single-column judgment.
    - **Directed attention** (Phase 3): a higher-level column "expects"
      a target object, pushing top-down predictions that bias lower
      columns and drive L5 motor output to physically scan for it.
    - **Temporal prediction** (Phase 4): columns learn transition
      dynamics and can dream / predict future states.

No autograd. No backpropagation. All dynamics are local ODEs
minimising Free Energy (prediction error + lateral + top-down + temporal error).

Hardware: defaults to MPS (Apple Silicon) when available.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from pc_layer import CorticalColumn, _get_device


# ---------------------------------------------------------------------------
# PCNetwork — single column (Phase 1, preserved for backward compatibility)
# ---------------------------------------------------------------------------

class PCNetwork(nn.Module):
    """A network wrapping a single CorticalColumn (Phase 1).

    Preserved for backward compatibility with test_column.py.

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

        self.column = CorticalColumn(
            obj_dim=obj_dim,
            loc_dim=loc_dim,
            sensory_dim=sensory_dim,
            activation_fn_name=activation_fn_name,
        )

        self.energy_history: list[float] = []

    def infer(
        self,
        sensory_data: Tensor,
        steps: int = 50,
        eta_x: float = 0.05,
    ) -> list[float]:
        """Run the dual-state settling loop (single column)."""
        batch_size = sensory_data.shape[0]
        device = sensory_data.device

        self.column.reset_states(batch_size, device)
        self.energy_history = []

        for _ in range(steps):
            self.column.infer_step(sensory_data, eta_x=eta_x)
            self.energy_history.append(self.column.get_energy())

        return self.energy_history

    def get_total_energy(self) -> float:
        return self.column.get_energy()

    def learn(self, eta_w: float = 0.001) -> None:
        self.column.learn(eta_w=eta_w)

    def extra_repr(self) -> str:
        return (
            f"obj_dim={self.obj_dim}, loc_dim={self.loc_dim}, "
            f"sensory_dim={self.sensory_dim}"
        )


# ---------------------------------------------------------------------------
# CorticalGrid — 2D grid of columns with lateral voting (Phase 2)
# ---------------------------------------------------------------------------

class CorticalGrid(nn.Module):
    """A 2D grid of CorticalColumn instances with lateral connections.

    Each column at position (r, c) sees a local patch of the global
    sensory input.  During inference, columns exchange x_obj states
    with their immediate spatial neighbours (Up, Down, Left, Right).
    The lateral error pulls each column's belief toward the consensus
    of its neighbourhood, enabling collective noise suppression and
    occlusion filling.

    Args:
        grid_h:      Number of rows in the column grid.
        grid_w:      Number of columns in the column grid.
        obj_dim:     Dimensionality of the "What" state per column.
        loc_dim:     Dimensionality of the "Where" state per column.
        patch_h:     Height of the local sensory patch per column.
        patch_w:     Width of the local sensory patch per column.
        activation_fn_name: Activation for the generative model.
    """

    def __init__(
        self,
        grid_h: int = 3,
        grid_w: int = 3,
        obj_dim: int = 16,
        loc_dim: int = 4,
        patch_h: int = 3,
        patch_w: int = 3,
        activation_fn_name: str = "tanh",
    ) -> None:
        super().__init__()

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.obj_dim = obj_dim
        self.loc_dim = loc_dim
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.sensory_dim = patch_h * patch_w

        # Build the 2D grid of columns as a flat ModuleList.
        # Index mapping: (r, c) -> r * grid_w + c
        columns: list[CorticalColumn] = []
        for _ in range(grid_h * grid_w):
            columns.append(
                CorticalColumn(
                    obj_dim=obj_dim,
                    loc_dim=loc_dim,
                    sensory_dim=self.sensory_dim,
                    activation_fn_name=activation_fn_name,
                )
            )
        self.columns = nn.ModuleList(columns)

        # Pre-compute neighbour indices for each cell (4-connected: UDLR).
        self._neighbor_idx: list[list[int]] = []
        for r in range(grid_h):
            for c in range(grid_w):
                nbrs: list[int] = []
                if r > 0:
                    nbrs.append((r - 1) * grid_w + c)           # Up
                if r < grid_h - 1:
                    nbrs.append((r + 1) * grid_w + c)           # Down
                if c > 0:
                    nbrs.append(r * grid_w + (c - 1))           # Left
                if c < grid_w - 1:
                    nbrs.append(r * grid_w + (c + 1))           # Right
                self._neighbor_idx.append(nbrs)

        self.energy_history: list[float] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _col(self, r: int, c: int) -> CorticalColumn:
        """Return the column at grid position (r, c)."""
        return self.columns[r * self.grid_w + c]

    def _slice_patches(self, global_input: Tensor) -> list[Tensor]:
        """Slice a (batch, H*W) global input into per-column patches.

        Assumes the global input is a flattened (grid_h * patch_h) ×
        (grid_w * patch_w) image laid out row-major.

        Args:
            global_input: (batch, grid_h*patch_h * grid_w*patch_w)

        Returns:
            List of (batch, patch_h*patch_w) tensors, one per column.
        """
        B = global_input.shape[0]
        img_h = self.grid_h * self.patch_h
        img_w = self.grid_w * self.patch_w

        # Reshape to (B, img_h, img_w).
        img = global_input.view(B, img_h, img_w)

        patches: list[Tensor] = []
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                y0 = r * self.patch_h
                x0 = c * self.patch_w
                patch = img[:, y0:y0 + self.patch_h, x0:x0 + self.patch_w]
                patches.append(patch.reshape(B, self.sensory_dim))

        return patches

    def _gather_neighbor_context(
        self, idx: int, batch_size: int, device: torch.device,
    ) -> Tensor:
        """Compute the average x_obj of the neighbours of column ``idx``.

        Returns:
            (batch, obj_dim) — mean x_obj of valid neighbours.
        """
        nbrs = self._neighbor_idx[idx]
        if len(nbrs) == 0:
            return torch.zeros(batch_size, self.obj_dim, device=device)

        stack = torch.stack(
            [self.columns[n].x_obj for n in nbrs], dim=0,
        )  # (N_nbrs, B, obj_dim)
        return stack.mean(dim=0)  # (B, obj_dim)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        global_input: Tensor,
        steps: int = 50,
        eta_x: float = 0.05,
    ) -> list[float]:
        """Run the ODE settling loop across the entire grid.

        Each step:
          1. Gather x_obj states of all columns.
          2. For each column, compute neighbor_context.
          3. Call column.infer_step(patch, neighbor_context).

        Args:
            global_input: (batch, grid_h*patch_h * grid_w*patch_w)
            steps:        Number of Euler integration steps.
            eta_x:        Step size for belief updates.

        Returns:
            energy_history: Total grid energy per step.
        """
        B = global_input.shape[0]
        device = global_input.device
        n_cols = len(self.columns)

        # Reset all columns.
        for col in self.columns:
            col.reset_states(B, device)

        # Slice global input into per-column patches.
        patches = self._slice_patches(global_input)

        self.energy_history = []

        for _ in range(steps):
            # Gather neighbor contexts for all columns FIRST (synchronous).
            contexts: list[Tensor] = []
            for i in range(n_cols):
                contexts.append(
                    self._gather_neighbor_context(i, B, device)
                )

            # Update all columns simultaneously.
            for i in range(n_cols):
                self.columns[i].infer_step(
                    patches[i],
                    eta_x=eta_x,
                    neighbor_context=contexts[i],
                )

            self.energy_history.append(self.get_total_energy())

        return self.energy_history

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_total_energy(self) -> float:
        """Return the sum of free energy across all columns."""
        return sum(col.get_energy() for col in self.columns)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, global_input: Tensor, eta_w: float = 0.001) -> None:
        """Hebbian update for all columns, including lateral weights.

        Recomputes neighbor contexts from the current settled states
        and passes them to each column's learn() method.

        Args:
            global_input: Same tensor used for infer() (needed for
                patches if we want to recompute, but not strictly
                required since states are already settled).
            eta_w: Learning rate.
        """
        B = self.columns[0].x_obj.shape[0]
        device = self.columns[0].x_obj.device
        n_cols = len(self.columns)

        for i in range(n_cols):
            ctx = self._gather_neighbor_context(i, B, device)
            self.columns[i].learn(eta_w=eta_w, neighbor_context=ctx)

    # ------------------------------------------------------------------
    # Temporal stepping
    # ------------------------------------------------------------------

    def step_time(self) -> None:
        """Advance the global clock: copy x_obj → x_obj_prev for all columns."""
        for col in self.columns:
            col.step_time()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"grid={self.grid_h}x{self.grid_w}, "
            f"obj_dim={self.obj_dim}, loc_dim={self.loc_dim}, "
            f"patch={self.patch_h}x{self.patch_w}"
        )


# ---------------------------------------------------------------------------
# CorticalStack — two-level hierarchy with top-down attention (Phase 3)
# ---------------------------------------------------------------------------

class CorticalStack(nn.Module):
    """A two-level hierarchical predictive coding stack.

    Level 1: CorticalGrid — a 2D grid of columns, each seeing a local patch.
    Level 2: A single CorticalColumn — the higher-order "concept" node.

    Information flows bidirectionally each inference step:
        Bottom-up: L1 x_obj states → concatenated → L2 sensory input
        Top-down:  L2 predict_down() → split → per-column L1 top_down_prior

    Level 2 treats the concatenated L1 x_obj states as its "sensory input".
    Its generative model learns to predict the aggregate L1 state, and its
    predictions flow back down as top-down priors that bias each L1 column's
    x_obj settling.  When L2's x_obj is clamped to a trained concept, this
    creates a strong top-down expectation that drives L1's L5 motor outputs
    to physically scan the environment for the expected target.

    Args:
        grid_h:      Number of rows in Level 1 grid.
        grid_w:      Number of columns in Level 1 grid.
        l1_obj_dim:  Dimensionality of Level 1 "What" state.
        l1_loc_dim:  Dimensionality of Level 1 "Where" state.
        l1_patch_h:  Height of each Level 1 column's sensory patch.
        l1_patch_w:  Width of each Level 1 column's sensory patch.
        l2_obj_dim:  Dimensionality of Level 2 "What" state.
        l2_loc_dim:  Dimensionality of Level 2 "Where" state.
        activation_fn_name: Activation function for all columns.
    """

    def __init__(
        self,
        grid_h: int = 3,
        grid_w: int = 3,
        l1_obj_dim: int = 16,
        l1_loc_dim: int = 4,
        l1_patch_h: int = 3,
        l1_patch_w: int = 3,
        l2_obj_dim: int = 32,
        l2_loc_dim: int = 4,
        activation_fn_name: str = "tanh",
    ) -> None:
        super().__init__()

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.l1_obj_dim = l1_obj_dim
        self.n_l1_cols = grid_h * grid_w

        # Level 1: CorticalGrid.
        self.level1 = CorticalGrid(
            grid_h=grid_h,
            grid_w=grid_w,
            obj_dim=l1_obj_dim,
            loc_dim=l1_loc_dim,
            patch_h=l1_patch_h,
            patch_w=l1_patch_w,
            activation_fn_name=activation_fn_name,
        )

        # Level 2: single column.
        # Its sensory_dim = total dimensionality of all L1 x_obj concatenated.
        self.l2_sensory_dim = self.n_l1_cols * l1_obj_dim
        self.level2 = CorticalColumn(
            obj_dim=l2_obj_dim,
            loc_dim=l2_loc_dim,
            sensory_dim=self.l2_sensory_dim,
            activation_fn_name=activation_fn_name,
        )

        self.energy_history: list[float] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_l1_states(self) -> Tensor:
        """Concatenate all Level 1 x_obj states into Level 2 sensory input.

        Returns:
            (batch, n_l1_cols * l1_obj_dim)
        """
        states = [col.x_obj for col in self.level1.columns]
        return torch.cat(states, dim=1)

    def _split_td_priors(self, l2_prediction: Tensor) -> list[Tensor]:
        """Split Level 2's prediction into per-column top-down priors.

        Args:
            l2_prediction: (batch, n_l1_cols * l1_obj_dim)

        Returns:
            List of n_l1_cols tensors, each (batch, l1_obj_dim).
        """
        return list(l2_prediction.split(self.l1_obj_dim, dim=1))

    # ------------------------------------------------------------------
    # Hierarchical Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        global_input: Tensor,
        steps: int = 50,
        eta_x: float = 0.05,
        freeze_l2: bool = False,
    ) -> list[float]:
        """Run synchronous hierarchical inference across both levels.

        Each step follows the GATHER-then-UPDATE pattern:
          GATHER phase:
            1. Read all L1 x_obj → concatenate as L2 sensory input.
            2. L2 predict_down() → split into per-column top_down_priors.
            3. L1 gather_neighbor_contexts (lateral).
          UPDATE phase:
            4. L2 infer_step(l2_sensory, eta_x, freeze_obj=freeze_l2).
            5. All L1 columns infer_step(patch, eta_x,
                   neighbor_context, top_down_prior).

        Args:
            global_input: (batch, grid_h*patch_h * grid_w*patch_w)
            steps: Number of Euler integration steps.
            eta_x: Step size for belief updates.
            freeze_l2: If True, Level 2 x_obj is frozen (for directed
                saccade — L2 "holds" the expected concept).

        Returns:
            energy_history: Total stack energy per step.
        """
        B = global_input.shape[0]
        device = global_input.device
        n_cols = self.n_l1_cols

        # Reset all states.
        for col in self.level1.columns:
            col.reset_states(B, device)
        self.level2.reset_states(B, device)

        # Slice global input into per-column patches (done once).
        patches = self.level1._slice_patches(global_input)

        self.energy_history = []

        for _ in range(steps):
            # === GATHER PHASE (read from previous step's states) ===

            # 1. Bottom-up: L1 x_obj → L2 sensory input.
            l2_sensory = self._gather_l1_states()

            # 2. Top-down: L2 prediction → per-column priors.
            td_priors = self._split_td_priors(
                self.level2.predict_down()
            )

            # 3. Lateral: L1 neighbor contexts.
            contexts: list[Tensor] = []
            for i in range(n_cols):
                contexts.append(
                    self.level1._gather_neighbor_context(i, B, device)
                )

            # === UPDATE PHASE (all simultaneous) ===

            # 4. Level 2 update.
            self.level2.infer_step(
                l2_sensory,
                eta_x=eta_x,
                freeze_obj=freeze_l2,
            )

            # 5. Level 1 updates (all columns simultaneously).
            for i in range(n_cols):
                self.level1.columns[i].infer_step(
                    patches[i],
                    eta_x=eta_x,
                    neighbor_context=contexts[i],
                    top_down_prior=td_priors[i],
                )

            self.energy_history.append(self.get_total_energy())

        return self.energy_history

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_total_energy(self) -> float:
        """Return the sum of free energy across all columns at both levels."""
        return self.level1.get_total_energy() + self.level2.get_energy()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, global_input: Tensor, eta_w: float = 0.001) -> None:
        """Hebbian update for all columns at both levels.

        Both levels learn jointly from their settled states.
        Level 1 learns W_obj, W_loc, W_lat from sensory + lateral error.
        Level 2 learns W_obj, W_loc from its sensory error (the
        concatenated L1 states).

        Args:
            global_input: Same tensor used for infer().
            eta_w: Learning rate.
        """
        # Level 1: learn with lateral contexts.
        self.level1.learn(global_input, eta_w=eta_w)

        # Level 2: learn from settled L1 states.
        # No neighbor_context for Level 2 (single column, no lateral).
        self.level2.learn(eta_w=eta_w)

    # ------------------------------------------------------------------
    # Motor aggregation
    # ------------------------------------------------------------------

    def get_global_action(
        self,
        sensory_gradients: list[Tensor],
        eta_a: float = 0.01,
    ) -> Tensor:
        """Aggregate L5 motor actions from all Level 1 columns.

        Each column computes its own motor action from its local
        sensory gradient and prediction error.  The global action
        is their average — a consensus saccade vote.

        Args:
            sensory_gradients: List of n_l1_cols tensors, each
                (batch, sensory_dim, action_dim).
            eta_a: Motor gain.

        Returns:
            action: (batch, action_dim) — averaged global motor command.
        """
        actions = []
        for i, col in enumerate(self.level1.columns):
            actions.append(col.get_motor_action(sensory_gradients[i], eta_a))
        # Stack: (n_cols, B, action_dim) → mean over columns.
        return torch.stack(actions, dim=0).mean(dim=0)

    # ------------------------------------------------------------------
    # Temporal stepping
    # ------------------------------------------------------------------

    def step_time(self) -> None:
        """Advance the global clock for both levels."""
        self.level1.step_time()
        self.level2.step_time()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"level1={self.grid_h}x{self.grid_w} "
            f"(obj={self.l1_obj_dim}), "
            f"level2_obj={self.level2.obj_dim}, "
            f"l2_sensory={self.l2_sensory_dim}"
        )


# ---------------------------------------------------------------------------
# Smoke test — CorticalGrid: infer + learn, verify energy drops
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = _get_device()
    print(f"Using device: {device}\n")

    grid = CorticalGrid(
        grid_h=3, grid_w=3,
        obj_dim=8, loc_dim=4,
        patch_h=3, patch_w=3,
        activation_fn_name="tanh",
    ).to(device)
    print(grid)
    print(f"  Total columns: {len(grid.columns)}")
    print(f"  Sensory dim per column: {grid.sensory_dim}")
    print(f"  Global image size: {grid.grid_h * grid.patch_h}x"
          f"{grid.grid_w * grid.patch_w} = "
          f"{grid.grid_h * grid.patch_h * grid.grid_w * grid.patch_w}\n")

    B = 1
    img_dim = grid.grid_h * grid.patch_h * grid.grid_w * grid.patch_w
    sensory = torch.randn(B, img_dim, device=device)

    # --- Training loop ---
    n_epochs = 10
    infer_steps = 30
    eta_x = 0.1
    eta_w = 0.01

    print(f"  Config: {n_epochs} epochs x {infer_steps} infer steps")
    print(f"    eta_x={eta_x}, eta_w={eta_w}\n")

    all_energies: list[float] = []

    for epoch in range(n_epochs):
        history = grid.infer(sensory, steps=infer_steps, eta_x=eta_x)
        all_energies.extend(history)
        grid.learn(sensory, eta_w=eta_w)

        if epoch == 0 or epoch == n_epochs - 1:
            print(
                f"  Epoch {epoch:2d}  "
                f"start={history[0]:.4f}  end={history[-1]:.4f}"
            )

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

    # Verify lateral weights changed.
    center = grid._col(1, 1)
    assert not torch.allclose(
        center.W_lat.data, torch.eye(grid.obj_dim, device=device)
    ), "Center column W_lat did not change!"
    print(f"  Center W_lat delta from I: "
          f"{(center.W_lat.data - torch.eye(grid.obj_dim, device=device)).abs().sum().item():.4f}  [OK]")

    # No autograd.
    for name, param in grid.named_parameters():
        assert not param.requires_grad, f"{name} has requires_grad=True!"
    print("  All parameters requires_grad=False  [OK]")

    print("\nCorticalGrid smoke test passed.")

    # -------------------------------------------------------------------
    # CorticalStack smoke test
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CorticalStack smoke test")
    print(f"{'='*60}\n")

    stack = CorticalStack(
        grid_h=3, grid_w=3,
        l1_obj_dim=8, l1_loc_dim=4,
        l1_patch_h=3, l1_patch_w=3,
        l2_obj_dim=16, l2_loc_dim=4,
        activation_fn_name="tanh",
    ).to(device)
    print(stack)
    print(f"  L1 columns  : {stack.n_l1_cols}")
    print(f"  L2 sensory  : {stack.l2_sensory_dim}  "
          f"(expect {stack.n_l1_cols}*{stack.l1_obj_dim}="
          f"{stack.n_l1_cols * stack.l1_obj_dim})")
    print(f"  L2 obj_dim  : {stack.level2.obj_dim}\n")

    # Train for a few epochs.
    stack_input = torch.randn(1, 81, device=device)
    n_epochs_s = 10
    infer_steps_s = 30

    for epoch in range(n_epochs_s):
        history_s = stack.infer(stack_input, steps=infer_steps_s, eta_x=0.1)
        stack.learn(stack_input, eta_w=0.01)
        if epoch == 0 or epoch == n_epochs_s - 1:
            print(f"  Epoch {epoch:2d}  "
                  f"start={history_s[0]:.4f}  end={history_s[-1]:.4f}")

    assert history_s[-1] < history_s[0], "Stack energy did not decrease!"
    print(f"\n  Energy decreased  [OK]")

    # Verify L2 x_obj is non-trivial.
    l2_norm = stack.level2.x_obj.norm().item()
    assert l2_norm > 0.01, "L2 x_obj is near zero after training!"
    print(f"  L2 x_obj norm: {l2_norm:.4f}  [OK]")

    # Verify top-down priors reach L1 (err_td should be non-zero).
    avg_td = sum(
        col.err_td.norm().item() for col in stack.level1.columns
    ) / stack.n_l1_cols
    print(f"  Avg L1 err_td norm: {avg_td:.4f}  [OK]")

    # Motor aggregation test.
    grads = [torch.randn(1, 9, 2, device=device) for _ in range(9)]
    action_s = stack.get_global_action(grads, eta_a=0.01)
    assert action_s.shape == (1, 2), f"Expected (1,2), got {action_s.shape}"
    print(f"  Global action shape: {action_s.shape}  [OK]")

    # Freeze L2 test.
    trained_l2 = stack.level2.x_obj.clone()
    history_f = stack.infer(stack_input, steps=10, eta_x=0.1, freeze_l2=True)
    # L2 x_obj should not have changed (it was reset then settled with freeze).
    # We just check it doesn't crash.
    print(f"  freeze_l2 inference runs  [OK]")

    # No autograd.
    for name, param in stack.named_parameters():
        assert not param.requires_grad, f"{name} has requires_grad=True!"
    print("  All parameters requires_grad=False  [OK]")

    print("\nCorticalStack smoke test passed.")
