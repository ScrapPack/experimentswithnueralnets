"""
Predictive Coding Network — Phases 2, 3, 5a & 5b

A hierarchical generative model that performs inference by iteratively
settling its internal belief states to minimise variational free energy.
There is no backpropagation — every computation is local.

Phase 5b adds **lateral connections** within each PCLayer.  Neurons now
predict their peers horizontally, adding a third pressure term to the
inference ODE and a lateral Hebbian update to learning.

Fast dynamics (inference — Euler integration):

    pe_i  = e_i * pi_i
    dx_i  = -pe_i
            + f'(x_i) * (pe_below @ W_i)          [bottom-up]
            + f'(x_i) * (pe_i @ L_i)              [lateral]

Slow dynamics (learning — Hebbian + lateral + variance descent):

    dW_i  = pe_below^T @ f(x_i) / B               [top-down weights]
    db_i  = pe_below.sum(0) / B                    [top-down bias]
    dL_i  = pe_i^T @ f(x_i) / B;  diag(L_i) := 0  [lateral weights]
    d_lv  = 0.5 * (1 - (e^2 * pi).mean(0))        [variance]

Free energy under a diagonal-Gaussian generative model:

    F = 0.5 * sum_i [ e_i^2 * pi_i + log_var_i ]

Hardware: defaults to MPS (Apple Silicon) when available.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from pc_layer import PCLayer, _get_device


class PCNetwork(nn.Module):
    """A hierarchical Predictive Coding network with precision weighting
    and lateral connections.

    Builds a stack of :class:`PCLayer` objects and runs continuous-time
    Euler-integration inference to settle belief states toward a
    free-energy minimum.  Each layer (and the sensory surface) maintains
    a learnable log-variance that controls per-dimension precision, and
    a lateral weight matrix for intra-layer predictions.

    Hierarchy convention (top -> bottom):
        ``layer_dims[0]``  is the **top** (most abstract) layer.
        ``layer_dims[-1]`` is the **sensory** layer (clamped data, not a
        PCLayer).

    For ``layer_dims = [16, 64, 32]``:
        layers[0] = PCLayer(input_dim=16, output_dim=64)  — top
        layers[1] = PCLayer(input_dim=64, output_dim=32)  — bottom PCLayer
        sensory dim = 32  (clamped, no learnable state)

    Args:
        layer_dims:          Dimensions from top to bottom, e.g. [16, 64, 32].
                             Must contain at least 2 entries (one PCLayer +
                             the sensory layer).
        activation_fn_name:  Activation for every PCLayer ('tanh', 'relu',
                             'sigmoid').
    """

    def __init__(
        self,
        layer_dims: list[int],
        activation_fn_name: str = "tanh",
    ) -> None:
        super().__init__()

        if len(layer_dims) < 2:
            raise ValueError("Need at least 2 dimensions (one PCLayer + sensory).")

        self.layer_dims = layer_dims
        self.sensory_dim = layer_dims[-1]

        # Build PCLayers: one for each adjacent pair of dims.
        # layers[0] is the top-most; layers[-1] predicts the sensory layer.
        pc_layers: list[PCLayer] = []
        for i in range(len(layer_dims) - 1):
            pc_layers.append(
                PCLayer(
                    input_dim=layer_dims[i],
                    output_dim=layer_dims[i + 1],
                    activation_fn_name=activation_fn_name,
                )
            )
        self.layers = nn.ModuleList(pc_layers)

        # Sensory precision.  The sensory layer is not a PCLayer, so its
        # log-variance lives here on the network.
        self.sensory_log_var = nn.Parameter(
            torch.zeros(self.sensory_dim), requires_grad=False
        )

        # Sensory error — stored so get_total_energy() can include it.
        self._e_sensory: Optional[Tensor] = None

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
        """Run the Euler-integration settling loop with precision
        weighting and lateral connections.

        Clamps ``sensory_data`` at the bottom and lets all PCLayer
        belief states evolve to minimise total free energy.  The state
        ODE now has three forces:

            1. **Self-correction** (``-pe_i``): precision-weighted error
               pulls the state toward consistency with predictions from
               above and from peers.
            2. **Bottom-up pressure**: error at the level below,
               propagated through the generative weights.
            3. **Lateral pressure**: this layer's own precision-weighted
               error propagated through the lateral weights.

        Args:
            sensory_data: (batch, sensory_dim) — observed data clamped
                          at layer 0.
            steps:        Number of Euler integration steps.
            eta_x:        Step size for belief updates.

        Returns:
            energy_history: A list of ``steps`` total-energy scalars,
            one per integration step.
        """
        batch_size = sensory_data.shape[0]
        device = sensory_data.device
        n_layers = len(self.layers)

        # --- 1. Reset all states ---
        for layer in self.layers:
            layer.reset_states(batch_size, device)

        self.energy_history = []

        # Pre-allocate a zero tensor for the top layer's prior.
        zero_prior = torch.zeros(
            batch_size, self.layers[0].input_dim, device=device
        )

        # --- 2. Euler integration ---
        for _ in range(steps):
            # ---- 2a. Compute predictions (top -> bottom) ----
            predictions: list[Tensor] = []
            for layer in self.layers:
                predictions.append(layer.predict_down(layer.x))
            # predictions[i] is layer i's prediction for the level below.
            # predictions[-1] targets the sensory layer.

            # ---- 2b. Compute raw errors (now includes lateral pred) ----
            # Top layer: zero prior from above.
            # compute_error now adds lateral prediction internally.
            self.layers[0].compute_error(zero_prior)

            # Middle / bottom layers.
            for i in range(1, n_layers):
                self.layers[i].compute_error(predictions[i - 1])

            # Sensory error: e_0 = sensory_data - prediction_for_sensory
            self._e_sensory = sensory_data - predictions[-1]

            # ---- 2c. Compute precision-weighted errors ----
            pe_layers: list[Tensor] = []
            for layer in self.layers:
                pe_layers.append(layer.e * layer.get_precision())

            pe_sensory = self._e_sensory * torch.exp(-self.sensory_log_var)

            # ---- 2d. Compute all dx FIRST (synchronous update) ----
            dx_list: list[Tensor] = []
            for i, layer in enumerate(self.layers):
                # pe_below: precision-weighted error of the level this
                # layer predicts.
                if i < n_layers - 1:
                    pe_below = pe_layers[i + 1]
                else:
                    pe_below = pe_sensory

                deriv = layer.activation_deriv(layer.x)

                # Bottom-up pressure:  f'(x_i) * (pe_below @ W_i)
                bottom_up = deriv * (pe_below @ layer.weight)

                # Lateral pressure:  f'(x_i) * (pe_i @ L_i)
                lateral_pressure = deriv * (
                    pe_layers[i] @ layer.lateral_weight
                )

                dx = -pe_layers[i] + bottom_up + lateral_pressure
                dx_list.append(dx)

            # ---- 2e. Apply updates simultaneously ----
            for layer, dx in zip(self.layers, dx_list):
                layer.x = layer.x + eta_x * dx

            # ---- 2f. Track energy ----
            self.energy_history.append(self.get_total_energy())

        return self.energy_history

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_total_energy(self) -> float:
        """Return the current total variational free energy (scalar).

        Under a diagonal-Gaussian generative model:

        .. math::
            F = \\frac{1}{2} \\sum_i \\sum_j
                \\left[ \\epsilon_{i,j}^2 \\, \\pi_{i,j}
                + \\text{log\\_var}_{i,j} \\right]

        The sum runs over every PCLayer error *and* the sensory error,
        with each term weighted by its layer's precision and including
        the log-determinant (log_var) cost that penalises trivially
        increasing variance to hide errors.
        """
        energy = 0.0

        for layer in self.layers:
            if layer.e is not None:
                pi = layer.get_precision()                       # (input_dim,)
                energy += 0.5 * (
                    layer.e * layer.e * pi + layer.log_var
                ).sum().item()

        if self._e_sensory is not None:
            sensory_pi = torch.exp(-self.sensory_log_var)        # (sensory_dim,)
            energy += 0.5 * (
                self._e_sensory * self._e_sensory * sensory_pi
                + self.sensory_log_var
            ).sum().item()

        return energy

    # ------------------------------------------------------------------
    # Learning (slow dynamics)
    # ------------------------------------------------------------------

    def learn(
        self,
        eta_w: float = 0.001,
        eta_v: float = 0.01,
        eta_l: float = 0.001,
    ) -> None:
        """Update all generative weights, lateral weights, and variances.

        Must be called **after** :meth:`infer` has settled the states.

        Four update channels per layer:
            1. **Top-down weights & biases** — Hebbian update driven by
               the precision-weighted error below (``pe_below``).
            2. **Lateral weights** — Hebbian update driven by this
               layer's own precision-weighted error (``pe_i``).
               Diagonal re-zeroed after each update.
            3. **Layer log-variance** — gradient descent on the
               Gaussian free energy.
            4. **Sensory log-variance** — same variance update for the
               sensory surface.

        Args:
            eta_w: Learning rate for top-down weight / bias updates.
            eta_v: Learning rate for log-variance updates.
            eta_l: Learning rate for lateral weight updates.

        Raises:
            RuntimeError: If ``infer()`` has not been called yet.
        """
        n_layers = len(self.layers)

        if self._e_sensory is None:
            raise RuntimeError(
                "No sensory error found. Call infer() before learn()."
            )

        # Sensory precision (used for the bottom PCLayer's pe_below).
        sensory_pi = torch.exp(-self.sensory_log_var)

        for i, layer in enumerate(self.layers):
            # --- Precision-weighted errors ---
            pe_i = layer.e * layer.get_precision()

            if i < n_layers - 1:
                below = self.layers[i + 1]
                pe_below = below.e * below.get_precision()
            else:
                pe_below = self._e_sensory * sensory_pi

            # --- Weight / bias / lateral update (Hebbian) ---
            layer.update_weights(pe_below, pe_i, eta_w, eta_l)

            # --- Log-variance update ---
            layer.update_variance(eta_v)

        # --- Sensory log-variance update ---
        d_sensory_lv = 0.5 * (
            1.0 - (self._e_sensory ** 2 * sensory_pi).mean(dim=0)
        )
        self.sensory_log_var.data -= eta_v * d_sensory_lv

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"layer_dims={self.layer_dims}"


# ---------------------------------------------------------------------------
# Smoke test — training loop with lateral connections: infer then learn,
# verify energy drops and lateral weights change across epochs.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = _get_device()
    print(f"Using device: {device}\n")

    # Build a 3-level hierarchy: [16, 64, 32]
    #   layers[0]: PCLayer(16, 64)  — top
    #   layers[1]: PCLayer(64, 32)  — bottom PCLayer
    #   sensory: dim 32 (clamped)
    net = PCNetwork(layer_dims=[16, 64, 32], activation_fn_name="tanh").to(device)
    print(net)
    print()

    batch_size = 8
    infer_steps = 30
    n_epochs = 10
    eta_x = 0.05
    eta_w = 0.001
    eta_v = 0.01
    eta_l = 0.001

    # Fixed sensory batch — the network must learn to reconstruct this.
    sensory = torch.randn(batch_size, 32, device=device)

    # Snapshot initial parameters.
    w0_before = net.layers[0].weight.data.clone()
    w1_before = net.layers[1].weight.data.clone()
    lv0_before = net.layers[0].log_var.data.clone()
    lv1_before = net.layers[1].log_var.data.clone()
    slv_before = net.sensory_log_var.data.clone()
    lat0_mean_before = net.layers[0].lateral_weight.data.abs().mean().item()
    lat1_mean_before = net.layers[1].lateral_weight.data.abs().mean().item()

    print(f"  Config: {n_epochs} epochs x {infer_steps} infer steps")
    print(f"    eta_x={eta_x}, eta_w={eta_w}, eta_v={eta_v}, eta_l={eta_l}")
    print(f"  Sensory batch: {sensory.shape}\n")

    # --- Training loop ---
    all_energies: list[float] = []

    for epoch in range(n_epochs):
        # Fast dynamics: settle beliefs.
        history = net.infer(sensory, steps=infer_steps, eta_x=eta_x)
        all_energies.extend(history)

        # Slow dynamics: update weights, lateral, variances.
        net.learn(eta_w=eta_w, eta_v=eta_v, eta_l=eta_l)

        if epoch == 0:
            print(f"  Epoch  0  step  0 energy: {history[0]:.4f}")
            print(f"  Epoch  0  step {infer_steps:2d} energy: {history[-1]:.4f}")
        elif epoch == n_epochs - 1:
            print(f"  Epoch  {epoch}  step  0 energy: {history[0]:.4f}")
            print(f"  Epoch  {epoch}  step {infer_steps:2d} energy: {history[-1]:.4f}")

    # --- Assertions ---
    print()

    start_energy = all_energies[0]
    final_energy = all_energies[-1]

    # 1. Energy must decrease.
    assert final_energy < start_energy, (
        f"Energy did not decrease! start={start_energy:.4f} end={final_energy:.4f}"
    )
    reduction_pct = (start_energy - final_energy) / abs(start_energy) * 100
    print(f"  Start energy : {start_energy:.4f}")
    print(f"  Final energy : {final_energy:.4f}")
    print(f"  Reduction    : {reduction_pct:.1f}%")
    print("  Energy decreased across training  [OK]")

    # 2. Learning reshapes the landscape.
    epoch0_final = all_energies[infer_steps - 1]
    epoch9_final = all_energies[-1]
    assert epoch9_final < epoch0_final, (
        f"Learning did not help! epoch0_final={epoch0_final:.4f} "
        f"epoch9_final={epoch9_final:.4f}"
    )
    print(f"  Epoch 0 settled energy : {epoch0_final:.4f}")
    print(f"  Epoch 9 settled energy : {epoch9_final:.4f}")
    print("  Learning reshapes the energy landscape  [OK]")

    # 3. Top-down weights changed.
    w0_delta = (net.layers[0].weight.data - w0_before).abs().sum().item()
    w1_delta = (net.layers[1].weight.data - w1_before).abs().sum().item()
    assert w0_delta > 0, "Layer 0 weights did not change!"
    assert w1_delta > 0, "Layer 1 weights did not change!"
    print(f"  Layer 0 weight delta : {w0_delta:.6f}")
    print(f"  Layer 1 weight delta : {w1_delta:.6f}")
    print("  Top-down weights updated  [OK]")

    # 4. Lateral weights changed.
    lat0_mean_after = net.layers[0].lateral_weight.data.abs().mean().item()
    lat1_mean_after = net.layers[1].lateral_weight.data.abs().mean().item()
    print(f"  Layer 0 lateral |mean| before: {lat0_mean_before:.6f}  after: {lat0_mean_after:.6f}")
    print(f"  Layer 1 lateral |mean| before: {lat1_mean_before:.6f}  after: {lat1_mean_after:.6f}")
    assert lat0_mean_after != lat0_mean_before, "Layer 0 lateral weights did not change!"
    assert lat1_mean_after != lat1_mean_before, "Layer 1 lateral weights did not change!"
    print("  Lateral weights updated  [OK]")

    # 5. Lateral diagonal is still zero.
    for idx, layer in enumerate(net.layers):
        diag_max = layer.lateral_weight.data.diag().abs().max().item()
        assert diag_max == 0.0, f"Layer {idx} lateral diagonal not zero!"
    print("  Lateral diagonal == 0 after training  [OK]")

    # 6. Log-variances changed.
    lv0_delta = (net.layers[0].log_var.data - lv0_before).abs().sum().item()
    lv1_delta = (net.layers[1].log_var.data - lv1_before).abs().sum().item()
    slv_delta = (net.sensory_log_var.data - slv_before).abs().sum().item()
    assert lv0_delta > 0, "Layer 0 log_var did not change!"
    assert lv1_delta > 0, "Layer 1 log_var did not change!"
    assert slv_delta > 0, "Sensory log_var did not change!"
    print(f"  Layer 0 log_var delta   : {lv0_delta:.6f}")
    print(f"  Layer 1 log_var delta   : {lv1_delta:.6f}")
    print(f"  Sensory log_var delta   : {slv_delta:.6f}")
    print("  Log-variances updated  [OK]")

    # 7. Precision ranges.
    for idx, layer in enumerate(net.layers):
        pi = layer.get_precision()
        print(f"  Layer {idx} precision range: "
              f"[{pi.min().item():.4f}, {pi.max().item():.4f}]")
    sensory_pi = torch.exp(-net.sensory_log_var)
    print(f"  Sensory precision range: "
          f"[{sensory_pi.min().item():.4f}, {sensory_pi.max().item():.4f}]")

    # 8. No autograd contamination.
    for name, param in net.named_parameters():
        assert not param.requires_grad, f"{name} has requires_grad=True!"
    print("  All parameters have requires_grad=False  [OK]")

    print("\nPhase 5b smoke test passed.")
