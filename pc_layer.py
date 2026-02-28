"""
Predictive Coding Layer — Phases 1, 3, 5a & 5b

A stateful, locally autonomous layer for a predictive coding network.
This is an energy-based model component that maintains its own internal
dynamics (beliefs and prediction errors) and updates via local Hebbian
rules. No autograd. No global clock. No backpropagation.

Mathematical foundation:
    - Top-down prediction:  p_td = activation(x_above) @ W^T + b
    - Lateral prediction:   p_lat = activation(x) @ L^T
    - Local error:          e = x - p_td - p_lat
    - Precision:            pi = exp(-log_var)   (per-dimension)
    - Precision-weighted error:  pe = e * pi
    - The derivative of the activation function is computed analytically
      (not via autograd) to support bottom-up and lateral pressure.
    - Hebbian weight update (slow dynamics):
          dW = pe_below^T @ activation(x) / B
          db = pe_below.sum(dim=0) / B
    - Lateral Hebbian update (slow dynamics):
          dL = pe_i^T @ activation(x) / B
          L += eta_l * dL;  diag(L) := 0
    - Variance update (slow dynamics):
          d_log_var = 0.5 * (1 - (e^2 * pi).mean(dim=0))
          log_var -= eta_v * d_log_var

Hardware: defaults to MPS (Apple Silicon) when available.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


def _get_device() -> torch.device:
    """Return the best available device, preferring MPS on Apple Silicon."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Activation helpers — closed-form function + analytical derivative
# ---------------------------------------------------------------------------

def _tanh_pair() -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return (tanh, d_tanh/dx) where the derivative is 1 - tanh^2(x)."""
    def fwd(x: Tensor) -> Tensor:
        return torch.tanh(x)

    def deriv(x: Tensor) -> Tensor:
        t = torch.tanh(x)
        return 1.0 - t * t

    return fwd, deriv


def _relu_pair() -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return (relu, d_relu/dx) where the derivative is the Heaviside step."""
    def fwd(x: Tensor) -> Tensor:
        return torch.relu(x)

    def deriv(x: Tensor) -> Tensor:
        return (x > 0).to(x.dtype)

    return fwd, deriv


def _sigmoid_pair() -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return (sigmoid, d_sigmoid/dx) where the derivative is sig(x)(1 - sig(x))."""
    def fwd(x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    def deriv(x: Tensor) -> Tensor:
        s = torch.sigmoid(x)
        return s * (1.0 - s)

    return fwd, deriv


_ACTIVATION_REGISTRY: dict[str, Callable[[], tuple[Callable, Callable]]] = {
    "tanh": _tanh_pair,
    "relu": _relu_pair,
    "sigmoid": _sigmoid_pair,
}


# ---------------------------------------------------------------------------
# PCLayer
# ---------------------------------------------------------------------------

class PCLayer(nn.Module):
    """A single Predictive Coding layer with lateral connections.

    This layer is a *generative* unit: it holds top-down weights that
    predict the activity of the layer below, and **lateral weights**
    that let neurons within the same layer predict each other.  It
    maintains its own mutable state tensors (belief ``x`` and error
    ``e``) which evolve during the iterative settling phase.

    Parameters (slow dynamics, updated by Hebbian rules — not by autograd):
        weight         : (output_dim, input_dim) — generative weights, top-down.
        bias           : (output_dim,)           — generative bias.
        log_var        : (input_dim,)            — log-variance of beliefs.
        lateral_weight : (input_dim, input_dim)  — lateral (horizontal)
                         weights.  Diagonal is always zero to prevent
                         self-excitation.

    States (fast dynamics, reset each observation):
        x : (batch, input_dim)  — current belief / neural activity.
        e : (batch, input_dim)  — local prediction error (same shape as x).

    The convention ``input_dim -> output_dim`` follows the *generative*
    (top-down) direction: ``input_dim`` is the dimensionality of *this*
    layer's beliefs, and ``output_dim`` is the dimensionality of the
    layer below that we are predicting.

    Args:
        input_dim:  Dimensionality of this layer's activity (x).
        output_dim: Dimensionality of the layer below (prediction target).
        activation_fn_name: One of ``'tanh'``, ``'relu'``, ``'sigmoid'``.
    """

    weight: Tensor
    bias: Tensor
    log_var: Tensor
    lateral_weight: Tensor
    x: Optional[Tensor]
    e: Optional[Tensor]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_fn_name: str = "tanh",
    ) -> None:
        super().__init__()

        if activation_fn_name not in _ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unknown activation '{activation_fn_name}'. "
                f"Choose from {list(_ACTIVATION_REGISTRY.keys())}."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn_name = activation_fn_name

        # Activation function and its analytical derivative.
        self.activation_fn, self.activation_deriv = _ACTIVATION_REGISTRY[
            activation_fn_name
        ]()

        # --- Slow parameters (no autograd) ---
        # Weight shape: (output_dim, input_dim) — maps from this layer's
        # activated state down to the layer below.
        weight = torch.empty(output_dim, input_dim)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight, requires_grad=False)

        # Bias initialised following the same fan-in convention as nn.Linear.
        fan_in = input_dim
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        bias = torch.empty(output_dim).uniform_(-bound, bound)
        self.bias = nn.Parameter(bias, requires_grad=False)

        # Log-variance of this layer's beliefs.  Initialised to 0 so that
        # precision = exp(-0) = 1  (uniform confidence everywhere).
        self.log_var = nn.Parameter(
            torch.zeros(input_dim), requires_grad=False
        )

        # Lateral (horizontal) weight matrix — neurons predicting their
        # peers within the same layer.  Small random init; diagonal is
        # zeroed to prevent self-excitation.
        lateral = torch.empty(input_dim, input_dim)
        nn.init.normal_(lateral, mean=0.0, std=0.01)
        lateral.fill_diagonal_(0)
        self.lateral_weight = nn.Parameter(lateral, requires_grad=False)

        # --- Fast states (initialised lazily per-batch) ---
        self.x = None
        self.e = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_states(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """Zero-initialise the belief and error states for a new observation.

        Call this once before each settling phase begins.

        Args:
            batch_size: Number of samples in the current batch.
            device:     Target device. If ``None``, uses the device of
                        ``self.weight``.
        """
        if device is None:
            device = self.weight.device

        self.x = torch.zeros(batch_size, self.input_dim, device=device)
        self.e = torch.zeros(batch_size, self.input_dim, device=device)

    # ------------------------------------------------------------------
    # Precision (attention / neuromodulation)
    # ------------------------------------------------------------------

    def get_precision(self) -> Tensor:
        """Return the per-dimension precision vector.

        .. math::
            \\pi = \\exp(-\\text{log\\_var})

        High precision (low variance) means the layer is *confident* in
        that dimension — its prediction errors will exert more force on
        both inference and learning.

        Returns:
            precision: (input_dim,)
        """
        return torch.exp(-self.log_var)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def predict_down(self, x_above: Tensor) -> Tensor:
        """Compute the top-down prediction for the layer below.

        .. math::
            p = \\text{activation}(x_{above}) \\; W^T + b

        ``x_above`` is the raw (pre-activation) belief state of the layer
        above this one.  We apply the activation here so that the
        non-linearity lives in the generative model, not in the state
        update rule.

        Args:
            x_above: (batch, input_dim) — beliefs of the layer above.

        Returns:
            prediction: (batch, output_dim) — predicted activity for the
            layer below.
        """
        activated = self.activation_fn(x_above)                 # (B, input_dim)
        prediction = activated @ self.weight.t() + self.bias    # (B, output_dim)
        return prediction

    def compute_error(self, prediction_from_above: Tensor) -> Tensor:
        """Compute and store the local prediction error.

        The error now accounts for both the top-down prediction from the
        layer above *and* the lateral prediction from peer neurons:

        .. math::
            p_{\\text{lat}} = f(x) \\; L^T

            \\epsilon = x - p_{\\text{above}} - p_{\\text{lat}}

        Args:
            prediction_from_above: (batch, input_dim) — the top-down
                prediction targeting this layer's activity.

        Returns:
            error: (batch, input_dim) — the freshly computed error, also
            stored in ``self.e``.

        Raises:
            RuntimeError: If ``reset_states`` has not been called yet.
        """
        if self.x is None:
            raise RuntimeError(
                "States not initialised. Call reset_states() before compute_error()."
            )

        # Lateral prediction: peers predicting each other.
        p_lat = self.activation_fn(self.x) @ self.lateral_weight.t()

        self.e = self.x - prediction_from_above - p_lat
        return self.e

    # ------------------------------------------------------------------
    # Learning (slow dynamics)
    # ------------------------------------------------------------------

    def update_weights(
        self,
        pe_below: Tensor,
        pe_i: Tensor,
        eta_w: float,
        eta_l: float = 0.001,
    ) -> None:
        """Hebbian update for top-down weights, bias, and lateral weights.

        Called *after* the inference loop has settled.

        **Top-down weights & bias** — driven by the precision-weighted
        error at the layer below:

        .. math::
            \\Delta W = \\frac{1}{B} \\; \\tilde{\\epsilon}_{\\text{below}}^{\\top} \\; f(x)

            \\Delta b = \\frac{1}{B} \\; \\sum_{\\text{batch}} \\tilde{\\epsilon}_{\\text{below}}

        **Lateral weights** — driven by this layer's own precision-weighted
        error:

        .. math::
            \\Delta L = \\frac{1}{B} \\; \\tilde{\\epsilon}_i^{\\top} \\; f(x)

        followed by zeroing the diagonal to prevent self-excitation.

        Args:
            pe_below: (batch, output_dim) — precision-weighted error at
                      the layer this PCLayer predicts (one level down).
            pe_i:     (batch, input_dim) — this layer's own precision-
                      weighted error.
            eta_w:    Learning rate for top-down weight / bias updates.
            eta_l:    Learning rate for lateral weight updates.

        Raises:
            RuntimeError: If states have not been initialised.
        """
        if self.x is None:
            raise RuntimeError(
                "States not initialised.  Run infer() before update_weights()."
            )

        batch_size = pe_below.shape[0]

        # activated: (B, input_dim)
        activated = self.activation_fn(self.x)

        # --- Top-down weight update ---
        # dW: (output_dim, B) @ (B, input_dim) -> (output_dim, input_dim)
        dW = (pe_below.t() @ activated) / batch_size
        db = pe_below.sum(dim=0) / batch_size

        self.weight.data += eta_w * dW
        self.bias.data += eta_w * db

        # --- Lateral weight update ---
        # dL: (input_dim, B) @ (B, input_dim) -> (input_dim, input_dim)
        dL = (pe_i.t() @ activated) / batch_size

        self.lateral_weight.data += eta_l * dL
        # CRITICAL: re-zero diagonal to prevent self-excitation.
        self.lateral_weight.data.fill_diagonal_(0)

    def update_variance(self, eta_v: float) -> None:
        """Update the log-variance via gradient descent on free energy.

        The free-energy contribution from this layer under a Gaussian
        generative model is:

        .. math::
            F_i = \\frac{1}{2} \\sum_j \\left[ \\epsilon_j^2 \\, \\pi_j
                  + \\log \\sigma_j^2 \\right]

        Taking the derivative w.r.t. ``log_var_j``:

        .. math::
            \\frac{\\partial F_i}{\\partial \\text{log\\_var}_j}
                = \\frac{1}{2} \\left(1 - \\epsilon_j^2 \\, \\pi_j \\right)

        averaged over the batch.  The update is gradient descent:

        .. math::
            \\text{log\\_var} \\;-\\!=\\; \\eta_v \\cdot d\\text{log\\_var}

        **Intuition**: If squared errors are large relative to precision,
        the gradient is negative, so ``log_var`` *increases* (variance up,
        precision down) — the layer becomes less confident.  If errors
        are small, ``log_var`` *decreases* (precision up) — the layer
        becomes more confident.

        Args:
            eta_v: Learning rate for variance updates.

        Raises:
            RuntimeError: If states have not been initialised.
        """
        if self.e is None:
            raise RuntimeError(
                "States not initialised.  Run infer() before update_variance()."
            )

        pi = self.get_precision()                                # (input_dim,)
        # d_log_var: (input_dim,) — mean over batch
        d_log_var = 0.5 * (1.0 - (self.e ** 2 * pi).mean(dim=0))
        self.log_var.data -= eta_v * d_log_var

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"activation={self.activation_fn_name}"
        )


# ---------------------------------------------------------------------------
# Quick smoke test — verifies shapes, device placement, precision,
# lateral connections, and no autograd.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = _get_device()
    print(f"Using device: {device}\n")

    # -----------------------------------------------------------------
    # Build a two-layer hierarchy to exercise all methods properly.
    #
    #   Layer 2 (top):  dim=16  --predict_down-->  Layer 1:  dim=64
    #   Layer 1:        dim=64  --predict_down-->  Layer 0 (sensory): dim=32
    #
    # layer_2: input_dim=16, output_dim=64  (predicts layer 1)
    # layer_1: input_dim=64, output_dim=32  (predicts layer 0 / sensory)
    # -----------------------------------------------------------------

    batch_size = 8

    layer_2 = PCLayer(input_dim=16, output_dim=64, activation_fn_name="tanh").to(device)
    layer_1 = PCLayer(input_dim=64, output_dim=32, activation_fn_name="tanh").to(device)

    print("--- Layer 2 (top) ---")
    print(layer_2)
    print(f"  weight         shape : {layer_2.weight.shape}")
    print(f"  bias           shape : {layer_2.bias.shape}")
    print(f"  log_var        shape : {layer_2.log_var.shape}")
    print(f"  lateral_weight shape : {layer_2.lateral_weight.shape}")

    print("\n--- Layer 1 ---")
    print(layer_1)
    print(f"  weight         shape : {layer_1.weight.shape}")
    print(f"  bias           shape : {layer_1.bias.shape}")
    print(f"  log_var        shape : {layer_1.log_var.shape}")
    print(f"  lateral_weight shape : {layer_1.lateral_weight.shape}")

    # Verify lateral diagonal is zero at init.
    diag_2 = layer_2.lateral_weight.data.diag()
    assert diag_2.abs().max().item() == 0.0, "Lateral diagonal not zero at init!"
    print(f"\n  layer_2 lateral diag max: {diag_2.abs().max().item()}  (zero)  [OK]")
    lat_mean_2 = layer_2.lateral_weight.data.abs().mean().item()
    print(f"  layer_2 lateral |mean|: {lat_mean_2:.6f}")

    # Verify precision at init (log_var=0 -> precision=1).
    prec_2 = layer_2.get_precision()
    assert prec_2.shape == (16,)
    assert torch.allclose(prec_2, torch.ones(16, device=device)), "Initial precision should be 1.0!"
    print(f"  layer_2 precision at init: {prec_2[:4].tolist()} ...  (all 1.0)  [OK]")

    # Reset states for a new observation.
    layer_2.reset_states(batch_size, device)
    layer_1.reset_states(batch_size, device)
    print(f"\n  layer_2.x shape : {layer_2.x.shape}")
    print(f"  layer_1.x shape : {layer_1.x.shape}")

    # Layer 2 predicts down to layer 1.
    prediction_for_1 = layer_2.predict_down(layer_2.x)
    print(f"\n  layer_2.predict_down -> {prediction_for_1.shape}  (should be [8, 64])")
    assert prediction_for_1.shape == (batch_size, 64)

    # Layer 1 computes its error with lateral prediction included.
    error_1 = layer_1.compute_error(prediction_for_1)
    print(f"  layer_1.compute_error -> {error_1.shape}  (should be [8, 64])")
    assert error_1.shape == (batch_size, 64)

    # Layer 1 predicts down to sensory layer (dim=32).
    prediction_for_0 = layer_1.predict_down(layer_1.x)
    print(f"  layer_1.predict_down -> {prediction_for_0.shape}  (should be [8, 32])")
    assert prediction_for_0.shape == (batch_size, 32)

    # Verify analytical derivatives for each activation type.
    print("\n--- Analytical derivatives ---")
    for act_name in _ACTIVATION_REGISTRY:
        test_layer = PCLayer(16, 8, activation_fn_name=act_name).to(device)
        test_input = torch.randn(4, 16, device=device)
        d = test_layer.activation_deriv(test_input)
        print(f"  {act_name:>8s}  deriv shape={d.shape}  "
              f"range=[{d.min().item():.4f}, {d.max().item():.4f}]")

    # --- Precision & variance update test ---
    print("\n--- Precision / variance update ---")
    layer_1.e = torch.randn(batch_size, 64, device=device) * 0.5
    lv_before = layer_1.log_var.data.clone()
    layer_1.update_variance(eta_v=0.01)
    lv_delta = (layer_1.log_var.data - lv_before).abs().sum().item()
    assert lv_delta > 0, "log_var did not change after update_variance!"
    print(f"  log_var delta after update_variance: {lv_delta:.6f}  [OK]")

    # --- Lateral update test ---
    print("\n--- Lateral weight update ---")
    layer_1.x = torch.randn(batch_size, 64, device=device)
    layer_1.e = torch.randn(batch_size, 64, device=device) * 0.3
    lat_before = layer_1.lateral_weight.data.clone()
    lat_mean_before = lat_before.abs().mean().item()

    # Fake pe_below and pe_i for testing.
    fake_pe_below = torch.randn(batch_size, 32, device=device) * 0.1
    fake_pe_i = layer_1.e * layer_1.get_precision()

    layer_1.update_weights(fake_pe_below, fake_pe_i, eta_w=0.001, eta_l=0.01)

    lat_after = layer_1.lateral_weight.data
    lat_mean_after = lat_after.abs().mean().item()
    lat_delta = (lat_after - lat_before).abs().sum().item()
    assert lat_delta > 0, "Lateral weights did not change!"
    print(f"  lateral |mean| before: {lat_mean_before:.6f}")
    print(f"  lateral |mean| after : {lat_mean_after:.6f}")
    print(f"  lateral total delta  : {lat_delta:.6f}  [OK]")

    # Verify diagonal is still zero after update.
    diag_after = lat_after.diag()
    assert diag_after.abs().max().item() == 0.0, "Lateral diagonal not zero after update!"
    print(f"  lateral diag max after update: {diag_after.abs().max().item()}  (zero)  [OK]")

    # Confirm no gradients anywhere.
    for layer_name, layer in [("layer_2", layer_2), ("layer_1", layer_1)]:
        for name, param in layer.named_parameters():
            assert not param.requires_grad, f"{layer_name}.{name} has requires_grad=True!"
    print("\n  All parameters have requires_grad=False  [OK]")

    # Verify reset clears state.
    layer_1.x.fill_(99.0)
    layer_1.reset_states(batch_size, device)
    assert layer_1.x.abs().max().item() == 0.0, "reset_states did not zero x!"
    print("  reset_states zeros out state  [OK]")

    print("\nPhase 5b layer smoke test passed.")
