"""
Cortical Column — Phase 1: The Sensorimotor Column

A biologically-structured cortical column implementing the Thousand Brains
theory.  Replaces the monolithic PCLayer with a CorticalColumn that natively
separates object identity ("What" / L2/3), sensor location ("Where" / L6),
and motor output ("Action" / L5).

Mathematical foundation (Free Energy Principle):
    - L2/3 generative prediction (composite):
          prediction = activation(W_obj @ x_obj + W_loc @ x_loc)
    - L4 sensory error:
          error = sensory_input - prediction
    - Internal ODE settling (dual state):
          dx_obj = W_obj.T @ error
          dx_loc = W_loc.T @ error
    - L5 motor action (Active Inference):
          action = eta_a * (sensory_gradient.T @ error)
    - Hebbian learning:
          dW_obj = eta_w * (error @ x_obj.T)
          dW_loc = eta_w * (error @ x_loc.T)

No autograd. No backpropagation. All dynamics are local ODEs
minimising Free Energy (prediction error).

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
    """Return (sigmoid, d_sigmoid/dx) where the derivative is sig(x)(1-sig(x))."""
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
# CorticalColumn
# ---------------------------------------------------------------------------

class CorticalColumn(nn.Module):
    """A single Cortical Column with separated What/Where/Action streams.

    Implements the Thousand Brains theory at the single-column level.
    Internally maintains two latent state vectors and two generative
    weight matrices that jointly predict sensory input:

    States (fast dynamics — settle each observation):
        x_obj : (batch, obj_dim)  — L2/3 "What" belief (object identity).
        x_loc : (batch, loc_dim)  — L6 "Where" belief (sensor location).
        error : (batch, sensory_dim) — L4 prediction error.

    Parameters (slow dynamics — Hebbian learning):
        W_obj : (sensory_dim, obj_dim) — generative weights for object.
        W_loc : (sensory_dim, loc_dim) — generative weights for location.

    The composite generative prediction sent from L2/3 down to L4:
        prediction = activation(W_obj @ x_obj + W_loc @ x_loc)

    L5 motor output computes Active Inference action commands from
    the spatial gradient of sensory input and the L4 error.

    Args:
        obj_dim:     Dimensionality of the "What" state (L2/3).
        loc_dim:     Dimensionality of the "Where" state (L6).
        sensory_dim: Dimensionality of the sensory input (L4 target).
        activation_fn_name: One of 'tanh', 'relu', 'sigmoid'.
    """

    W_obj: Tensor
    W_loc: Tensor
    x_obj: Optional[Tensor]
    x_loc: Optional[Tensor]
    error: Optional[Tensor]

    def __init__(
        self,
        obj_dim: int,
        loc_dim: int,
        sensory_dim: int,
        activation_fn_name: str = "tanh",
    ) -> None:
        super().__init__()

        if activation_fn_name not in _ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unknown activation '{activation_fn_name}'. "
                f"Choose from {list(_ACTIVATION_REGISTRY.keys())}."
            )

        self.obj_dim = obj_dim
        self.loc_dim = loc_dim
        self.sensory_dim = sensory_dim
        self.activation_fn_name = activation_fn_name

        # Activation function and its analytical derivative.
        self.activation_fn, self.activation_deriv = _ACTIVATION_REGISTRY[
            activation_fn_name
        ]()

        # --- Slow parameters (no autograd) ---
        # W_obj: (sensory_dim, obj_dim) — "What" generative weights.
        w_obj = torch.empty(sensory_dim, obj_dim)
        nn.init.kaiming_uniform_(w_obj, a=math.sqrt(5))
        self.W_obj = nn.Parameter(w_obj, requires_grad=False)

        # W_loc: (sensory_dim, loc_dim) — "Where" generative weights.
        w_loc = torch.empty(sensory_dim, loc_dim)
        nn.init.kaiming_uniform_(w_loc, a=math.sqrt(5))
        self.W_loc = nn.Parameter(w_loc, requires_grad=False)

        # --- Fast states (initialised lazily per observation) ---
        self.x_obj = None
        self.x_loc = None
        self.error = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_states(
        self, batch_size: int, device: Optional[torch.device] = None,
    ) -> None:
        """Zero-initialise all fast states for a new observation.

        Args:
            batch_size: Number of samples in the current batch.
            device:     Target device (defaults to W_obj's device).
        """
        if device is None:
            device = self.W_obj.device

        self.x_obj = torch.zeros(batch_size, self.obj_dim, device=device)
        self.x_loc = torch.zeros(batch_size, self.loc_dim, device=device)
        self.error = torch.zeros(batch_size, self.sensory_dim, device=device)

    # ------------------------------------------------------------------
    # L2/3 → L4: Generative prediction
    # ------------------------------------------------------------------

    def predict_down(self) -> Tensor:
        """Compute the composite L2/3 generative prediction for L4.

        .. math::
            p = \\text{activation}(W_{obj} \\cdot x_{obj} + W_{loc} \\cdot x_{loc})

        Returns:
            prediction: (batch, sensory_dim)
        """
        # x_obj: (B, obj_dim), W_obj: (sensory_dim, obj_dim)
        # x_loc: (B, loc_dim), W_loc: (sensory_dim, loc_dim)
        composite = (
            self.x_obj @ self.W_obj.t()    # (B, sensory_dim)
            + self.x_loc @ self.W_loc.t()  # (B, sensory_dim)
        )
        return self.activation_fn(composite)

    # ------------------------------------------------------------------
    # L4: Sensory error
    # ------------------------------------------------------------------

    def compute_error(self, sensory_input: Tensor) -> Tensor:
        """Compute and store the L4 prediction error.

        .. math::
            \\epsilon = \\text{sensory\\_input} - \\text{prediction}

        Args:
            sensory_input: (batch, sensory_dim) — observed data.

        Returns:
            error: (batch, sensory_dim)
        """
        prediction = self.predict_down()
        self.error = sensory_input - prediction
        return self.error

    # ------------------------------------------------------------------
    # Internal ODE settling
    # ------------------------------------------------------------------

    def infer_step(
        self,
        sensory_input: Tensor,
        eta_x: float,
        freeze_obj: bool = False,
    ) -> None:
        """One Euler step of the dual-state settling ODE.

        Both x_obj and x_loc evolve simultaneously to minimise the L4
        prediction error.  The gradient flows through the activation
        derivative to respect the nonlinearity.

        .. math::
            dx_{obj} = W_{obj}^T \\cdot \\text{error\\_grad}
            dx_{loc} = W_{loc}^T \\cdot \\text{error\\_grad}

        where error_grad = error * activation'(composite) accounts for
        the chain rule through the activation function.

        When ``freeze_obj=True``, only x_loc updates.  This is used
        during sensorimotor tracking: L2/3 object identity is stable
        across saccades while L6 location adapts to each new view.

        Args:
            sensory_input: (batch, sensory_dim) — observed data.
            eta_x: Step size for belief updates.
            freeze_obj: If True, hold x_obj fixed (only x_loc settles).
        """
        # Recompute composite pre-activation and prediction.
        composite = (
            self.x_obj @ self.W_obj.t()
            + self.x_loc @ self.W_loc.t()
        )
        prediction = self.activation_fn(composite)
        self.error = sensory_input - prediction

        # Chain rule: error gradient through activation derivative.
        deriv = self.activation_deriv(composite)  # (B, sensory_dim)
        error_grad = self.error * deriv           # (B, sensory_dim)

        # Dual-state update: both states chase the same error signal.
        if not freeze_obj:
            dx_obj = error_grad @ self.W_obj   # (B, obj_dim)
            self.x_obj = self.x_obj + eta_x * dx_obj

        dx_loc = error_grad @ self.W_loc   # (B, loc_dim)
        self.x_loc = self.x_loc + eta_x * dx_loc

    # ------------------------------------------------------------------
    # L5: Motor action (Active Inference)
    # ------------------------------------------------------------------

    def get_motor_action(
        self,
        sensory_gradient: Tensor,
        eta_a: float = 0.01,
    ) -> Tensor:
        """Compute Active Inference motor commands from L5.

        The motor system acts on the *world* to reduce prediction error
        by moving the sensor.  The action is the gradient of free energy
        w.r.t. the sensory input, projected through the spatial gradient
        of the sensor.

        .. math::
            a = \\eta_a \\cdot (\\nabla_{\\text{spatial}} s)^T \\cdot \\epsilon

        Args:
            sensory_gradient: (batch, sensory_dim, action_dim) — spatial
                gradient of the sensory input w.r.t. motor coordinates
                (e.g., [dx, dy] shift of the fovea).
            eta_a: Motor learning rate / gain.

        Returns:
            action: (batch, action_dim) — motor velocity commands.
        """
        if self.error is None:
            raise RuntimeError(
                "No error computed. Call infer_step() before get_motor_action()."
            )

        # Active Inference: minimise free energy by acting on the world.
        #   dF/da = error * ds/da   (chain rule through sensory input)
        #   action = -dF/da         (gradient DESCENT on free energy)
        #
        # error: (B, sensory_dim)
        # sensory_gradient: (B, sensory_dim, action_dim)
        # result: (B, action_dim)
        action = -eta_a * torch.einsum("bs,bsa->ba", self.error, sensory_gradient)
        return action

    # ------------------------------------------------------------------
    # Learning (slow dynamics — Hebbian)
    # ------------------------------------------------------------------

    def learn(self, eta_w: float = 0.001) -> None:
        """Hebbian update for both generative weight matrices.

        Called *after* the inference loop has settled.

        .. math::
            \\Delta W_{obj} = \\eta_w \\cdot \\epsilon^T \\cdot x_{obj}
            \\Delta W_{loc} = \\eta_w \\cdot \\epsilon^T \\cdot x_{loc}

        Averaged over the batch.

        Args:
            eta_w: Learning rate for weight updates.
        """
        if self.error is None or self.x_obj is None:
            raise RuntimeError(
                "States not initialised. Run inference before learn()."
            )

        batch_size = self.error.shape[0]

        # dW_obj: (sensory_dim, B) @ (B, obj_dim) -> (sensory_dim, obj_dim)
        dW_obj = (self.error.t() @ self.x_obj) / batch_size
        dW_loc = (self.error.t() @ self.x_loc) / batch_size

        self.W_obj.data += eta_w * dW_obj
        self.W_loc.data += eta_w * dW_loc

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_energy(self) -> float:
        """Return the current free energy (sum of squared prediction error).

        .. math::
            F = \\frac{1}{2} \\sum \\epsilon^2
        """
        if self.error is None:
            return 0.0
        return 0.5 * (self.error * self.error).sum().item()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"obj_dim={self.obj_dim}, loc_dim={self.loc_dim}, "
            f"sensory_dim={self.sensory_dim}, activation={self.activation_fn_name}"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = _get_device()
    print(f"Using device: {device}\n")

    col = CorticalColumn(
        obj_dim=8, loc_dim=4, sensory_dim=25, activation_fn_name="tanh",
    ).to(device)
    print(col)

    B = 1
    col.reset_states(B, device)
    print(f"\n  x_obj shape: {col.x_obj.shape}  (expect [1, 8])")
    print(f"  x_loc shape: {col.x_loc.shape}  (expect [1, 4])")

    # Fake sensory input.
    sensory = torch.randn(B, 25, device=device)

    # Settle for 20 steps.
    energies = []
    for step in range(20):
        col.infer_step(sensory, eta_x=0.1)
        energies.append(col.get_energy())

    print(f"\n  Energy step  0: {energies[0]:.4f}")
    print(f"  Energy step 19: {energies[-1]:.4f}")
    assert energies[-1] < energies[0], "Energy did not decrease!"
    print("  Energy decreased  [OK]")

    # Motor action test.
    grad = torch.randn(B, 25, 2, device=device)
    action = col.get_motor_action(grad, eta_a=0.01)
    print(f"\n  Motor action shape: {action.shape}  (expect [1, 2])")
    assert action.shape == (B, 2)
    print("  Motor action computed  [OK]")

    # Learning test.
    w_obj_before = col.W_obj.data.clone()
    col.learn(eta_w=0.01)
    w_delta = (col.W_obj.data - w_obj_before).abs().sum().item()
    assert w_delta > 0, "Weights did not change!"
    print(f"  W_obj delta after learn: {w_delta:.6f}  [OK]")

    # No autograd.
    for name, param in col.named_parameters():
        assert not param.requires_grad, f"{name} has requires_grad=True!"
    print("  All parameters requires_grad=False  [OK]")

    print("\nCorticalColumn smoke test passed.")
