"""
Cortical Column — Phase 4: Temporal Predictive Coding

A biologically-structured cortical column implementing the Thousand Brains
theory.  Each CorticalColumn natively separates object identity ("What" /
L2/3), sensor location ("Where" / L6), and motor output ("Action" / L5).

Phase 2 adds **lateral connections** (W_lat) between neighbouring columns.
Phase 3 adds **top-down priors** from a higher-level column, enabling
hierarchical attention.
Phase 4 adds **temporal prediction** (W_trans) — the column remembers its
previous belief (x_obj_prev) and learns a transition matrix that predicts
how beliefs evolve over time.  This enables the network to learn simple
physics (kinematics) and "dream" future states with zero sensory input.

Mathematical foundation (Free Energy Principle):
    - L2/3 generative prediction (composite):
          prediction = activation(W_obj @ x_obj + W_loc @ x_loc)
    - L4 sensory error:
          err_sensory = sensory_input - prediction
    - Lateral consensus error (Phase 2):
          err_lat = x_obj - W_lat @ neighbor_context
    - Top-down attention error (Phase 3):
          err_td = x_obj - top_down_prior
    - Temporal prediction error (Phase 4):
          err_time = x_obj - W_trans @ x_obj_prev
    - Internal ODE settling (dual state + lateral + top-down + temporal):
          dx_obj = (W_obj.T @ err_sensory_grad) - err_lat - err_td - err_time
          dx_loc = W_loc.T @ err_sensory_grad
    - L5 motor action (Active Inference):
          action = -eta_a * (sensory_gradient.T @ error)
    - Hebbian learning:
          dW_obj = eta_w * (err_sensory @ x_obj.T)
          dW_loc = eta_w * (err_sensory @ x_loc.T)
          dW_lat = eta_w * (err_lat @ neighbor_context.T)
          dW_trans = eta_w * (err_time @ x_obj_prev.T)

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
    """A single Cortical Column with separated What/Where/Action streams,
    lateral connections for inter-column consensus, top-down hierarchical
    attention, and temporal prediction.

    Implements the Thousand Brains theory at the single-column level.
    Internally maintains two latent state vectors and four weight
    matrices that jointly predict sensory input, align with neighbours,
    and anticipate temporal transitions:

    States (fast dynamics — settle each observation):
        x_obj     : (batch, obj_dim)     — L2/3 "What" belief (object identity).
        x_loc     : (batch, loc_dim)     — L6 "Where" belief (sensor location).
        error     : (batch, sensory_dim) — L4 sensory prediction error.
        err_lat   : (batch, obj_dim)     — lateral consensus error (Phase 2).
        err_td    : (batch, obj_dim)     — top-down attention error (Phase 3).
        err_time  : (batch, obj_dim)     — temporal prediction error (Phase 4).

    Temporal memory (persists across observations within a sequence):
        x_obj_prev : (batch, obj_dim)    — previous x_obj (set by step_time()).

    Parameters (slow dynamics — Hebbian learning):
        W_obj   : (sensory_dim, obj_dim) — generative weights for object.
        W_loc   : (sensory_dim, loc_dim) — generative weights for location.
        W_lat   : (obj_dim, obj_dim)     — lateral weights mapping neighbour
                  context to predicted local x_obj (Phase 2).
        W_trans : (obj_dim, obj_dim)     — temporal transition matrix mapping
                  previous belief to predicted current belief (Phase 4).

    The composite generative prediction sent from L2/3 down to L4:
        prediction = activation(W_obj @ x_obj + W_loc @ x_loc)

    Lateral consensus (Phase 2):
        err_lat = x_obj - W_lat @ neighbor_context
        dx_obj += -err_lat   (pulls toward neighbour agreement)

    Top-down attention (Phase 3):
        err_td = x_obj - top_down_prior
        dx_obj += -err_td    (pulls toward higher-level expectation)

    Temporal prediction (Phase 4):
        err_time = x_obj - W_trans @ x_obj_prev
        dx_obj += -err_time  (pulls toward temporally predicted state)

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
    W_lat: Tensor
    W_trans: Tensor
    x_obj: Optional[Tensor]
    x_loc: Optional[Tensor]
    error: Optional[Tensor]
    err_lat: Optional[Tensor]
    err_td: Optional[Tensor]
    err_time: Optional[Tensor]
    x_obj_prev: Optional[Tensor]

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

        # W_lat: (obj_dim, obj_dim) — lateral weights for inter-column
        # consensus (Phase 2).  Initialized to identity so that the
        # column initially expects its neighbours to have the same x_obj.
        self.W_lat = nn.Parameter(
            torch.eye(obj_dim), requires_grad=False,
        )

        # W_trans: (obj_dim, obj_dim) — temporal transition matrix (Phase 4).
        # Initialized to identity: default assumption is things stay the same.
        self.W_trans = nn.Parameter(
            torch.eye(obj_dim), requires_grad=False,
        )

        # --- Fast states (initialised lazily per observation) ---
        self.x_obj = None
        self.x_loc = None
        self.error = None
        self.err_lat = None       # lateral consensus error (Phase 2)
        self.err_td = None        # top-down attention error (Phase 3)
        self.err_time = None      # temporal prediction error (Phase 4)

        # --- Temporal memory (persists across observations in a sequence) ---
        self.x_obj_prev = None    # set by step_time(), NOT zeroed by reset_states()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_states(
        self, batch_size: int, device: Optional[torch.device] = None,
    ) -> None:
        """Zero-initialise all fast states for a new observation.

        Note: x_obj_prev is NOT zeroed — temporal memory persists across
        observations within a sequence.  Only step_time() or direct
        assignment modifies x_obj_prev.

        Args:
            batch_size: Number of samples in the current batch.
            device:     Target device (defaults to W_obj's device).
        """
        if device is None:
            device = self.W_obj.device

        self.x_obj = torch.zeros(batch_size, self.obj_dim, device=device)
        self.x_loc = torch.zeros(batch_size, self.loc_dim, device=device)
        self.error = torch.zeros(batch_size, self.sensory_dim, device=device)
        self.err_lat = torch.zeros(batch_size, self.obj_dim, device=device)
        self.err_td = torch.zeros(batch_size, self.obj_dim, device=device)
        self.err_time = torch.zeros(batch_size, self.obj_dim, device=device)

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
        neighbor_context: Optional[Tensor] = None,
        top_down_prior: Optional[Tensor] = None,
    ) -> None:
        """One Euler step of the dual-state settling ODE.

        Both x_obj and x_loc evolve simultaneously to minimise the L4
        prediction error.  The gradient flows through the activation
        derivative to respect the nonlinearity.

        Phase 2 adds lateral consensus: when ``neighbor_context`` is
        provided, x_obj is additionally pulled toward the lateral
        prediction (W_lat @ neighbor_context).

        Phase 3 adds top-down attention: when ``top_down_prior`` is
        provided, x_obj is additionally pulled toward the higher-level
        expectation, enabling hierarchical directed attention.

        .. math::
            \\text{err\\_sensory} = \\text{input} - \\text{prediction}
            \\text{err\\_lat} = x_{obj} - W_{lat} \\cdot \\bar{x}_{obj}^{\\text{neighbors}}
            \\text{err\\_td} = x_{obj} - \\text{top\\_down\\_prior}
            dx_{obj} = W_{obj}^T \\cdot \\text{err\\_grad} - \\text{err\\_lat} - \\text{err\\_td}
            dx_{loc} = W_{loc}^T \\cdot \\text{err\\_grad}

        When ``freeze_obj=True``, only x_loc updates.  This is used
        during sensorimotor tracking: L2/3 object identity is stable
        across saccades while L6 location adapts to each new view.

        Args:
            sensory_input: (batch, sensory_dim) — observed data.
            eta_x: Step size for belief updates.
            freeze_obj: If True, hold x_obj fixed (only x_loc settles).
            neighbor_context: (batch, obj_dim) — average x_obj of spatial
                neighbours.  If None, no lateral pressure is applied
                (Phase 1 backward-compatible behaviour).
            top_down_prior: (batch, obj_dim) — expected x_obj from a
                higher-level column.  If None, no top-down pressure is
                applied (Phase 1/2 backward-compatible behaviour).
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
            # Bottom-up sensory pressure.
            dx_obj = error_grad @ self.W_obj   # (B, obj_dim)

            # Lateral consensus pressure (Phase 2).
            if neighbor_context is not None:
                # Lateral error: how far is x_obj from what neighbours predict?
                lateral_pred = neighbor_context @ self.W_lat.t()  # (B, obj_dim)
                self.err_lat = self.x_obj - lateral_pred
                dx_obj = dx_obj - self.err_lat
            else:
                self.err_lat = torch.zeros_like(self.x_obj)

            # Top-down hierarchical pressure (Phase 3).
            if top_down_prior is not None:
                # Top-down error: how far is x_obj from what the
                # higher level expects?
                self.err_td = self.x_obj - top_down_prior  # (B, obj_dim)
                dx_obj = dx_obj - self.err_td
            else:
                self.err_td = torch.zeros_like(self.x_obj)

            # Temporal prediction pressure (Phase 4).
            if self.x_obj_prev is not None:
                temporal_pred = self.x_obj_prev @ self.W_trans.t()  # (B, obj_dim)
                self.err_time = self.x_obj - temporal_pred
                dx_obj = dx_obj - self.err_time
            else:
                self.err_time = torch.zeros_like(self.x_obj)

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

    def learn(
        self,
        eta_w: float = 0.001,
        neighbor_context: Optional[Tensor] = None,
    ) -> None:
        """Hebbian update for generative, lateral, and temporal weight matrices.

        Called *after* the inference loop has settled.

        .. math::
            \\Delta W_{obj}   = \\eta_w \\cdot \\epsilon^T \\cdot x_{obj}
            \\Delta W_{loc}   = \\eta_w \\cdot \\epsilon^T \\cdot x_{loc}
            \\Delta W_{lat}   = \\eta_w \\cdot \\text{err\\_lat}^T \\cdot \\bar{x}_{obj}^{\\text{nbr}}
            \\Delta W_{trans} = \\eta_w \\cdot \\text{err\\_time}^T \\cdot x_{obj\\_prev}

        Averaged over the batch.

        Args:
            eta_w: Learning rate for weight updates.
            neighbor_context: (batch, obj_dim) — average x_obj of spatial
                neighbours.  If provided, W_lat is updated to associate
                this column's state with its neighbours' consensus.
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

        # Lateral weight update (Phase 2).
        if neighbor_context is not None and self.err_lat is not None:
            # dW_lat: (obj_dim, B) @ (B, obj_dim) -> (obj_dim, obj_dim)
            dW_lat = (self.err_lat.t() @ neighbor_context) / batch_size
            self.W_lat.data += eta_w * dW_lat

        # Temporal weight update (Phase 4).
        if self.x_obj_prev is not None and self.err_time is not None:
            # dW_trans: (obj_dim, B) @ (B, obj_dim) -> (obj_dim, obj_dim)
            dW_trans = (self.err_time.t() @ self.x_obj_prev) / batch_size
            self.W_trans.data += eta_w * dW_trans

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_energy(self) -> float:
        """Return the current free energy (sensory + lateral + top-down + temporal).

        .. math::
            F = \\frac{1}{2} \\sum \\epsilon_{\\text{sensory}}^2
              + \\frac{1}{2} \\sum \\epsilon_{\\text{lateral}}^2
              + \\frac{1}{2} \\sum \\epsilon_{\\text{top-down}}^2
              + \\frac{1}{2} \\sum \\epsilon_{\\text{temporal}}^2
        """
        e = 0.0
        if self.error is not None:
            e += 0.5 * (self.error * self.error).sum().item()
        if self.err_lat is not None:
            e += 0.5 * (self.err_lat * self.err_lat).sum().item()
        if self.err_td is not None:
            e += 0.5 * (self.err_td * self.err_td).sum().item()
        if self.err_time is not None:
            e += 0.5 * (self.err_time * self.err_time).sum().item()
        return e

    # ------------------------------------------------------------------
    # Temporal stepping
    # ------------------------------------------------------------------

    def step_time(self) -> None:
        """Shift the present belief into temporal memory (the global clock tick).

        Copies the current x_obj into x_obj_prev.  This should be called
        once per observation *after* inference and learning have completed,
        before moving to the next frame in a sequence.
        """
        if self.x_obj is not None:
            self.x_obj_prev = self.x_obj.clone().detach()

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

    # Learning test (sensory weights).
    w_obj_before = col.W_obj.data.clone()
    col.learn(eta_w=0.01)
    w_delta = (col.W_obj.data - w_obj_before).abs().sum().item()
    assert w_delta > 0, "Weights did not change!"
    print(f"  W_obj delta after learn: {w_delta:.6f}  [OK]")

    # Lateral connection test.
    print(f"\n  W_lat shape: {col.W_lat.shape}  (expect [{col.obj_dim}, {col.obj_dim}])")
    assert col.W_lat.shape == (col.obj_dim, col.obj_dim)
    # W_lat should be identity at init.
    assert torch.allclose(col.W_lat.data, torch.eye(col.obj_dim, device=device)), \
        "W_lat not identity at init!"
    print("  W_lat is identity at init  [OK]")

    # Test infer with neighbor context.
    col.reset_states(B, device)
    nbr = torch.randn(B, col.obj_dim, device=device)
    for step in range(10):
        col.infer_step(sensory, eta_x=0.1, neighbor_context=nbr)
    assert col.err_lat is not None
    print(f"  err_lat shape: {col.err_lat.shape}  (expect [1, {col.obj_dim}])")
    print(f"  err_lat norm : {col.err_lat.norm().item():.4f}  [OK]")

    # Lateral learning test.
    w_lat_before = col.W_lat.data.clone()
    col.learn(eta_w=0.01, neighbor_context=nbr)
    w_lat_delta = (col.W_lat.data - w_lat_before).abs().sum().item()
    assert w_lat_delta > 0, "W_lat did not change after lateral learn!"
    print(f"  W_lat delta after learn: {w_lat_delta:.6f}  [OK]")

    # Top-down prior test (Phase 3).
    col.reset_states(B, device)
    td_prior = torch.randn(B, col.obj_dim, device=device)
    for step in range(10):
        col.infer_step(sensory, eta_x=0.1, top_down_prior=td_prior)
    assert col.err_td is not None
    print(f"\n  err_td shape: {col.err_td.shape}  (expect [1, {col.obj_dim}])")
    print(f"  err_td norm : {col.err_td.norm().item():.4f}  [OK]")

    # Combined lateral + top-down test.
    col.reset_states(B, device)
    for step in range(10):
        col.infer_step(
            sensory, eta_x=0.1,
            neighbor_context=nbr, top_down_prior=td_prior,
        )
    combined_energy = col.get_energy()
    print(f"  Combined (lat+td) energy: {combined_energy:.4f}  [OK]")

    # Temporal prediction test (Phase 4).
    print("\n  --- Phase 4: Temporal prediction ---")
    col.reset_states(B, device)

    # W_trans should be identity at init.
    assert torch.allclose(col.W_trans.data, torch.eye(col.obj_dim, device=device)), \
        "W_trans not identity at init!"
    print(f"  W_trans shape: {col.W_trans.shape}  (expect [{col.obj_dim}, {col.obj_dim}])")
    print("  W_trans is identity at init  [OK]")

    # x_obj_prev is None at start — no temporal pressure.
    assert col.x_obj_prev is None, "x_obj_prev should be None before step_time!"
    for step in range(10):
        col.infer_step(sensory, eta_x=0.1)
    assert col.err_time is not None
    assert col.err_time.norm().item() < 1e-6, \
        "err_time should be zero when x_obj_prev is None!"
    print("  No temporal pressure when x_obj_prev=None  [OK]")

    # Call step_time to store current x_obj.
    col.step_time()
    assert col.x_obj_prev is not None, "x_obj_prev should be set after step_time!"
    prev_norm = col.x_obj_prev.norm().item()
    print(f"  x_obj_prev norm after step_time: {prev_norm:.4f}  [OK]")

    # Now settle again with a different sensory input — temporal pressure should appear.
    sensory2 = torch.randn(B, 25, device=device)
    col.reset_states(B, device)
    # Verify x_obj_prev persists across reset_states.
    assert col.x_obj_prev is not None, "x_obj_prev should persist across reset_states!"
    print("  x_obj_prev persists across reset_states  [OK]")

    for step in range(10):
        col.infer_step(sensory2, eta_x=0.1)
    assert col.err_time is not None
    err_time_norm = col.err_time.norm().item()
    assert err_time_norm > 0.001, "err_time should be non-zero with x_obj_prev set!"
    print(f"  err_time norm (with temporal): {err_time_norm:.4f}  [OK]")

    # Temporal learning test.
    w_trans_before = col.W_trans.data.clone()
    col.learn(eta_w=0.01)
    w_trans_delta = (col.W_trans.data - w_trans_before).abs().sum().item()
    assert w_trans_delta > 0, "W_trans did not change after temporal learn!"
    print(f"  W_trans delta after learn: {w_trans_delta:.6f}  [OK]")

    # Energy includes temporal term.
    energy_with_time = col.get_energy()
    print(f"  Energy (with temporal): {energy_with_time:.4f}  [OK]")

    # No autograd.
    for name, param in col.named_parameters():
        assert not param.requires_grad, f"{name} has requires_grad=True!"
    print("  All parameters requires_grad=False  [OK]")

    print("\nCorticalColumn smoke test passed.")
