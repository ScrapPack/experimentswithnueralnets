"""
Cortical Column — Convention B (standard FEP: ε = prediction − input)

A biologically-structured cortical column implementing the Thousand Brains
theory.  Each CorticalColumn natively separates object identity ("What" /
L2/3), sensor location ("Where" / L6), and motor output ("Action" / L5).

Phase 2 adds **lateral connections** (W_lat) between neighbouring columns.
Phase 3 adds **top-down priors** from a higher-level column, enabling
hierarchical attention.
Phase 4 adds **temporal prediction** (W_trans) with memory (x_obj_prev).
Phase 4.5 upgrades to **dendritic gating** (pre-multiplication activation)
and **non-linear temporal transitions**, with overcomplete latent spaces.
Phase 4.6 adds **precision weighting** (Π scalars) for each error modality
and a **Gaussian prior** (state decay) on latent variables.

Error convention: ε = prediction − input (Bogacz 2017, PyMDP standard).
With this convention, ∇F aligns with the raw error products, so every
ODE and weight update uses strict subtraction for gradient descent.

Mathematical foundation (Free Energy Principle):
    - L2/3 generative prediction (pre-multiplication activation):
          z_obj = W_obj @ x_obj
          z_loc = W_loc @ x_loc
          pred_obj = activation(z_obj)              # bounded independently
          pred_loc = activation(z_loc)              # bounded independently
          prediction = pred_obj * pred_loc          # element-wise gating
    - L4 sensory error (Convention B):
          ε = prediction - sensory_input
    - Scaled errors (product-rule chain rule, two independent derivatives):
          err_scaled_obj = ε * pred_loc * activation_deriv(z_obj)
          err_scaled_loc = ε * pred_obj * activation_deriv(z_loc)
    - Lateral consensus error (Phase 2):
          err_lat = x_obj - W_lat @ neighbor_context
    - Top-down attention error (Phase 3):
          err_td = x_obj - top_down_prior
    - Temporal prediction error (Phase 4, non-linear):
          err_time = x_obj - tanh(W_trans @ x_obj_prev)
    - Internal ODE settling (gradient descent, all terms subtract):
          dx_obj = - π_s * W_obj.T @ err_scaled_obj
                   - π_l * err_lat - π_td * err_td - π_t * err_time
                   - α * x_obj
          dx_loc = - π_s * W_loc.T @ err_scaled_loc - α * x_loc
    - L5 motor action (Active Inference, precision-scaled):
          action = +eta_a * π_s * (sensory_gradient.T @ ε)
    - Variational Free Energy (precision-weighted):
          F = ½ π_s ||ε||² + ½ π_l ||err_lat||²
            + ½ π_td ||err_td||² + ½ π_t ||err_time||²
            + ½ α (||x_obj||² + ||x_loc||²)
    - Weight learning (precision-scaled gradient descent):
          W_obj -= η * (π_s * err_scaled_obj).T @ x_obj
          W_loc -= η * (π_s * err_scaled_loc).T @ x_loc
          W_lat += η * (π_l * err_lat).T @ neighbor_context     (prior: additive)
          W_trans += η * (π_t * err_time_scaled).T @ x_obj_prev  (prior: additive)
            where err_time_scaled = err_time * activation_deriv(temporal_z)

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
    attention, temporal prediction, and dendritic gating.

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

    Cached intermediates (set by infer_step, consumed by learn):
        _z_obj         : (batch, sensory_dim) — raw linear projection (pre-activation).
        _z_loc         : (batch, sensory_dim) — raw linear projection (pre-activation).
        _pred_obj      : (batch, sensory_dim) — activated object pathway.
        _pred_loc      : (batch, sensory_dim) — activated location pathway.
        _prediction    : (batch, sensory_dim) — pred_obj * pred_loc.
        _temporal_pred : (batch, obj_dim)     — tanh(W_trans @ x_obj_prev).

    Parameters (slow dynamics — Hebbian learning):
        W_obj   : (sensory_dim, obj_dim) — generative weights for object.
        W_loc   : (sensory_dim, loc_dim) — generative weights for location.
        W_lat   : (obj_dim, obj_dim)     — lateral weights (Phase 2).
        W_trans : (obj_dim, obj_dim)     — temporal transition matrix (Phase 4).

    Dendritic gating (pre-multiplication activation) — eliminates dead zone:
        z_obj = W_obj @ x_obj
        z_loc = W_loc @ x_loc
        pred_obj = activation(z_obj)      # bounded independently
        pred_loc = activation(z_loc)      # bounded independently
        prediction = pred_obj * pred_loc  # element-wise gating

    Convention B (ε = p − s): the product rule gives two gated gradients,
    and gradient descent uses strict subtraction on all terms:
        err_scaled_obj = error * pred_loc * activation_deriv(z_obj)
        err_scaled_loc = error * pred_obj * activation_deriv(z_loc)
        dx_obj = -W_obj.T @ err_scaled_obj - err_lat - err_td - err_time
        dx_loc = -W_loc.T @ err_scaled_loc
      [error = prediction - input]

    Non-linear temporal (Phase 4.5):
        temporal_pred = tanh(W_trans @ x_obj_prev)
        err_time = x_obj - temporal_pred

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
    _pred_obj: Optional[Tensor]
    _pred_loc: Optional[Tensor]
    _prediction: Optional[Tensor]
    _temporal_pred: Optional[Tensor]

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

        # --- Precision scalars (Π) — balance error modalities (Phase 4.6) ---
        # In FEP, each error term is weighted by the inverse variance
        # (precision) of the corresponding generative model component.
        # Higher precision = more trusted signal.
        self.pi_sensory = 1.0   # direct sensory — highest precision
        self.pi_lat = 0.5       # lateral consensus — slightly less precise
        self.pi_td = 0.5        # top-down attention — slightly less precise
        self.pi_time = 0.8      # temporal prediction — fairly precise

        # State decay constant (Gaussian prior on latent states).
        # In FEP, x has a standard normal prior N(0, 1/alpha), introducing
        # a -alpha * x term to the ODE that prevents runaway activations.
        self.alpha = 0.05

        # --- Cached intermediates for gated gradients ---
        self._z_obj = None          # x_obj @ W_obj.T  (pre-activation, for deriv)
        self._z_loc = None          # x_loc @ W_loc.T  (pre-activation, for deriv)
        self._pred_obj = None       # activation(z_obj)  (post-activation, for cross-gate)
        self._pred_loc = None       # activation(z_loc)  (post-activation, for cross-gate)
        self._prediction = None     # pred_obj * pred_loc  (final prediction)
        self._temporal_pred = None  # activation(x_obj_prev @ W_trans.T) (cached for learn)
        self._temporal_z = None     # x_obj_prev @ W_trans.T (pre-activation, for deriv)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_states(
        self, batch_size: int, device: Optional[torch.device] = None,
    ) -> None:
        """Initialise all fast states for a new observation.

        x_obj and x_loc are initialised to small random values (not zeros)
        to break symmetry and bootstrap the dendritic gate — with pure
        zeros, pred_obj * pred_loc = 0 and no gradient flows.

        Note: x_obj_prev is NOT zeroed — temporal memory persists across
        observations within a sequence.  Only step_time() or direct
        assignment modifies x_obj_prev.

        Args:
            batch_size: Number of samples in the current batch.
            device:     Target device (defaults to W_obj's device).
        """
        if device is None:
            device = self.W_obj.device

        # Random init to break symmetry and bootstrap the dendritic gate.
        # Scale 0.1 provides enough signal for gradients to flow through
        # the multiplicative interaction from the first step.
        self.x_obj = 0.1 * torch.randn(batch_size, self.obj_dim, device=device)
        self.x_loc = 0.1 * torch.randn(batch_size, self.loc_dim, device=device)
        self.error = torch.zeros(batch_size, self.sensory_dim, device=device)
        self.err_lat = torch.zeros(batch_size, self.obj_dim, device=device)
        self.err_td = torch.zeros(batch_size, self.obj_dim, device=device)
        self.err_time = torch.zeros(batch_size, self.obj_dim, device=device)

        # Clear cached intermediates.
        self._z_obj = None
        self._z_loc = None
        self._pred_obj = None
        self._pred_loc = None
        self._prediction = None
        self._temporal_pred = None
        self._temporal_z = None

    # ------------------------------------------------------------------
    # L2/3 → L4: Generative prediction
    # ------------------------------------------------------------------

    def predict_down(self) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the L2/3 generative prediction via dendritic gating.

        Pre-multiplication activation: activation is applied to each
        pathway independently BEFORE the multiplicative interaction.
        This keeps both streams bounded (e.g. [-1, 1] for tanh),
        preventing the pre-activation product from exploding into the
        activation's flat asymptotes (the "Asymptotic Dead Zone").

        .. math::
            z_{obj} = x_{obj} \\cdot W_{obj}^T
            z_{loc} = x_{loc} \\cdot W_{loc}^T
            p_{obj} = \\text{activation}(z_{obj})
            p_{loc} = \\text{activation}(z_{loc})
            \\text{prediction} = p_{obj} * p_{loc}

        Returns:
            (prediction, pred_obj, pred_loc) where:
                prediction: (batch, sensory_dim) — gated prediction.
                pred_obj:   (batch, sensory_dim) — activated object pathway.
                pred_loc:   (batch, sensory_dim) — activated location pathway.
        """
        z_obj = self.x_obj @ self.W_obj.t()       # (B, sensory_dim)
        z_loc = self.x_loc @ self.W_loc.t()       # (B, sensory_dim)
        pred_obj = self.activation_fn(z_obj)       # bounded
        pred_loc = self.activation_fn(z_loc)       # bounded
        prediction = pred_obj * pred_loc           # element-wise gating
        return prediction, pred_obj, pred_loc

    # ------------------------------------------------------------------
    # L4: Sensory error
    # ------------------------------------------------------------------

    def compute_error(self, sensory_input: Tensor) -> Tensor:
        """Compute and store the L4 prediction error (Convention B).

        Standard FEP convention: ε = prediction - input.

        .. math::
            \\epsilon = \\text{prediction} - \\text{sensory\\_input}

        Args:
            sensory_input: (batch, sensory_dim) — observed data.

        Returns:
            error: (batch, sensory_dim)
        """
        prediction, _, _ = self.predict_down()
        self.error = prediction - sensory_input
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
        """One Euler step of the dual-state settling ODE (Convention B).

        Convention B (ε = p − s): With this standard FEP error convention,
        ``grad_obj`` is the positive gradient ∇F.  All ODE terms use
        strict subtraction for gradient descent on free energy.

        Pre-multiplication activation: prediction = activation(z_obj) * activation(z_loc).
        Precision weighting and state decay (Phase 4.6).

        .. math::
            dx_{obj} = - \\pi_s \\cdot W_{obj}^T \\text{err\\_scaled\\_obj}
                       - \\pi_l \\cdot \\text{err\\_lat}
                       - \\pi_{td} \\cdot \\text{err\\_td}
                       - \\pi_t \\cdot \\text{err\\_time}
                       - \\alpha \\cdot x_{obj}
            dx_{loc} = - \\pi_s \\cdot W_{loc}^T \\text{err\\_scaled\\_loc}
                       - \\alpha \\cdot x_{loc}

        When ``freeze_obj=True``, only x_loc updates.

        Args:
            sensory_input: (batch, sensory_dim) — observed data.
            eta_x: Step size for belief updates.
            freeze_obj: If True, hold x_obj fixed (only x_loc settles).
            neighbor_context: (batch, obj_dim) — average x_obj of spatial
                neighbours.  If None, no lateral pressure is applied.
            top_down_prior: (batch, obj_dim) — expected x_obj from a
                higher-level column.  If None, no top-down pressure.
        """
        # Pre-multiplication activation: activate each pathway independently,
        # then multiply.  Keeps both streams bounded, avoiding the dead zone.
        z_obj = self.x_obj @ self.W_obj.t()        # (B, sensory_dim)
        z_loc = self.x_loc @ self.W_loc.t()        # (B, sensory_dim)
        pred_obj = self.activation_fn(z_obj)        # bounded (e.g. [-1, 1])
        pred_loc = self.activation_fn(z_loc)        # bounded (e.g. [-1, 1])
        prediction = pred_obj * pred_loc            # element-wise gating

        # Convention B (standard FEP): ε = prediction - input.
        # With this convention, grad_obj = ∇F (positive gradient of F),
        # so all ODE terms use strict subtraction for gradient descent.
        self.error = prediction - sensory_input

        # Cache intermediates for learn().
        self._z_obj = z_obj
        self._z_loc = z_loc
        self._pred_obj = pred_obj
        self._pred_loc = pred_loc
        self._prediction = prediction

        # Product-rule chain rule with two independent derivatives:
        #   prediction = f(z_obj) * f(z_loc)
        #   d(prediction)/d(z_obj) = f'(z_obj) * f(z_loc) = deriv_obj * pred_loc
        #   d(prediction)/d(z_loc) = f(z_obj) * f'(z_loc) = pred_obj * deriv_loc
        deriv_obj = self.activation_deriv(z_obj)    # (B, sensory_dim)
        deriv_loc = self.activation_deriv(z_loc)    # (B, sensory_dim)
        err_scaled_obj = self.error * pred_loc * deriv_obj  # (B, sensory_dim)
        err_scaled_loc = self.error * pred_obj * deriv_loc  # (B, sensory_dim)

        # Dual-state update: gradient descent on F.
        # Because ε = p - s, grad_obj is ∇F_obj. All terms subtract.
        # Precision-weighted errors and Gaussian prior state decay (Phase 4.6).
        if not freeze_obj:
            # Bottom-up sensory gradient — object pathway.
            grad_obj = err_scaled_obj @ self.W_obj  # (B, obj_dim)
            dx_obj = -(self.pi_sensory * grad_obj)

            # Lateral consensus pressure (Phase 2).
            if neighbor_context is not None:
                lateral_pred = neighbor_context @ self.W_lat.t()  # (B, obj_dim)
                self.err_lat = self.x_obj - lateral_pred
                dx_obj = dx_obj - self.pi_lat * self.err_lat
            else:
                self.err_lat = torch.zeros_like(self.x_obj)

            # Top-down hierarchical pressure (Phase 3).
            if top_down_prior is not None:
                self.err_td = self.x_obj - top_down_prior  # (B, obj_dim)
                dx_obj = dx_obj - self.pi_td * self.err_td
            else:
                self.err_td = torch.zeros_like(self.x_obj)

            # Temporal prediction pressure (Phase 4, non-linear).
            if self.x_obj_prev is not None:
                temporal_z = self.x_obj_prev @ self.W_trans.t()  # (B, obj_dim)
                temporal_pred = torch.tanh(temporal_z)
                self._temporal_z = temporal_z
                self._temporal_pred = temporal_pred
                self.err_time = self.x_obj - temporal_pred
                dx_obj = dx_obj - self.pi_time * self.err_time
            else:
                self._temporal_pred = None
                self.err_time = torch.zeros_like(self.x_obj)

            # State decay — Gaussian prior N(0, 1/alpha) on x_obj.
            # Prevents runaway activations and provides biological gain control.
            dx_obj = dx_obj - self.alpha * self.x_obj

            self.x_obj = self.x_obj + eta_x * dx_obj

        # Location pathway — with state decay.
        grad_loc = err_scaled_loc @ self.W_loc     # (B, loc_dim)
        dx_loc = -(self.pi_sensory * grad_loc) - self.alpha * self.x_loc
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
        by moving the sensor.  With Convention B (ε = p - s), the
        derivative of F w.r.t. sensory input is -ε, so gradient descent
        on F w.r.t. action yields a positive sign:

        .. math::
            \\frac{\\partial F}{\\partial a}
              = \\frac{\\partial F}{\\partial s} \\cdot \\frac{\\partial s}{\\partial a}
              = -\\epsilon \\cdot \\frac{\\partial s}{\\partial a}

            a = -\\frac{\\partial F}{\\partial a}
              = \\eta_a \\cdot \\pi_s \\cdot \\epsilon \\cdot \\frac{\\partial s}{\\partial a}

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
        # Convention B (ε = p - s):
        #   dF/ds = -ε  (F = ½||p-s||², dF/ds = -(p-s) = -ε)
        #   dF/da = dF/ds · ds/da = -ε · ds/da
        #   action = -dF/da = +π_s · ε · ds/da  (precision-weighted gradient descent on F)
        #
        # error: (B, sensory_dim)
        # sensory_gradient: (B, sensory_dim, action_dim)
        # result: (B, action_dim)
        action = eta_a * self.pi_sensory * torch.einsum("bs,bsa->ba", self.error, sensory_gradient)
        return action

    # ------------------------------------------------------------------
    # Learning (slow dynamics — Hebbian)
    # ------------------------------------------------------------------

    def learn(
        self,
        eta_w: float = 0.001,
        neighbor_context: Optional[Tensor] = None,
    ) -> None:
        """Gradient descent on weights (Convention B: ε = p - s).

        Generative weights (W_obj, W_loc): With ε = p - s, the gradient
        ∇_W F is in the same direction as the raw Hebbian product, so
        gradient descent uses W -= η · ∇F  (subtraction).

        Prior weights (W_lat, W_trans): The prior error definitions
        (err_lat = x - Wx_nbr) already follow belief - expectation,
        and ∇_W F_lat = -err_lat · x_nbr^T.  Gradient descent
        W -= η · ∇F = W += η · dW  (addition, unchanged).

        .. math::
            \\text{err\\_scaled\\_obj} = \\epsilon \\cdot p_{loc} \\cdot f'(z_{obj})
            \\text{err\\_scaled\\_loc} = \\epsilon \\cdot p_{obj} \\cdot f'(z_{loc})
            W_{obj} -= \\eta_w \\cdot (\\pi_s \\cdot \\text{err\\_scaled\\_obj})^T \\cdot x_{obj}
            W_{loc} -= \\eta_w \\cdot (\\pi_s \\cdot \\text{err\\_scaled\\_loc})^T \\cdot x_{loc}
            W_{lat} += \\eta_w \\cdot (\\pi_l \\cdot \\text{err\\_lat})^T \\cdot \\bar{x}_{obj}^{\\text{nbr}}
            W_{trans} += \\eta_w \\cdot (\\pi_t \\cdot \\text{err\\_time\\_scaled})^T \\cdot x_{obj\\_prev}

        Averaged over the batch.

        Args:
            eta_w: Learning rate for weight updates.
            neighbor_context: (batch, obj_dim) — average x_obj of spatial
                neighbours.  If provided, W_lat is updated.
        """
        if self.error is None or self.x_obj is None:
            raise RuntimeError(
                "States not initialised. Run inference before learn()."
            )
        if self._prediction is None:
            raise RuntimeError(
                "Intermediates not cached. Run infer_step() before learn()."
            )

        batch_size = self.error.shape[0]

        # Product-rule chain rule with two independent derivatives:
        #   prediction = f(z_obj) * f(z_loc)
        #   err_scaled_obj = error * f(z_loc) * f'(z_obj) = error * pred_loc * deriv_obj
        #   err_scaled_loc = error * f(z_obj) * f'(z_loc) = error * pred_obj * deriv_loc
        deriv_obj = self.activation_deriv(self._z_obj)
        deriv_loc = self.activation_deriv(self._z_loc)
        err_scaled_obj = self.error * self._pred_loc * deriv_obj
        err_scaled_loc = self.error * self._pred_obj * deriv_loc

        # dW_obj: ∇_W F for generative object weights.
        # Precision-scaled: π_s weights the sensory error contribution.
        # (sensory_dim, B) @ (B, obj_dim) -> (sensory_dim, obj_dim)
        dW_obj = ((self.pi_sensory * err_scaled_obj).t() @ self.x_obj) / batch_size
        # dW_loc: ∇_W F for generative location weights.
        dW_loc = ((self.pi_sensory * err_scaled_loc).t() @ self.x_loc) / batch_size

        # Gradient descent: W -= η · ∇F  (Convention B).
        self.W_obj.data -= eta_w * dW_obj
        self.W_loc.data -= eta_w * dW_loc

        # Lateral weight update (Phase 2).
        # F_lat = ½||x - Wx_nbr||², ∇_W F = -err_lat · x_nbr^T.
        # Gradient descent: W -= η · (-err_lat · x_nbr^T) = W += η · dW.
        if neighbor_context is not None and self.err_lat is not None:
            dW_lat = ((self.pi_lat * self.err_lat).t() @ neighbor_context) / batch_size
            self.W_lat.data += eta_w * dW_lat

        # Temporal weight update (Phase 4.5, non-linear).
        # Same logic as lateral: prior error, additive update.
        if (self.x_obj_prev is not None
                and self.err_time is not None
                and self._temporal_pred is not None):
            # Chain rule through activation: scale err_time by activation derivative
            # applied to the pre-activation temporal value (generic registry).
            err_time_scaled = self.err_time * self.activation_deriv(self._temporal_z)
            dW_trans = ((self.pi_time * err_time_scaled).t() @ self.x_obj_prev) / batch_size
            self.W_trans.data += eta_w * dW_trans

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_energy(self) -> float:
        """Return the precision-weighted variational free energy.

        Phase 4.6: Each error term is weighted by its precision scalar,
        and a Gaussian prior energy term penalises large latent states.

        .. math::
            F = \\frac{1}{2} \\pi_s \\sum \\epsilon_{\\text{sensory}}^2
              + \\frac{1}{2} \\pi_l \\sum \\epsilon_{\\text{lateral}}^2
              + \\frac{1}{2} \\pi_{td} \\sum \\epsilon_{\\text{top-down}}^2
              + \\frac{1}{2} \\pi_t \\sum \\epsilon_{\\text{temporal}}^2
              + \\frac{1}{2} \\alpha (\\|x_{obj}\\|^2 + \\|x_{loc}\\|^2)
        """
        e = 0.0
        if self.error is not None:
            e += 0.5 * self.pi_sensory * (self.error * self.error).sum().item()
        if self.err_lat is not None:
            e += 0.5 * self.pi_lat * (self.err_lat * self.err_lat).sum().item()
        if self.err_td is not None:
            e += 0.5 * self.pi_td * (self.err_td * self.err_td).sum().item()
        if self.err_time is not None:
            e += 0.5 * self.pi_time * (self.err_time * self.err_time).sum().item()
        # Gaussian prior energy: penalise deviation from zero.
        if self.x_obj is not None:
            e += 0.5 * self.alpha * (self.x_obj * self.x_obj).sum().item()
        if self.x_loc is not None:
            e += 0.5 * self.alpha * (self.x_loc * self.x_loc).sum().item()
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

    # Overcomplete: obj_dim (100) >> sensory_dim (25) — sparse dictionary.
    col = CorticalColumn(
        obj_dim=100, loc_dim=4, sensory_dim=25, activation_fn_name="tanh",
    ).to(device)
    print(col)

    B = 1
    col.reset_states(B, device)
    print(f"\n  x_obj shape: {col.x_obj.shape}  (expect [1, 100])")
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

    # Dendritic gating test (Phase 4.5).
    print("\n  --- Phase 4.5: Dendritic gating ---")
    col.reset_states(B, device)
    pred, pred_obj, pred_loc = col.predict_down()
    print(f"  predict_down returns tuple of 3  [OK]")
    print(f"  prediction shape: {pred.shape}  (expect [1, 25])")
    print(f"  pred_obj shape:   {pred_obj.shape}  (expect [1, 25])")
    print(f"  pred_loc shape:   {pred_loc.shape}  (expect [1, 25])")
    assert pred.shape == (B, 25)
    assert pred_obj.shape == (B, 25)
    assert pred_loc.shape == (B, 25)

    # No autograd.
    for name, param in col.named_parameters():
        assert not param.requires_grad, f"{name} has requires_grad=True!"
    print("  All parameters requires_grad=False  [OK]")

    print("\nCorticalColumn smoke test passed.")
