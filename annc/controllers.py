"""Controllers of Section III plus the PD baseline of Section IV-C.

Control laws (15) and (19) contain the joint acceleration a = x2_dot, which
itself depends on tau. Each controller therefore returns an affine law

    tau = tau0 + B @ a

and the simulator solves this together with the plant (an algebraic loop, as
Simulink would). This assumes instantaneous acceleration measurement, which is
how the paper uses it ("we can obtain the accelerations logically").
"""
from dataclasses import dataclass, field

import numpy as np

from .params import ConstraintParams, MBGains, NNGains, PDGains
from .rbf import GaussianRBF

EPS_Z2 = 1e-9      # below this, (z2^T)^+ = 0 (eq. 14)
EPS_BLF = 1e-9     # floor on k^2 - z^2 when a constraint is breached


@dataclass
class ControlLaw:
    tau0: np.ndarray
    B: np.ndarray
    dW: np.ndarray = None
    log: dict = field(default_factory=dict)


def row_pinv(z2):
    """Moore-Penrose inverse of the row vector z2^T, eq. (14): z2 / ||z2||^2."""
    nrm2 = float(z2 @ z2)
    return np.zeros_like(z2) if nrm2 < EPS_Z2**2 else z2 / nrm2


def backstep_errors(q, dq, ref, k1, ka):
    """z1, virtual control alpha (8)-(9), its derivative, and z2."""
    qd, dqd, ddqd = ref
    z1 = q - qd
    A = (ka**2 - z1**2) * k1 * z1
    alpha = dqd - A
    dz1 = dq - dqd
    dalpha = ddqd - k1 * (ka**2 - 3.0 * z1**2) * dz1
    z2 = dq - alpha
    return z1, z2, alpha, dalpha


def _blf_den(k, z):
    den = k**2 - z**2
    return np.maximum(den, EPS_BLF), bool(np.any(den <= 0.0))


class ModelBasedController:
    """Eq. (15): requires exact M, C, G, J and f."""
    name = "MB"

    def __init__(self, robot, cons=ConstraintParams(), gains=MBGains()):
        self.robot, self.c, self.g = robot, cons, gains

    def initial_weights(self):
        return None

    def __call__(self, t, q, dq, ref, f, W=None):
        ka, kb = self.c.ka, self.c.kb
        z1, z2, alpha, dalpha = backstep_errors(q, dq, ref, self.g.k1, ka)
        da, va = _blf_den(ka, z1)
        db, vb = _blf_den(kb, z2)
        P = row_pinv(z2)
        w = z2 / db                                       # z2i / (kb_i^2 - z2i^2)
        r = self.robot
        tau0 = (r.J(q).T @ f + r.C(q, dq) @ dq + r.G(q) + r.M(q) @ dalpha
                - self.g.K2 * z2
                - P * np.sum(z1 * z2 / da)
                + P * (w @ dalpha))
        B = -np.outer(P, w)
        return ControlLaw(tau0, B, None,
                          dict(z1=z1, z2=z2, alpha=alpha, violated=va or vb))


class AdaptiveNNController:
    """Eqs. (17) and (19): model-free; an RBF network learns
    -(J^T f + C x2 + G + M alpha_dot) (eq. 18)."""
    name = "NN"

    def __init__(self, cons=ConstraintParams(), gains=NNGains(), n=2):
        self.c, self.g, self.n = cons, gains, n
        self.rbf = GaussianRBF.grid(4 * n, gains.rbf_levels, gains.rbf_width)

    def initial_weights(self):
        return np.zeros((self.n, self.rbf.size))

    def __call__(self, t, q, dq, ref, f, W):
        g, ka, kb = self.g, self.c.ka, self.c.kb
        z1, z2, alpha, dalpha = backstep_errors(q, dq, ref, g.k1, ka)
        da, va = _blf_den(ka, z1)
        db, vb = _blf_den(kb, z2)
        P = row_pinv(z2)
        w = z2 / db
        S = self.rbf(np.concatenate([q, dq, alpha, dalpha]))   # Z = [x1, x2, alpha, alpha_dot]
        nn = W @ S                                              # W_hat^T S(Z)
        bracket_no_a = (-(w @ dalpha)
                        + np.sum(g.k1 * z1**2 / da)
                        + np.sum(g.k2 * z2**2 / db)
                        + np.sum(z1 * z2 / da))
        tau0 = -P * bracket_no_a - nn - g.K3 * z2
        B = -np.outer(P, w)
        # Adaptive law (17): W_i_dot = Gamma_i [S z2i - sigma_i |z2i| W_i]
        dW = g.gamma[:, None] * (np.outer(z2, S) - (g.sigma * np.abs(z2))[:, None] * W)
        return ControlLaw(tau0, B, dW,
                          dict(z1=z1, z2=z2, alpha=alpha, nn=nn, violated=va or vb))


class PDController:
    """tau = -Kp z1 - Kd z1_dot, no barrier term."""
    name = "PD"

    def __init__(self, cons=ConstraintParams(), gains=PDGains(), k1_for_logging=MBGains().k1):
        self.c, self.g, self.k1 = cons, gains, k1_for_logging

    def initial_weights(self):
        return None

    def __call__(self, t, q, dq, ref, f, W=None):
        qd, dqd, _ = ref
        z1, dz1 = q - qd, dq - dqd
        tau0 = -self.g.Kp * z1 - self.g.Kd * dz1
        # z2 = dq - alpha is logged with the MB virtual control for comparison.
        _, z2, alpha, _ = backstep_errors(q, dq, ref, self.k1, self.c.ka)
        violated = bool(np.any(np.abs(z1) >= self.c.ka) or np.any(np.abs(z2) >= self.c.kb))
        return ControlLaw(tau0, np.zeros((2, 2)), None,
                          dict(z1=z1, z2=z2, alpha=alpha, violated=violated))
