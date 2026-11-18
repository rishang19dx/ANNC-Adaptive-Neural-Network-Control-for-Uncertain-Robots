"""Parameters from He, Chen & Yin, "Adaptive Neural Network Control of an
Uncertain Robot With Full-State Constraints", IEEE T-Cyb 46(3), 2016.

Values not stated in the paper are marked ASSUMPTION.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RobotParams:
    """Table I. Two-link arm moving in the vertical plane."""
    m1: float = 2.0
    m2: float = 0.85
    l1: float = 0.35
    l2: float = 0.31
    I1: float = 61.25e-3
    I2: float = 20.42e-3
    # ASSUMPTION: the paper does not give the centre-of-mass distances.
    # Mid-link values reproduce the paper's PD steady-state error (Fig. 15).
    lc1: float = 0.35 / 2
    lc2: float = 0.31 / 2
    g: float = 9.81


def _vec(*v):
    return field(default_factory=lambda: np.array(v, dtype=float))


@dataclass
class ConstraintParams:
    """Section IV: |q| <= kc1, |dq| <= kc2, |q_d| <= X0, |dq_d| <= Y0."""
    kc1: np.ndarray = _vec(1.17, 2.17)
    kc2: np.ndarray = _vec(0.68, 1.18)
    X0: np.ndarray = _vec(1.0, 2.0)
    Y0: np.ndarray = _vec(0.5, 1.0)

    @property
    def ka(self):  # bound on z1
        return self.kc1 - self.X0

    @property
    def kb(self):  # bound on z2
        return self.kc2 - self.Y0


@dataclass
class MBGains:
    """Section IV-A (model-based control, eq. 15)."""
    k1: np.ndarray = _vec(40.0, 5.0)
    K2: np.ndarray = _vec(28.0, 21.0)


@dataclass
class NNGains:
    """Section IV-B (adaptive NN control, eqs. 17 and 19)."""
    k1: np.ndarray = _vec(40.0, 1.0)
    k2: np.ndarray = _vec(5.0, 16.0)
    K3: np.ndarray = _vec(70.0, 19.0)
    sigma: np.ndarray = _vec(0.001, 0.001)
    gamma: np.ndarray = _vec(50.0, 100.0)   # Gamma_i = gamma_i * I
    # 256 = 2^8 nodes: centres on the vertices of [-1, 1]^8 (Z has 8 entries).
    rbf_levels: tuple = (-1.0, 1.0)
    # ASSUMPTION: the paper does not give the Gaussian width.
    rbf_width: float = 2.0


@dataclass
class PDGains:
    """Section IV-C."""
    Kp: np.ndarray = _vec(40.0, 4.0)
    Kd: np.ndarray = _vec(0.4, 1.0)


@dataclass
class SimParams:
    t_final: float = 30.0
    dt: float = 1e-4
    log_every: int = 10
    q0: np.ndarray = _vec(0.0, 2.0)
    dq0: np.ndarray = _vec(0.5, 0.0)
    disturbance: bool = True
    # ASSUMPTION: the paper gives no noise sample time; d(t) is held for this long.
    noise_hold: float = 0.01
    seed: int = 0
