"""Reference trajectory and external force (Section IV)."""
import numpy as np


def reference(t):
    """q_d = [sin(0.5 t), 2 cos(0.5 t)] with its first two derivatives."""
    s, c = np.sin(0.5 * t), np.cos(0.5 * t)
    qd = np.array([s, 2.0 * c])
    dqd = np.array([0.5 * c, -s])
    ddqd = np.array([-0.25 * s, -0.5 * c])
    return qd, dqd, ddqd


class ExternalForce:
    """f(t) = [sin t + 1 + d(t), 2 cos t + d(t)]^T.

    d(t) is zero-mean, unit-variance Gaussian noise held piecewise constant and
    clipped to the bounds the paper reports (-0.783 <= d <= 0.818).
    """
    D_MIN, D_MAX = -0.783, 0.818
    F_BAR = np.array([3.0, 3.4])

    def __init__(self, t_final, hold=0.01, seed=0, enabled=True):
        self.enabled = enabled
        self.hold = hold
        n = int(np.ceil(t_final / hold)) + 2
        rng = np.random.default_rng(seed)
        self.d = np.clip(rng.standard_normal(n), self.D_MIN, self.D_MAX)

    def __call__(self, t):
        if not self.enabled:
            return np.zeros(2)
        d = self.d[min(int(t / self.hold), len(self.d) - 1)]
        return np.array([np.sin(t) + 1.0 + d, 2.0 * np.cos(t) + d])
