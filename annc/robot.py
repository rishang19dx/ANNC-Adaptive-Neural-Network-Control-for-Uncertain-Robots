"""Two-link planar arm in the vertical plane, eqs. (31)-(37).

    M(q) qdd + C(q, qd) qd + G(q) = tau - J(q)^T f
"""
import numpy as np

from .params import RobotParams


class TwoLinkArm:
    n = 2

    def __init__(self, p: RobotParams = RobotParams()):
        self.p = p

    def M(self, q):
        p = self.p
        c2 = np.cos(q[1])
        m11 = p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2 * p.l1 * p.lc2 * c2) + p.I1 + p.I2
        m12 = p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2) + p.I2
        m22 = p.m2 * p.lc2**2 + p.I2
        return np.array([[m11, m12], [m12, m22]])

    def C(self, q, dq):
        p = self.p
        h = p.m2 * p.l1 * p.lc2 * np.sin(q[1])
        return np.array([[-h * dq[1], -h * (dq[0] + dq[1])],
                         [h * dq[0], 0.0]])

    def G(self, q):
        p = self.p
        # The paper prints (m1*lc2 + m2*l1) in G11; dP/dq1 gives m1*lc1 (typo in paper).
        g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * np.cos(q[0]) + p.m2 * p.lc2 * p.g * np.cos(q[0] + q[1])
        g2 = p.m2 * p.lc2 * p.g * np.cos(q[0] + q[1])
        return np.array([g1, g2])

    def J(self, q):
        p = self.p
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        return np.array([[-p.l1 * s1 - p.l2 * s12, -p.l2 * s12],
                         [p.l1 * c1 + p.l2 * c12, p.l2 * c12]])

    def fk(self, q):
        p = self.p
        return np.array([p.l1 * np.cos(q[0]) + p.l2 * np.cos(q[0] + q[1]),
                         p.l1 * np.sin(q[0]) + p.l2 * np.sin(q[0] + q[1])])

    # Energies (eqs. 32-33), used to verify M, C, G in the tests.
    def kinetic_energy(self, q, dq):
        p = self.p
        return (0.5 * p.m1 * p.lc1**2 * dq[0]**2 + 0.5 * p.I1 * dq[0]**2
                + 0.5 * p.m2 * p.l1**2 * dq[0]**2
                + p.m2 * p.l1 * p.lc2 * dq[0] * (dq[0] + dq[1]) * np.cos(q[1])
                + 0.5 * p.m2 * p.lc2**2 * (dq[0] + dq[1])**2
                + 0.5 * p.I2 * (dq[0] + dq[1])**2)

    def potential_energy(self, q):
        p = self.p
        return (p.m1 * p.g * p.lc1 * np.sin(q[0])
                + p.m2 * p.g * (p.l1 * np.sin(q[0]) + p.lc2 * np.sin(q[0] + q[1])))

    def bias(self, q, dq, f):
        """h = J^T f + C dq + G, so that M a = tau - h."""
        return self.J(q).T @ f + self.C(q, dq) @ dq + self.G(q)
