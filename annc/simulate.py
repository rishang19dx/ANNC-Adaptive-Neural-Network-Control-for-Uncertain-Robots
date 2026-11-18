"""Fixed-step RK4 closed-loop simulation."""
import numpy as np

from .params import SimParams
from .signals import ExternalForce, reference


def solve_loop(law, M, h):
    """Solve tau = tau0 + B a together with M a = tau - h.

    (I - B M^-1) tau = tau0 - B M^-1 h
    """
    Minv = np.linalg.inv(M)
    BM = law.B @ Minv
    tau = np.linalg.solve(np.eye(len(h)) - BM, law.tau0 - BM @ h)
    a = Minv @ (tau - h)
    return tau, a


class Simulation:
    def __init__(self, robot, controller, sim: SimParams = SimParams()):
        self.robot, self.ctrl, self.sim = robot, controller, sim
        self.force = ExternalForce(sim.t_final, sim.noise_hold, sim.seed, sim.disturbance)
        W0 = controller.initial_weights()
        self.wshape = None if W0 is None else W0.shape
        self.y0 = np.concatenate([sim.q0, sim.dq0] + ([] if W0 is None else [W0.ravel()]))

    def _eval(self, t, y):
        n = self.robot.n
        q, dq = y[:n], y[n:2 * n]
        W = None if self.wshape is None else y[2 * n:].reshape(self.wshape)
        f = self.force(t)
        ref = reference(t)
        law = self.ctrl(t, q, dq, ref, f, W)
        tau, a = solve_loop(law, self.robot.M(q), self.robot.bias(q, dq, f))
        dy = [dq, a] + ([] if law.dW is None else [law.dW.ravel()])
        return np.concatenate(dy), dict(tau=tau, f=f, ref=ref, law=law, W=W)

    def run(self, progress=False):
        s = self.sim
        steps = int(round(s.t_final / s.dt))
        n = self.robot.n
        y, t = self.y0.copy(), 0.0
        keys = ["t", "q", "dq", "qd", "dqd", "alpha", "z1", "z2", "tau", "f", "Wnorm", "violated"]
        log = {k: [] for k in keys}
        for k in range(steps + 1):
            k1, info = self._eval(t, y)
            if k % s.log_every == 0 or k == steps:
                lw = info["law"].log
                log["t"].append(t)
                log["q"].append(y[:n].copy())
                log["dq"].append(y[n:2 * n].copy())
                log["qd"].append(info["ref"][0])
                log["dqd"].append(info["ref"][1])
                log["alpha"].append(lw["alpha"])
                log["z1"].append(lw["z1"])
                log["z2"].append(lw["z2"])
                log["tau"].append(info["tau"])
                log["f"].append(info["f"])
                log["Wnorm"].append(np.linalg.norm(info["W"], axis=1) if info["W"] is not None else np.zeros(n))
                log["violated"].append(lw["violated"])
                if progress and k % (s.log_every * 5000) == 0:
                    print(f"  [{self.ctrl.name}] t = {t:6.2f} s", flush=True)
            if k == steps:
                break
            h = s.dt
            k2, _ = self._eval(t + h / 2, y + h / 2 * k1)
            k3, _ = self._eval(t + h / 2, y + h / 2 * k2)
            k4, _ = self._eval(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t = (k + 1) * h
            if not np.all(np.isfinite(y)):
                print(f"  [{self.ctrl.name}] state diverged at t = {t:.4f} s")
                break
        return {k: np.array(v) for k, v in log.items()}
