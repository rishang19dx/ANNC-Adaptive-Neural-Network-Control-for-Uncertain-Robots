"""PyBullet demos of the BLF model-based / adaptive-NN controllers.

    python demos/run_demo.py --robot planar --controller NN --gui
    python demos/run_demo.py --robot iiwa --controller NN --payload 2 --record demos/media/iiwa_nn.mp4
    python demos/run_demo.py --robot iiwa --controller all --payload 2      # MB vs NN vs PD + plots

Robots
  planar : the paper's 2-link arm (Table I) built as a URDF - same plant as Section IV.
  iiwa   : KUKA LBR iiwa 14 (pybullet_data). Joints 2 and 4 (shoulder/elbow pitch) are
           controlled; the other five are locked at 0 by stiff position servos.

The external force f(t) of the paper is applied at the end effector. `--payload KG`
adds mass to the last link at `--payload-time`; the MB controller's model (a separate
PyBullet client) is never told, the NN controller has no model at all.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos"))

import pybullet_data  # noqa: E402

from annc.controllers import AdaptiveNNController, ModelBasedController, PDController  # noqa: E402
from annc.params import ConstraintParams, MBGains, NNGains, PDGains  # noqa: E402
from annc.plotting import comparison_figure, controller_figures  # noqa: E402
from annc.signals import ExternalForce  # noqa: E402
from annc.simulate import solve_loop  # noqa: E402
from bullet_arm import BulletArm, BulletModel  # noqa: E402


class Sinusoid:
    """q_d = c + A sin(w t + phi). The paper's reference is c = 0, A = [1, 2], phi = [0, pi/2]."""

    def __init__(self, c, A, phi, w=0.5):
        self.c, self.A, self.phi, self.w = map(np.asarray, (c, A, phi, w))

    def __call__(self, t):
        s, co = np.sin(self.w * t + self.phi), np.cos(self.w * t + self.phi)
        return self.c + self.A * s, self.A * self.w * co, -self.A * self.w**2 * s

    def constraints(self, ka=(0.17, 0.17), kb=(0.18, 0.18)):
        """Keep the paper's error bounds k_a, k_b; derive k_c1, k_c2 from the reference bounds."""
        X0 = np.abs(self.c) + np.abs(self.A)
        Y0 = np.abs(self.A) * self.w
        return ConstraintParams(kc1=X0 + np.array(ka), kc2=Y0 + np.array(kb), X0=X0, Y0=Y0)


ROBOTS = {
    "planar": dict(
        urdf=str(ROOT / "demos" / "planar_2link.urdf"), active=[0, 1], locked={}, ee_link=1,
        ee_offset=(0.31, 0, 0),
        ref=Sinusoid([0, 0], [1, 2], [0, np.pi / 2]),
        camera=dict(target=[0.1, 0, 0.05], distance=1.35, yaw=0, pitch=-2), floor=False,
        gains=dict(MB=MBGains(), NN=NNGains(), PD=PDGains())),
    "iiwa": dict(
        urdf="kuka_iiwa/model.urdf", active=[1, 3], locked={}, ee_link=6, ee_offset=(0, 0, 0.045),
        ref=Sinusoid([0.5, -1.0], [0.4, 0.8], [0, np.pi / 2]),
        camera=dict(target=[0.3, 0, 0.6], distance=2.0, yaw=0, pitch=-5), floor=True,
        # heavier robot -> stronger gains than Table I's tiny arm (see README)
        gains=dict(MB=MBGains(k1=np.array([40.0, 40.0]), K2=np.array([60.0, 40.0])),
                   NN=NNGains(k1=np.array([40.0, 40.0]), k2=np.array([5.0, 5.0]),
                              K3=np.array([150.0, 80.0]), gamma=np.array([200.0, 200.0])),
                   PD=PDGains(Kp=np.array([400.0, 200.0]), Kd=np.array([40.0, 20.0])))),
}


def run(robot_key, ctrl_name, tf, dt, gui, record, payload, payload_time, disturbance, q0_offset, seed,
        realtime):
    R = ROBOTS[robot_key]
    ref, cons = R["ref"], R["ref"].constraints()
    plant = BulletArm(R["urdf"], R["active"], R["locked"], R["ee_link"], R["ee_offset"], dt=dt, gui=gui,
                      camera=R["camera"], floor=R["floor"])
    if ctrl_name == "MB":
        ctrl = ModelBasedController(BulletModel(R["urdf"], R["active"], R["locked"], R["ee_link"], R["ee_offset"]),
                                    cons, R["gains"]["MB"])
    elif ctrl_name == "NN":
        ctrl = AdaptiveNNController(cons, R["gains"]["NN"])
    else:
        ctrl = PDController(cons, R["gains"]["PD"], k1_for_logging=R["gains"]["MB"].k1)
    force = ExternalForce(tf, seed=seed, enabled=disturbance)

    qd0, dqd0, _ = ref(0.0)
    plant.reset(qd0 + np.asarray(q0_offset), dqd0)
    W = ctrl.initial_weights()
    frames, fps, next_frame = [], 30, 0.0
    keys = ["t", "q", "dq", "qd", "dqd", "alpha", "z1", "z2", "tau", "f", "Wnorm", "violated"]
    L = {k: [] for k in keys}
    steps = int(round(tf / dt))
    log_every = max(1, int(round(1e-2 / dt)))
    payload_done = payload <= 0
    wall0 = time.time()
    for k in range(steps + 1):
        t = k * dt
        if not payload_done and t >= payload_time:
            plant.add_payload(payload)
            payload_done = True
        q, dq = plant.state()
        r = ref(t)
        f = force(t)
        law = ctrl(t, q, dq, r, f, W)
        M, h, _ = plant.plant_model(f)          # true plant (with payload) for the algebraic loop
        tau, _ = solve_loop(law, M, h)
        if k % log_every == 0:
            lw = law.log
            for key, v in (("t", t), ("q", q), ("dq", dq), ("qd", r[0]), ("dqd", r[1]), ("alpha", lw["alpha"]),
                           ("z1", lw["z1"]), ("z2", lw["z2"]), ("tau", tau), ("f", f),
                           ("Wnorm", np.linalg.norm(W, axis=1) if W is not None else np.zeros(2)),
                           ("violated", lw["violated"])):
                L[key].append(v)
        if k == steps:
            break
        if not np.all(np.isfinite(tau)):
            print(f"  [{ctrl_name}] non-finite torque at t = {t:.3f} s, stopping")
            break
        plant.set_ghost(r[0])
        plant.step(tau, f)
        if W is not None:
            W = W + dt * law.dW
        if record and t >= next_frame:
            cap = f"{robot_key}  |  {ctrl_name} control  |  t = {t:5.1f} s"
            if payload > 0:
                cap += f"  |  payload +{payload:g} kg" if payload_done else f"  |  payload at {payload_time:g} s"
            sub = f"|z1| = {np.abs(law.log['z1']).round(3).tolist()}  (barrier k_a = {cons.ka[0]:.2f})"
            frames.append(caption(plant.frame(), cap, sub, bool(law.log["violated"])))
            next_frame += 1.0 / fps
        if gui and realtime:
            lag = t - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)
    plant.close()
    return {k: np.array(v) for k, v in L.items()}, cons, frames, fps


def caption(img, line1, line2, alarm=False):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 44], fill=(245, 245, 245))
    d.text((10, 6), line1, fill=(20, 20, 20))
    d.text((10, 24), line2 + ("   BARRIER VIOLATED" if alarm else ""), fill=(200, 30, 30) if alarm else (60, 60, 60))
    d.text((10, im.height - 18), "green = desired trajectory q_d (ghost)", fill=(30, 140, 50))
    return np.asarray(im)


def save_video(frames, fps, path):
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gif":
        imageio.mimsave(path, frames[::2], duration=2.0 / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    print(f"  video -> {path}")


def summarize(L, cons):
    after = L["t"] >= 5.0
    return dict(max_abs_z1=np.abs(L["z1"]).max(0).round(4).tolist(),
                max_abs_z2=np.abs(L["z2"]).max(0).round(4).tolist(),
                rms_z1_after_5s=np.sqrt((L["z1"][after] ** 2).mean(0)).round(4).tolist(),
                max_abs_tau=np.abs(L["tau"]).max(0).round(1).tolist(),
                barrier_respected=bool(not L["violated"].any()),
                t_end=float(L["t"][-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=list(ROBOTS), default="planar")
    ap.add_argument("--controller", choices=["MB", "NN", "PD", "all"], default="NN")
    ap.add_argument("--tf", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--gui", action="store_true", help="open the PyBullet window")
    ap.add_argument("--realtime", action="store_true", help="with --gui, slow down to wall-clock time")
    ap.add_argument("--record", help="save an .mp4 or .gif (one controller only)")
    ap.add_argument("--payload", type=float, default=0.0, help="kg added to the last link")
    ap.add_argument("--payload-time", type=float, default=10.0)
    ap.add_argument("--q0-offset", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--no-disturbance", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="plot directory (default demos/results/<robot>)")
    a = ap.parse_args()

    names = ["MB", "NN", "PD"] if a.controller == "all" else [a.controller]
    out = Path(a.out or ROOT / "demos" / "results" / a.robot)
    logs, summary = {}, {}
    for name in names:
        t0 = time.time()
        L, cons, frames, fps = run(a.robot, name, a.tf, a.dt, a.gui, bool(a.record), a.payload, a.payload_time,
                                   not a.no_disturbance, a.q0_offset, a.seed, a.realtime)
        logs[name], summary[name] = L, summarize(L, cons)
        controller_figures(name, L, cons, out / name)
        print(f"{a.robot}/{name}: {time.time() - t0:.1f} s  {summary[name]}")
        if a.record:
            rec = Path(a.record)
            save_video(frames, fps, rec if len(names) == 1 else rec.with_name(f"{rec.stem}_{name}{rec.suffix}"))
    if len(logs) > 1:
        comparison_figure(logs, cons, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(args=vars(a), results=summary), indent=2))
    print(f"plots -> {out}")


if __name__ == "__main__":
    import pybullet as pb  # noqa: F401  (fail early if missing)
    _ = pybullet_data.getDataPath()
    main()
