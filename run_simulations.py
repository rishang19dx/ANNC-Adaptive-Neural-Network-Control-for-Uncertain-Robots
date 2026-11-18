"""Reproduce Section IV of He, Chen & Yin (IEEE T-Cyb 2016).

    python run_simulations.py                     # all three cases, 30 s
    python run_simulations.py --cases NN --tf 10
    python run_simulations.py --q0 0.1 2.1        # non-zero initial error
    python run_simulations.py --no-disturbance
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from annc.controllers import AdaptiveNNController, ModelBasedController, PDController
from annc.params import ConstraintParams, SimParams
from annc.plotting import comparison_figure, controller_figures
from annc.robot import TwoLinkArm
from annc.simulate import Simulation


def metrics(L, cons):
    after = L["t"] >= 5.0          # steady-state window, as in the paper's zoomed figures
    return {
        "max_abs_z1": np.abs(L["z1"]).max(0).tolist(),
        "max_abs_z2": np.abs(L["z2"]).max(0).tolist(),
        "rms_z1_after_5s": np.sqrt((L["z1"][after] ** 2).mean(0)).tolist(),
        "max_abs_q": np.abs(L["q"]).max(0).tolist(),
        "max_abs_dq": np.abs(L["dq"]).max(0).tolist(),
        "max_abs_tau": np.abs(L["tau"]).max(0).tolist(),
        "z1_within_ka": bool(np.all(np.abs(L["z1"]) < cons.ka)),
        "z2_within_kb": bool(np.all(np.abs(L["z2"]) < cons.kb)),
        "q_within_kc1": bool(np.all(np.abs(L["q"]) <= cons.kc1)),
        "dq_within_kc2": bool(np.all(np.abs(L["dq"]) <= cons.kc2)),
        "t_end": float(L["t"][-1]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=["MB", "NN", "PD"], choices=["MB", "NN", "PD"])
    ap.add_argument("--tf", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--q0", type=float, nargs=2, default=[0.0, 2.0])
    ap.add_argument("--dq0", type=float, nargs=2, default=[0.5, 0.0])
    ap.add_argument("--no-disturbance", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    robot, cons = TwoLinkArm(), ConstraintParams()
    sim = SimParams(t_final=a.tf, dt=a.dt, seed=a.seed, q0=np.array(a.q0), dq0=np.array(a.dq0),
                    disturbance=not a.no_disturbance, log_every=max(1, int(round(1e-2 / a.dt))))
    ctrls = {"MB": lambda: ModelBasedController(robot, cons),
             "NN": lambda: AdaptiveNNController(cons),
             "PD": lambda: PDController(cons)}
    print(f"k_a = {cons.ka}, k_b = {cons.kb}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logs, summary = {}, {}
    for name in a.cases:
        t0 = time.time()
        L = Simulation(robot, ctrls[name](), sim).run(progress=True)
        logs[name] = L
        summary[name] = metrics(L, cons)
        np.savez_compressed(out / f"{name}_log.npz", **L)
        controller_figures(name, L, cons, out / name)
        print(f"{name} finished in {time.time() - t0:.1f} s")
        for k, v in summary[name].items():
            print(f"    {k:18s} {np.round(v, 4).tolist() if isinstance(v, list) else v}")
    if len(logs) > 1:
        comparison_figure(logs, cons, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Figures and logs written to {out.resolve()}")


if __name__ == "__main__":
    main()
