"""Figures mirroring Figs. 2-18 of the paper, plus a cross-controller comparison."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"MB": "#1f77b4", "NN": "#d62728", "PD": "#2ca02c"}


def _two_rows(title, ylabel):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.set_ylabel(ylabel.replace("{i}", str(i + 1)))
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    return fig, axes


def controller_figures(name, L, cons, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    t = L["t"]
    c = COLORS[name]

    fig, ax = _two_rows(f"{name}: position tracking  x1 vs x_d", r"$x_{{1{i}}}$ [rad]")
    for i in range(2):
        ax[i].plot(t, L["q"][:, i], c, label=f"$x_{{1{i+1}}}$")
        ax[i].plot(t, L["qd"][:, i], "k--", lw=1, label=f"$x_{{d{i+1}}}$")
        ax[i].axhline(cons.kc1[i], color="gray", ls=":", lw=1)
        ax[i].axhline(-cons.kc1[i], color="gray", ls=":", lw=1)
        ax[i].legend(loc="upper right")
    fig.savefig(out / "x1_tracking.png", dpi=130); plt.close(fig)

    fig, ax = _two_rows(f"{name}: velocity  x2 vs alpha", r"$x_{{2{i}}}$ [rad/s]")
    for i in range(2):
        ax[i].plot(t, L["dq"][:, i], c, label=f"$x_{{2{i+1}}}$")
        ax[i].plot(t, L["alpha"][:, i], "k--", lw=1, label=rf"$\alpha_{i+1}$")
        ax[i].axhline(cons.kc2[i], color="gray", ls=":", lw=1)
        ax[i].axhline(-cons.kc2[i], color="gray", ls=":", lw=1)
        ax[i].legend(loc="upper right")
    fig.savefig(out / "x2_tracking.png", dpi=130); plt.close(fig)

    for key, bound, lab in (("z1", cons.ka, "k_a"), ("z2", cons.kb, "k_b")):
        fig, ax = _two_rows(f"{name}: tracking error {key} (dotted = ±{lab})", "$" + key[0] + "_{" + key[1] + "{i}}$")
        for i in range(2):
            ax[i].plot(t, L[key][:, i], c)
            for s in (1, -1):
                ax[i].axhline(s * bound[i], color="gray", ls=":", lw=1.2)
        fig.savefig(out / f"{key}.png", dpi=130); plt.close(fig)

    fig, ax = _two_rows(f"{name}: control input", r"$\tau_{i}$ [Nm]")
    for i in range(2):
        ax[i].plot(t, L["tau"][:, i], c)
    fig.savefig(out / "tau.png", dpi=130); plt.close(fig)

    if name == "NN":
        fig, ax = _two_rows("NN: weight norms", r"$\|\hat W_{i}\|$")
        for i in range(2):
            ax[i].plot(t, L["Wnorm"][:, i], c)
        fig.savefig(out / "weights.png", dpi=130); plt.close(fig)


def comparison_figure(logs, cons, outdir):
    out = Path(outdir)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for name, L in logs.items():
        for i in range(2):
            axes[0, i].plot(L["t"], L["z1"][:, i], color=COLORS[name], label=name, lw=1.2)
            axes[1, i].plot(L["t"], L["z2"][:, i], color=COLORS[name], label=name, lw=1.2)
    for i in range(2):
        for r, (b, lab) in enumerate(((cons.ka, "z_1"), (cons.kb, "z_2"))):
            ax = axes[r, i]
            for s in (1, -1):
                ax.axhline(s * b[i], color="gray", ls=":", lw=1.2)
            ax.set_ylim(-3 * b[i], 3 * b[i])
            ax.set_ylabel(f"${lab[0]}_{{{lab[2]}{i + 1}}}$")
            ax.grid(alpha=0.3)
        axes[1, i].set_xlabel("t [s]")
    axes[0, 0].legend()
    fig.suptitle("Error comparison (dotted: barrier bounds ±k_a, ±k_b; y-axis clipped to ±3k)")
    fig.tight_layout()
    fig.savefig(out / "comparison_errors.png", dpi=130); plt.close(fig)
