"""Paper figures from the CSV outputs of e1-e4 (vector PDF, sized for AISTATS columns)."""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "out")
C = {"seq": "#4C78A8", "pess": "#54A24B", "oi": "#F58518", "greedy": "#9D755D", "pooled": "#B279A2",
     "kappa": "#E45756", "phi": "#4C78A8", "naive": "#888888"}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 5.8,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.1, "lines.markersize": 3})


def panel_e1(ax):
    d = pd.read_csv(os.path.join(OUT, "e1_support_separation.csv"))
    ax.loglog(d.q, d.rmse_kappa_ipw, "o-", color=C["kappa"], label=r"$\hat\kappa$ (IPW)")
    ax.loglog(d.q, d.rmse_phi_ipw, "s-", color=C["phi"], label=r"$\hat\Phi^\rho$ (IPW)")
    ax.loglog(d.q, np.abs(d.bias_kappa_naive), "^--", color=C["naive"], label=r"$|$bias$|$ naive $\hat\kappa$")
    ref = d.rmse_kappa_ipw.iloc[-1] * np.sqrt(d.q.iloc[-1] / d.q)
    ax.loglog(d.q, ref, ":", color="k", lw=0.8, label=r"$\propto q^{-1/2}$")
    ax.set_xlabel(r"probability $q$ of the reverse order")
    ax.set_ylabel("RMSE")
    ax.set_title("(a) support separation")
    ax.set_ylim(0.006, 2.5)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)


def panel_e2(ax):
    d = pd.read_csv(os.path.join(OUT, "e2_planning_summary.csv"))
    d = d[d.sweep == "eps"].sort_values("value")
    for key, name, col, mk in [("sequence", "sequence-sensitive", C["seq"], "o"),
                               ("pessimistic-1", "pessimistic", C["pess"], "s"),
                               ("order-insensitive", "order-insensitive", C["oi"], "^"),
                               ("greedy", "greedy", C["greedy"], "v")]:
        m, se = d[key + "_mean"], d[key + "_se"]
        ax.plot(d.value, m, mk + "-", color=col, label=name)
        ax.fill_between(d.value, m - 2 * se, m + 2 * se, color=col, alpha=0.15, lw=0)
    ax.set_xlabel(r"order-effect scale $\varepsilon$")
    ax.set_ylabel("regret")
    ax.set_title("(b) planning from logs")
    ax.set_ylim(0, 3.9)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)


def panel_e3(ax):
    d = pd.read_csv(os.path.join(OUT, "e3_power.csv"))
    s = pd.read_csv(os.path.join(OUT, "e3_null.csv")).iloc[0]
    ax.plot(d.lam, d.power, "o-", color=C["kappa"], label="rejection rate")
    ax.axhline(0.05, color="#888", ls="--", lw=0.8, label="level 0.05")
    ax.set_xlabel(r"non-additivity $\lambda/\sigma$")
    ax.set_ylabel("power of cube test")
    ax.set_ylim(0, 1.02)
    ax.set_title("(c) cube consistency")
    ax.text(0.97, 0.06, f"projection error ratio\n{s.mean_ratio:.3f} (theory {s.theory_ratio:.3f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8)
    ax.legend(frameon=False, loc="upper left")


def panel_e4(ax):
    d = pd.read_csv(os.path.join(OUT, "e4_scaling_summary.csv"))
    cols = {2: "#4C78A8", 3: "#54A24B", 4: "#F58518", 6: "#E45756"}
    for w, col in cols.items():
        s = d[d.w == w]
        ax.semilogy(s.H, s.states, "o-", color=col)
        ax.semilogy(s.H, s.plans, "--", color=col, lw=0.8)
        ax.text(s.H.iloc[-1] + 0.4, s.plans.iloc[-1], f"$w={w}$", color=col, fontsize=5.8, va="center")
    ax.plot([], [], "ko-", label="DP states")
    ax.plot([], [], "k--", lw=0.8, label=r"plans in $\Gamma_H$")
    ax.set_xlim(3, 23.5)
    ax.set_xlabel(r"horizon $H$")
    ax.set_ylabel("count")
    ax.set_title("(d) planning cost")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.1)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 1.72))
    panel_e1(axes[0]); panel_e2(axes[1]); panel_e3(axes[2]); panel_e4(axes[3])
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_synthetic.pdf"))
    plt.close(fig)

    # appendix: E2 sweeps over order bias and sample size
    d = pd.read_csv(os.path.join(OUT, "e2_planning_summary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.0))
    for ax, sw, xl in [(axes[0], "beta", r"order bias $\beta$ of the behaviour policy"),
                       (axes[1], "n", r"number of logged trajectories $n$")]:
        s = d[d.sweep == sw].sort_values("value")
        for key, name, col in [("sequence", "sequence-sensitive", C["seq"]),
                               ("pessimistic-1", "pessimistic ($c=1$)", C["pess"]),
                               ("pessimistic", "pessimistic (union bound)", "#2E6B2E"),
                               ("order-insensitive", "order-insensitive", C["oi"]),
                               ("endpoint-pooled", "endpoint-pooled", C["pooled"]),
                               ("greedy", "greedy", C["greedy"])]:
            ax.errorbar(s.value, s[key + "_mean"], yerr=2 * s[key + "_se"], fmt="o-", color=col, capsize=1.5,
                        label=name)
        if sw == "n":
            ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("regret")
    axes[1].legend(frameon=False, fontsize=5.5, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_e2_appendix.pdf"))
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
