#!/usr/bin/env python3
"""Semi-synthetic experiment on the ASSISTments subset (A3).

For every family (t; {u, w}) with >= 30 students in each order we keep the real students, their real
orders and their real classes, and replace the outcome by

    Y* = Y_perm + kappa * 1{w -> u},

where Y_perm permutes the real outcomes within classes. Y_perm keeps every class mean, so the
between-class association of order and outcome in the real log is preserved, while the within-class
order effect is exactly kappa. Two estimators are compared:

  pooled        difference of the two order means over all students (Welch standard error);
  within-class  the class-fixed-effects contrast of run_assistments.py (defined when at least one
                class contains both orders; "testable" as in A2: >= 3 such classes and >= 20
                students per order in them).

Outputs (out_assistments/): a3_semisynthetic_summary.csv, a3_semisynthetic_family.csv,
fig_semisynthetic.pdf, table_semisynthetic.tex.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import run_assistments as ra

OUT = ra.OUT
SEED = 29
KAPPAS = [0.0, 0.05, 0.1]
REPS = 100
Z = stats.norm.ppf(0.975)


def within(A, Y, cl, ncl):
    n1 = np.bincount(cl, weights=A, minlength=ncl)
    n0 = np.bincount(cl, weights=1 - A, minlength=ncl)
    s1 = np.bincount(cl, weights=A * Y, minlength=ncl)
    s0 = np.bincount(cl, weights=(1 - A) * Y, minlength=ncl)
    q1 = np.bincount(cl, weights=A * Y * Y, minlength=ncl)
    q0 = np.bincount(cl, weights=(1 - A) * Y * Y, minlength=ncl)
    m = (n1 > 0) & (n0 > 0)
    if not m.any():
        return np.nan, np.nan
    n1, n0, s1, s0, q1, q0 = n1[m], n0[m], s1[m], s0[m], q1[m], q0[m]
    mu1, mu0 = s1 / n1, s0 / n0
    v1 = np.where(n1 > 1, (q1 - n1 * mu1 ** 2) / np.maximum(n1 - 1, 1), 0.1)
    v0 = np.where(n0 > 1, (q0 - n0 * mu0 ** 2) / np.maximum(n0 - 1, 1), 0.1)
    h = 1.0 / (1.0 / n1 + 1.0 / n0)
    est = (h * (mu1 - mu0)).sum() / h.sum()
    se = np.sqrt((h ** 2 * (v1 / n1 + v0 / n0)).sum()) / h.sum()
    return est, max(se, 1e-6)


def pooled(A, Y):
    y1, y0 = Y[A == 1], Y[A == 0]
    return y1.mean() - y0.mean(), np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))


def main():
    d = ra.load()
    first, y, klass, _ = ra.prepare(d)
    S = pd.read_csv(os.path.join(OUT, "a1_support_all_families.csv"))
    E = S[S.min_order >= ra.MIN_ORDER].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    fam, reps = [], []
    for i, r in E.iterrows():
        users, A, Y, cl = ra.family_data(first, y, klass, r.t, r.u, r.w)
        A = A.astype(float)
        codes, cl = np.unique(cl, return_inverse=True)
        ncl = len(codes)
        n1 = np.bincount(cl, weights=A, minlength=ncl)
        n0 = np.bincount(cl, weights=1 - A, minlength=ncl)
        mixed = (n1 > 0) & (n0 > 0)
        h = np.where(mixed, 1.0 / (1.0 / np.maximum(n1, 1) + 1.0 / np.maximum(n0, 1)), 0.0)
        testable = mixed.sum() >= ra.MIN_CLASSES and min(n1[mixed].sum(), n0[mixed].sum()) >= ra.MIN_WITHIN
        fam.append(dict(family=i, n=len(A), n_eff_within=float(h.sum()),
                        n_eff_pooled=float(1 / (1 / A.sum() + 1 / (1 - A).sum())),
                        classes_both=int(mixed.sum()), identifiable=bool(mixed.any()), testable=bool(testable)))
        base = np.lexsort((np.arange(len(A)), cl))
        for rep in range(REPS):
            perm = np.lexsort((rng.random(len(A)), cl))
            Yp = np.empty_like(Y)
            Yp[base] = Y[perm]
            for k in KAPPAS:
                Ys = Yp + k * A
                pe, ps = pooled(A, Ys)
                we, ws = within(A, Ys, cl, ncl) if mixed.any() else (np.nan, np.nan)
                reps.append((i, rep, k, pe, ps, we, ws))
        if i % 500 == 0:
            print(i, len(E), flush=True)
    F = pd.DataFrame(fam)
    R = pd.DataFrame(reps, columns=["family", "rep", "kappa", "pe", "ps", "we", "ws"]).merge(F, on="family")
    R["p_rej"] = np.abs(R.pe / R.ps) > Z
    R["w_rej"] = np.abs(R.we / R.ws) > Z
    R["p_cov"] = np.abs(R.pe - R.kappa) <= Z * R.ps
    R["w_cov"] = np.abs(R.we - R.kappa) <= Z * R.ws

    rows = []
    for k in KAPPAS:
        Rk = R[R.kappa == k]
        T = Rk[Rk.testable]
        I = Rk[Rk.identifiable]
        rows.append(dict(kappa=k,
                         pooled_rej_all=Rk.p_rej.mean(), pooled_cov_all=Rk.p_cov.mean(),
                         pooled_bias_abs_all=Rk.groupby("family").pe.mean().sub(k).abs().median(),
                         pooled_rej_testable=T.p_rej.mean(), pooled_cov_testable=T.p_cov.mean(),
                         within_rej_testable=T.w_rej.mean(), within_cov_testable=T.w_cov.mean(),
                         within_bias_testable=float(T.we.mean() - k),
                         within_rmse_testable=float(np.sqrt(((T.we - k) ** 2).mean())),
                         pooled_rmse_testable=float(np.sqrt(((T.pe - k) ** 2).mean())),
                         within_bias_identifiable=float(I.we.mean() - k)))
    Sm = pd.DataFrame(rows)
    # rates: per-family RMSE against effective support, kappa = 0.1
    Rk = R[R.kappa == 0.1]
    G = Rk.groupby("family").agg(w_rmse=("we", lambda x: float(np.sqrt(((x - 0.1) ** 2).mean()))),
                                 p_rmse=("pe", lambda x: float(np.sqrt(((x - 0.1) ** 2).mean()))),
                                 w_pow=("w_rej", "mean"), p_pow=("p_rej", "mean")).reset_index().merge(F, on="family")
    Gi = G[G.identifiable & (G.n_eff_within > 0) & (G.w_rmse > 1e-8)]
    slope_w = np.polyfit(np.log(Gi.n_eff_within), np.log(Gi.w_rmse), 1)[0]
    slope_p = np.polyfit(np.log(G.n_eff_pooled), np.log(G.p_rmse), 1)[0]
    # power by effective within-class support
    bins = [0, 5, 10, 20, np.inf]
    Gi = Gi.assign(bin=pd.cut(Gi.n_eff_within, bins, right=False))
    pw = Gi.groupby("bin", observed=True).agg(families=("family", "size"), power=("w_pow", "mean"),
                                              rmse=("w_rmse", "median")).reset_index()
    extra = dict(n_families=len(F), frac_identifiable=float(F.identifiable.mean()),
                 n_testable=int(F.testable.sum()), slope_within=float(slope_w), slope_pooled=float(slope_p),
                 median_n_eff_within=float(F.loc[F.identifiable, "n_eff_within"].median()),
                 median_n_eff_pooled=float(F.n_eff_pooled.median()),
                 median_pooled_rmse=float(G.p_rmse.median()), median_within_rmse=float(Gi.w_rmse.median()))
    Sm.to_csv(os.path.join(OUT, "a3_semisynthetic_summary.csv"), index=False)
    pd.DataFrame([extra]).to_csv(os.path.join(OUT, "a3_semisynthetic_rates.csv"), index=False)
    pw.to_csv(os.path.join(OUT, "a3_semisynthetic_power_by_support.csv"), index=False)
    G.to_csv(os.path.join(OUT, "a3_semisynthetic_family.csv"), index=False)
    pd.set_option("display.width", 250)
    print(Sm.round(3).to_string(index=False))
    print(extra)
    print(pw.round(3).to_string(index=False))
    figure(G, Gi, Sm, extra)
    table(Sm, pw, extra)


def figure(G, Gi, Sm, extra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.fontsize": 5.8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.45), constrained_layout=True)
    Gp = G[G.p_rmse > 1e-8]
    ax[0].scatter(Gp.n_eff_pooled, Gp.p_rmse, s=1, alpha=0.25, color="#999999", lw=0, label="pooled")
    ax[0].scatter(Gi.n_eff_within, Gi.w_rmse, s=1, alpha=0.25, color="#4C78A8", lw=0, label="within class")
    xs = np.array([1, 200.0])
    c = np.exp(np.median(np.log(Gi.w_rmse) + 0.5 * np.log(Gi.n_eff_within)))
    ax[0].plot(xs, c / np.sqrt(xs), "k:", lw=0.8, label=r"$\propto n_{\mathrm{eff}}^{-1/2}$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(0.01, 1.2)
    ax[0].set_xlabel(r"effective two-sided support $n_{\mathrm{eff}}$")
    ax[0].set_ylabel(r"RMSE of $\hat\kappa$")
    ax[0].set_title("(a) error vs support", fontsize=7)
    ax[0].legend(frameon=False, loc="upper right", markerscale=6)
    k = Sm.kappa.to_numpy()
    ax[1].plot(k, Sm.pooled_rej_all, "o-", color="#999999", ms=2.5, lw=0.9, label="pooled, all")
    ax[1].plot(k, Sm.within_rej_testable, "s-", color="#4C78A8", ms=2.5, lw=0.9, label="within, testable")
    ax[1].axhline(0.05, color="#E45756", lw=0.7, ls="--")
    ax[1].set_xlabel(r"true order effect $\kappa$"); ax[1].set_ylabel("rejection rate")
    ax[1].set_ylim(0, 1); ax[1].set_xticks(k)
    ax[1].set_title("(b) size and power", fontsize=7)
    ax[1].legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig_semisynthetic.pdf"))


def table(Sm, pw, extra):
    f = lambda x: f"{x:.2f}"
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"True \(\kappa\) & 0 & 0.05 & 0.10 \\", r"\midrule",
         r"Pooled, all families: rejection rate & " + " & ".join(f(x) for x in Sm.pooled_rej_all) + r" \\",
         r"Pooled, all families: 95\% CI coverage & " + " & ".join(f(x) for x in Sm.pooled_cov_all) + r" \\",
         r"Within class, testable: rejection rate & " + " & ".join(f(x) for x in Sm.within_rej_testable) + r" \\",
         r"Within class, testable: 95\% CI coverage & " + " & ".join(f(x) for x in Sm.within_cov_testable) + r" \\",
         r"Within class, testable: mean error & " + " & ".join(f"{abs(x):.3f}" if abs(x) < 5e-4 else f"{x:+.3f}" for x in Sm.within_bias_testable) + r" \\",
         r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_semisynthetic.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
