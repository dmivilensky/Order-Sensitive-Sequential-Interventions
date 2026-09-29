#!/usr/bin/env python3
"""Real-data experiments on the ASSISTments 2009-2010 skill-builder data (Feng et al., 2009).

Subset: the first 192,000 rows of the public skill-builder file (sorted by skill id), i.e. skills with
ids <= 70, original (uncorrected) release in which a
problem tagged with several skills appears once per skill; rows without a skill name are dropped (44 skills, 3200 students, 183 classes, 63 schools).
The start of a skill is the first logged problem of that skill (order_id is chronological).

Families. For a target skill t and an unordered pair {u, w} of other skills (u <_tau w by name), a
student with a logged start of t has started none, one, or both of u, w before t, and if both, in one of
two orders (ties, i.e. a multi-skill problem starting both, are excluded). Outcome: mean first-attempt correctness on the first three original problems of t.

A1  Support separation at scale: for every family whose endpoint {u, w} is reached by >= 30 students,
    the number of students in the rarer order.
A2  Who sets the order?  Skill builders are assigned by teachers, so order varies mostly between
    classes. For every family with >= 30 students in each order we count the classes in which both
    orders occur, and estimate the order effect within them (class fixed effects, inverse-variance-free
    harmonic weights). Families with >= 3 such classes and >= 20 students per order in them are tested;
    Benjamini-Hochberg at q = 0.1; the number of discoveries is compared with a null that permutes the
    order labels within classes and is run through the same pipeline (N_PERM times).
"""
import itertools
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "assistments_subset.csv")
OUT = os.path.join(HERE, "out_assistments")
SEED = 13
MIN_ENDPOINT = 30
MIN_ORDER = 30
K_TARGET = 3
N_PERM = 20
MIN_CLASSES = 3      # classes in which both orders occur
MIN_WITHIN = 20      # students per order inside those classes
Q_BH = 0.1


def load():
    d = pd.read_csv(DATA, encoding="latin1", low_memory=False)
    d = d.dropna(subset=["skill_name"]).sort_values("order_id", kind="stable")
    return d


def prepare(d):
    first = d.groupby(["user_id", "skill_name"]).order_id.min().unstack()
    orig = d[d.original == 1]
    # outcome per (user, skill): mean correctness on the first K original problems of the skill
    orig = orig.sort_values("order_id").drop_duplicates(["user_id", "skill_name", "problem_id"])
    orig["rank"] = orig.groupby(["user_id", "skill_name"]).cumcount()
    y = orig[orig["rank"] < K_TARGET].groupby(["user_id", "skill_name"]).correct.mean().unstack()
    klass = d.groupby("user_id").student_class_id.agg(lambda s: s.mode().iloc[0])
    school = d.groupby("user_id").school_id.agg(lambda s: s.mode().iloc[0])
    return first, y, klass, school


def bh(p, q):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    ok = p[order] <= thresh
    k = np.max(np.flatnonzero(ok)) + 1 if ok.any() else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej


def family_data(first, y, klass, t, u, w):
    ft = first[t]
    bu, bw = first[u] < ft, first[w] < ft
    both = ft.notna() & bu & bw & y[t].notna()
    users = both[both].index
    A = (first.loc[users, w] < first.loc[users, u]).astype(int).to_numpy()  # 1 = w -> u (reverse)
    Y = y.loc[users, t].to_numpy(float)
    return users, A, Y, klass.loc[users].to_numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    first, y, klass, school = prepare(d)
    info = dict(n_rows=len(d), n_students=int(d.user_id.nunique()), n_skills=int(d.skill_name.nunique()),
                n_classes=int(d.student_class_id.nunique()), n_schools=int(d.school_id.nunique()))
    pd.DataFrame([info]).to_csv(os.path.join(OUT, "dataset_summary.csv"), index=False)
    print(info)
    skills = sorted(first.columns)

    # A1 support at scale
    rows = []
    F = first
    for t in skills:
        ft = F[t]
        has = ft.notna() & y[t].notna()
        for u, w in itertools.combinations([s for s in skills if s != t], 2):
            both = has & (F[u] < ft) & (F[w] < ft)
            uw = int((both & (F[u] < F[w])).sum())
            wu = int((both & (F[w] < F[u])).sum())
            tie = int((both & (F[w] == F[u])).sum())
            if uw + wu >= MIN_ENDPOINT:
                rows.append(dict(t=t, u=u, w=w, n_uw=uw, n_wu=wu, n_tie=tie, endpoint=uw + wu,
                                 min_order=min(uw, wu)))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "a1_support_all_families.csv"), index=False)
    a1 = dict(n_families=len(S), frac_min_order_lt5=float((S.min_order < 5).mean()),
              frac_min_order_eq0=float((S.min_order == 0).mean()),
              frac_min_order_ge30=float((S.min_order >= MIN_ORDER).mean()),
              median_ratio=float(np.median(S.min_order / S.endpoint)),
              frac_with_ties=float((S.n_tie > 0).mean()), total_ties=int(S.n_tie.sum()),
              total_strict=int(S.endpoint.sum()))
    pd.DataFrame([a1]).to_csv(os.path.join(OUT, "a1_summary.csv"), index=False)
    print("A1", a1)

    # A2 who chooses the order? support within the assignment unit (class)
    E = S[S.min_order >= MIN_ORDER].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    res, null_z = [], []
    for i, r in E.iterrows():
        users, A, Y, cl = family_data(first, y, klass, r.t, r.u, r.w)
        naive = float(Y[A == 1].mean() - Y[A == 0].mean())
        est, se, ncl, n1, n0 = fe_estimate(A, Y, cl)
        row = dict(t=r.t, u=r.u, w=r.w, n_uw=int((A == 0).sum()), n_wu=int(A.sum()), naive=naive,
                   classes=int(pd.Series(cl).nunique()), classes_both=ncl, wc_n_uw=n0, wc_n_wu=n1,
                   frac_students_mixed_classes=(n0 + n1) / len(A), fe=est, fe_se=se)
        res.append(row)
        if ncl >= MIN_CLASSES and min(n0, n1) >= MIN_WITHIN:
            zs = []
            for _ in range(N_PERM):
                Ap = permute_within(A, cl, rng)
                e2, s2, *_ = fe_estimate(Ap, Y, cl)
                zs.append(e2 / s2)
            null_z.append(zs)
    R = pd.DataFrame(res)
    elig = (R.classes_both >= MIN_CLASSES) & (np.minimum(R.wc_n_uw, R.wc_n_wu) >= MIN_WITHIN)
    R["estimable_within"] = elig
    R["z"] = R.fe / R.fe_se
    R["p"] = np.where(elig, 2 * stats.norm.sf(np.abs(R.z)), np.nan)
    R["bh"] = False
    R.loc[elig, "bh"] = bh(R.loc[elig, "p"].to_numpy(), Q_BH)
    null_z = np.array(null_z)
    null_p = 2 * stats.norm.sf(np.abs(null_z))
    null_disc = [int(bh(null_p[:, k], Q_BH).sum()) for k in range(null_z.shape[1])] if len(null_z) else [0]
    R.to_csv(os.path.join(OUT, "a2_order_effects.csv"), index=False)
    np.save(os.path.join(OUT, "a2_null_z.npy"), null_z)
    a2 = dict(n_families=len(R), frac_no_mixed_class=float((R.classes_both == 0).mean()),
              median_frac_students_in_mixed_classes=float(R.frac_students_mixed_classes.median()),
              n_estimable_within=int(elig.sum()), n_bh=int(R.bh.sum()),
              null_bh_mean=float(np.mean(null_disc)), null_bh_max=int(np.max(null_disc)),
              frac_p_lt_05=float((R.loc[elig, "p"] < 0.05).mean()) if elig.any() else np.nan,
              null_frac_p_lt_05=float((null_p < 0.05).mean()) if len(null_z) else np.nan,
              sd_naive=float(R.naive.std()), sd_fe=float(R.loc[elig, "fe"].std()),
              median_abs_naive=float(R.naive.abs().median()),
              median_abs_fe=float(R.loc[elig, "fe"].abs().median()),
              corr_naive_fe=float(np.corrcoef(R.loc[elig, "naive"], R.loc[elig, "fe"])[0, 1]),
              sign_agree=float((np.sign(R.loc[elig, "naive"]) == np.sign(R.loc[elig, "fe"])).mean()),
              n_perm=N_PERM)
    pd.DataFrame([a2]).to_csv(os.path.join(OUT, "a2_summary.csv"), index=False)
    print("A2", a2)
    T = R[elig].reindex(R[elig].z.abs().sort_values(ascending=False).index).head(10)
    T.to_csv(os.path.join(OUT, "a3_top_within_class.csv"), index=False)
    pd.set_option("display.width", 250)
    print(T[["t", "u", "w", "n_uw", "n_wu", "naive", "classes_both", "wc_n_uw", "wc_n_wu", "fe", "fe_se", "p", "bh"]]
          .round(3).to_string(index=False))
    write_outputs(S, R, null_z, T, a1, a2)


def fe_estimate(A, Y, cl):
    """Within-class (class fixed effects) difference w->u minus u->w, weighted by harmonic class sizes."""
    df = pd.DataFrame({"A": A, "Y": Y, "c": cl})
    d_, v_, w_ = [], [], []
    n1 = n0 = 0
    for _, g in df.groupby("c"):
        a1, a0 = g.Y[g.A == 1], g.Y[g.A == 0]
        if len(a1) and len(a0):
            wt = 1.0 / (1.0 / len(a1) + 1.0 / len(a0))
            var = (a1.var(ddof=1) if len(a1) > 1 else 0.1) / len(a1) + (a0.var(ddof=1) if len(a0) > 1 else 0.1) / len(a0)
            d_.append(a1.mean() - a0.mean()); v_.append(var); w_.append(wt)
            n1 += len(a1); n0 += len(a0)
    if not d_:
        return math.nan, math.nan, 0, 0, 0
    d_, v_, w_ = map(np.array, (d_, v_, w_))
    est = float((w_ * d_).sum() / w_.sum())
    se = float(np.sqrt((w_ ** 2 * v_).sum()) / w_.sum())
    return est, max(se, 1e-6), len(d_), n1, n0


def permute_within(A, cl, rng):
    Ap = A.copy()
    for c in np.unique(cl):
        idx = np.flatnonzero(cl == c)
        Ap[idx] = rng.permutation(A[idx])
    return Ap


def short(s, n=18):
    return s if len(s) <= n else s[: n - 1] + "."


def write_outputs(S, R, null_z, T, a1, a2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.fontsize": 5.8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.6))
    ax[0].scatter(S.endpoint, S.min_order + 0.5, s=1.2, alpha=0.15, color="#4C78A8", lw=0, rasterized=True)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    lim = [30, S.endpoint.max() * 1.1]
    ax[0].plot(lim, [l / 2 for l in lim], "k:", lw=0.7, label="balanced orders")
    ax[0].axhline(MIN_ORDER, color="#E45756", lw=0.7, ls="--", label=f"{MIN_ORDER} per order")
    ax[0].set_xlabel("endpoint support"); ax[0].set_ylabel("rarer order (+0.5)")
    ax[0].set_title("(a) support", fontsize=7)
    ax[0].legend(frameon=False, loc="lower right")
    bins = np.linspace(-0.8, 0.8, 41)
    el = R[R.estimable_within]
    ax[1].hist(R.naive, bins=bins, density=True, color="#BBBBBB", label="naive, all")
    ax[1].hist(el.fe, bins=bins, density=True, histtype="step", color="#E45756", lw=1.0, label="within class")
    ax[1].set_xlabel(r"$\hat\kappa$ (accuracy)"); ax[1].set_title("(b) order effects", fontsize=7)
    ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_assistments.pdf"), dpi=300)
    plt.close(fig)

    f = lambda x: f"{x:.2f}"
    L = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Quantity & value \\", r"\midrule",
         f"families with endpoint support $\\ge{MIN_ENDPOINT}$ & {a1['n_families']} \\\\",
         f"\\quad rarer order $<5$ / $=0$ & {100*a1['frac_min_order_lt5']:.0f}\\% / {100*a1['frac_min_order_eq0']:.0f}\\% \\\\",
         f"\\quad $\\ge{MIN_ORDER}$ students in each order & {a2['n_families']} \\\\",
         f"\\qquad no class with both orders & {100*a2['frac_no_mixed_class']:.0f}\\% \\\\",
         f"\\qquad testable within classes & {a2['n_estimable_within']} \\\\",
         f"\\qquad BH discoveries (FDR {Q_BH}), observed / null mean & {a2['n_bh']} / {a2['null_bh_mean']:.2f} \\\\",
         f"median $|\\hat\\kappa|$: naive / within-class & {f(a2['median_abs_naive'])} / {f(a2['median_abs_fe'])} \\\\",
         r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_assistments.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
