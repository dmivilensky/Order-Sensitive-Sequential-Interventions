#!/usr/bin/env python3
"""Real-data experiments on the Sepsis Cases event log (Mannhardt, 2016; 4TU.ResearchData,
doi:10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460).

S1  Main diamond before the first regular admission (target `Admission NC`):
    u = LacticAcid, w = IV Antibiotics (u <_tau w). Support of endpoints and of both orders;
    order effect kappa = Q(w->u) - Q(u->w) for three outcomes, naive and cross-fitted AIPW with
    baseline covariates recorded at ER registration; overlap diagnostics.
S2  Support separation across all pairs of the five pre-admission ER actions
    {CRP, Leucocytes, LacticAcid, IV Liquid, IV Antibiotics}: endpoint support, order support, ties.
    Order effects are estimated for the pairs with at least 20 cases in each order (declared in advance).
S3  Sensitivity of the main effect to treating near-simultaneous events as ties (window delta).
S4  Additivity check (Assumption C3): under C3 the order effect of (LacticAcid, IV Antibiotics)
    is the same whether or not IV Liquid follows; we compare the AIPW estimates in the two strata.
S5  Planning on the main diamond: static plans over {none, u, w, u->w, w->u} chosen on 70% of
    the cases and evaluated by doubly robust value on the other 30%, over 200 random splits.

Outcome ("reward"): minus the number of days from the target to the first recorded release,
so larger is better. Secondary outcomes: minus 1{Return ER after the target}, minus 1{ICU
admission after the target}.
"""
import itertools
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "Sepsis Cases - Event Log.xes")
OUT = os.path.join(HERE, "out_sepsis")
SEED = 11
B_BOOT = 500
N_SPLITS = 200
TARGET = "Admission NC"
U, W = "LacticAcid", "IV Antibiotics"
ER_ACTIONS = ["CRP", "Leucocytes", "LacticAcid", "IV Liquid", "IV Antibiotics"]  # tau order
# Primary adjustment set: baseline severity recorded at ER registration. A specification with all 22
# registration attributes, weak regularization and clipping at 0.01 is reported in S6.
COVARIATES = ["Age", "SIRSCriteria2OrMore", "SIRSCritTachypnea", "SIRSCritHeartRate",
              "SIRSCritTemperature", "SIRSCritLeucos", "Hypotensie", "Hypoxie", "Oligurie",
              "DisfuncOrg", "InfectionSuspected"]
# Extended set used in the sensitivity analysis: adds the diagnostic-order flags.
COVARIATES_EXT = COVARIATES + ["DiagnosticBlood", "DiagnosticArtAstrup", "DiagnosticIC",
                               "DiagnosticLacticAcid", "DiagnosticECG", "DiagnosticXthorax",
                               "DiagnosticUrinaryCulture", "DiagnosticSputum", "DiagnosticLiquor",
                               "DiagnosticOther", "DiagnosticUrinarySediment"]
PS_C, PS_CLIP = 0.1, 0.05   # primary propensity regularization and clipping
OUTCOMES = {"days": "minus days to release", "return_er": "minus return to ER", "icu": "minus ICU admission"}
MIN_ORDER_SUPPORT = 20


# ----------------------------------------------------------------------------- data
def _val(e):
    t = e.tag.split("}")[-1]
    v = e.get("value")
    if t == "int":
        return int(v)
    if t == "float":
        return float(v)
    if t == "boolean":
        return v == "true"
    return v


def load_events(path=DATA):
    root = ET.parse(path).getroot()
    rows = []
    for tr in root:
        if tr.tag.split("}")[-1] != "trace":
            continue
        case = {a.get("key"): _val(a) for a in tr if a.tag.split("}")[-1] != "event"}
        for ev in tr:
            if ev.tag.split("}")[-1] != "event":
                continue
            r = {"case": case.get("concept:name")}
            for a in ev:
                r[a.get("key")] = _val(a)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time:timestamp"], utc=True, format="mixed")
    return df.sort_values(["case", "time"], kind="stable").reset_index(drop=True)


def case_table(df):
    """One row per case with a target and a release at or after the target."""
    rows, drops = [], {"no_target": 0, "no_release": 0, "release_before_target": 0}
    for cid, g in df.groupby("case", sort=False):
        n, t = g["concept:name"], g["time"]
        tt = t[n.eq(TARGET)]
        rel = t[n.str.startswith("Release ", na=False)]
        if tt.empty:
            drops["no_target"] += 1
            continue
        if rel.empty:
            drops["no_release"] += 1
            continue
        t0, end = tt.min(), rel.min()
        if end < t0:
            drops["release_before_target"] += 1
            continue
        reg = g[n.eq("ER Registration")]
        reg = reg.iloc[0] if len(reg) else g.iloc[0]
        row = {"case": cid, "t0": t0,
               "days": -(end - t0).total_seconds() / 86400.0,
               "return_er": -int((n.eq("Return ER") & t.gt(t0)).any()),
               "icu": -int((n.eq("Admission IC") & t.gt(t0)).any())}
        for a in ER_ACTIONS:  # first occurrence strictly before the target
            ta = t[n.eq(a) & t.lt(t0)]
            row["t_" + a] = ta.min() if len(ta) else pd.NaT
        for c in COVARIATES_EXT:
            row[c] = reg.get(c, np.nan)
        rows.append(row)
    C = pd.DataFrame(rows)
    for c in COVARIATES_EXT:
        C[c] = pd.to_numeric(C[c].map({True: 1, False: 0}).fillna(C[c]), errors="coerce")
    C["age_missing"] = C["Age"].isna().astype(int)
    C[COVARIATES_EXT] = C[COVARIATES_EXT].fillna(C[COVARIATES_EXT].median())
    return C, drops


def family(C, u, w, delta_min=0.0):
    """Path of each case in the diamond (empty; u, w) before the target; ties within delta excluded."""
    tu, tw = C["t_" + u], C["t_" + w]
    gap = (tw - tu).dt.total_seconds() / 60.0
    path = pd.Series("none", index=C.index)
    path[tu.notna() & tw.isna()] = "u"
    path[tu.isna() & tw.notna()] = "w"
    both = tu.notna() & tw.notna()
    path[both & (gap > delta_min)] = "uw"
    path[both & (gap < -delta_min)] = "wu"
    path[both & (gap.abs() <= delta_min)] = "tie"
    E = C.copy()
    E["path"] = path
    E["gap_min"] = gap
    return E


def design(E, cols=None):
    X = E[(cols or COVARIATES) + ["age_missing"]].to_numpy(float)
    sd = X.std(0)
    sd[sd == 0] = 1
    return (X - X.mean(0)) / sd


# ----------------------------------------------------------------------------- estimators
def aipw_binary(X, A, Y, seed=SEED, folds=5, clip=PS_CLIP, C=PS_C):
    """Cross-fitted AIPW estimate of E[Y(1)] - E[Y(0)] and propensities."""
    n = len(Y)
    ps, m1, m0 = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        if A[tr].min() == A[tr].max() or A[tr].sum() < 2 or (1 - A[tr]).sum() < 2:
            return math.nan, math.nan, np.full(n, np.nan)
        lr = LogisticRegression(C=C, max_iter=5000).fit(X[tr], A[tr])
        ps[te] = lr.predict_proba(X[te])[:, 1]
        m1[te] = Ridge(alpha=1.0).fit(X[tr][A[tr] == 1], Y[tr][A[tr] == 1]).predict(X[te])
        m0[te] = Ridge(alpha=1.0).fit(X[tr][A[tr] == 0], Y[tr][A[tr] == 0]).predict(X[te])
    ps = np.clip(ps, clip, 1 - clip)
    psi = m1 - m0 + A * (Y - m1) / ps - (1 - A) * (Y - m0) / (1 - ps)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(n)), ps


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def order_effect(E, outcome, B=B_BOOT, seed=SEED, cols=None, clip=PS_CLIP, C=PS_C, winsor=None):
    D = E[E.path.isin(["uw", "wu"])]
    A = (D.path == "wu").to_numpy(int)
    Y = D[outcome].to_numpy(float)
    if winsor is not None:
        Y = np.maximum(Y, np.quantile(Y, winsor))  # outcomes are negative durations: cap the longest stays
    X = design(D, cols)
    naive = float(Y[A == 1].mean() - Y[A == 0].mean())
    est, se, ps = aipw_binary(X, A, Y, seed, clip=clip, C=C)
    rng = np.random.default_rng(seed)
    bn, ba = [], []
    for b in range(B):
        i = rng.integers(0, len(D), len(D))
        Ab, Yb, Xb = A[i], Y[i], X[i]
        if Ab.sum() < 5 or (1 - Ab).sum() < 5:
            continue
        bn.append(Yb[Ab == 1].mean() - Yb[Ab == 0].mean())
        v = aipw_binary(Xb, Ab, Yb, seed + b, clip=clip, C=C)[0]
        if not math.isnan(v):
            ba.append(v)
    q = lambda v: np.quantile(v, [0.025, 0.975]) if len(v) else np.array([np.nan, np.nan])
    z = est / np.std(ba, ddof=1) if len(ba) > 1 else np.nan
    return dict(n_uw=int((A == 0).sum()), n_wu=int(A.sum()), naive=naive,
                naive_lo=q(bn)[0], naive_hi=q(bn)[1], aipw=est, aipw_lo=q(ba)[0], aipw_hi=q(ba)[1],
                p_aipw=float(2 * stats.norm.sf(abs(z))),
                ps_min=float(ps.min()), ps_q05=float(np.quantile(ps, 0.05)),
                ps_q95=float(np.quantile(ps, 0.95)), ps_max=float(ps.max()),
                ess_wu=ess(A / ps), ess_uw=ess((1 - A) / (1 - ps)),
                omega_wu=float(np.median(1 / ps[A == 1])), omega_uw=float(np.median(1 / (1 - ps[A == 0]))))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ----------------------------------------------------------------------------- S5 planning
PATHS = ["none", "u", "w", "uw", "wu"]


def dr_path_values(Xtr, Ptr, Ytr, Xev, Pev, Yev):
    """Doubly robust path values on the evaluation set with nuisances fitted on the training set."""
    cls = [p for p in PATHS if (Ptr == p).sum() >= 5]
    lr = LogisticRegression(C=PS_C, max_iter=5000).fit(Xtr, Ptr)
    prob = pd.DataFrame(lr.predict_proba(Xev), columns=lr.classes_)
    out = {}
    for p in cls:
        m = Ridge(alpha=1.0).fit(Xtr[Ptr == p], Ytr[Ptr == p]).predict(Xev)
        e = np.clip(prob[p].to_numpy(), PS_CLIP, 1)
        psi = m + (Pev == p) * (Yev - m) / e
        out[p] = (float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi))))
    return out


def planning(E, outcome="days", n_splits=N_SPLITS, seed=SEED):
    D = E[E.path.isin(PATHS)].reset_index(drop=True)
    X, P, Y = design(D), D.path.to_numpy(), D[outcome].to_numpy(float)
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_splits):
        tr = np.zeros(len(D), bool)
        for p in PATHS:
            idx = np.flatnonzero(P == p)
            rng.shuffle(idx)
            tr[idx[: int(round(0.7 * len(idx)))]] = True
        te = ~tr
        # training estimates: DR with cross-fitting inside the training part
        fit_idx, ev_idx = np.flatnonzero(tr), np.flatnonzero(tr)
        kf = KFold(5, shuffle=True, random_state=s)
        acc = {p: [] for p in PATHS}
        for a, b in kf.split(fit_idx):
            vals = dr_path_values(X[fit_idx[a]], P[fit_idx[a]], Y[fit_idx[a]],
                                  X[fit_idx[b]], P[fit_idx[b]], Y[fit_idx[b]])
            for p, (m, _) in vals.items():
                acc[p].append((m, len(b)))
        est = {p: sum(m * k for m, k in v) / sum(k for _, k in v) for p, v in acc.items() if v}
        nsup = {p: int((P[tr] == p).sum()) for p in PATHS}
        se = {p: np.std(Y[tr][P[tr] == p], ddof=1) / np.sqrt(max(nsup[p], 2)) for p in est}
        best = lambda cands, score: max([c for c in cands if c in score], key=lambda c: score[c])
        endpoint = {"none": est.get("none"), "u": est.get("u"), "w": est.get("w"),
                    "uw": (nsup["uw"] * est["uw"] + nsup["wu"] * est["wu"]) / (nsup["uw"] + nsup["wu"])}
        chosen = {
            "sequence_sensitive": best(PATHS, est),
            "pessimistic": best(PATHS, {p: est[p] - se[p] for p in est}),
            "reference_path": best(["none", "u", "w", "uw"], est),
            "greedy": best(["none", "u", "w"], est),
            "order_insensitive": best(["none", "u", "w", "uw"], {k: v for k, v in endpoint.items() if v is not None}),
            "frequency": max(PATHS, key=lambda p: nsup[p]),
        }
        test_vals = dr_path_values(X[tr], P[tr], Y[tr], X[te], P[te], Y[te])
        for pol, path in chosen.items():
            rows.append(dict(split=s, policy=pol, path=path, value=test_vals[path][0]))
    L = pd.DataFrame(rows)
    ref = L[L.policy == "reference_path"].set_index("split")["value"]
    S = []
    for pol, g in L.groupby("policy", sort=False):
        g = g.set_index("split")
        gain = g["value"] - ref
        S.append(dict(policy=pol, mode_path=g["path"].mode().iloc[0],
                      mode_freq=float((g["path"] == g["path"].mode().iloc[0]).mean()),
                      value=float(g["value"].mean()), gain_vs_ref=float(gain.mean()),
                      gain_q05=float(gain.quantile(0.05)), gain_q95=float(gain.quantile(0.95)),
                      win_vs_ref=float((gain > 1e-12).mean())))
    return L, pd.DataFrame(S)


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_events()
    C, drops = case_table(df)
    info = dict(n_cases_log=int(df.case.nunique()), n_events=len(df), n_activities=int(df["concept:name"].nunique()),
                n_cases_analysed=len(C), **drops)
    pd.DataFrame([info]).to_csv(os.path.join(OUT, "dataset_summary.csv"), index=False)
    print(info)

    # S1 main diamond
    E = family(C, U, W)
    support = E.path.value_counts().reindex(["none", "u", "w", "uw", "wu", "tie"], fill_value=0)
    print("S1 support", support.to_dict())
    S1 = []
    for y in OUTCOMES:
        r = order_effect(E[E.path != "tie"], y)
        r["outcome"] = y
        S1.append(r)
    S1 = pd.DataFrame(S1)
    S1.to_csv(os.path.join(OUT, "s1_main_diamond.csv"), index=False)
    support.to_csv(os.path.join(OUT, "s1_support.csv"))
    gaps = E[E.path.isin(["uw", "wu"])].groupby("path")["gap_min"].apply(lambda s: s.abs().median())
    gaps.to_csv(os.path.join(OUT, "s1_median_gap_minutes.csv"))
    print(S1.round(3).to_string(index=False))
    print("median |gap| (min)", gaps.round(1).to_dict())

    # S2 all pairs
    S2 = []
    for a, b in itertools.combinations(ER_ACTIONS, 2):
        Ef = family(C, a, b)
        cnt = Ef.path.value_counts()
        row = dict(u=a, w=b, **{"n_" + k: int(cnt.get(k, 0)) for k in ["none", "u", "w", "uw", "wu", "tie"]})
        row["endpoint_uw"] = row["n_uw"] + row["n_wu"]
        if min(row["n_uw"], row["n_wu"]) >= MIN_ORDER_SUPPORT:
            r = order_effect(Ef[Ef.path != "tie"], "days")
            row.update(naive=r["naive"], aipw=r["aipw"], aipw_lo=r["aipw_lo"], aipw_hi=r["aipw_hi"],
                       p=r["p_aipw"], ess_wu=r["ess_wu"], ess_uw=r["ess_uw"])
        S2.append(row)
    S2 = pd.DataFrame(S2)
    m = S2["p"].notna()
    S2.loc[m, "p_holm"] = holm(S2.loc[m, "p"])
    S2.to_csv(os.path.join(OUT, "s2_all_pairs.csv"), index=False)
    print(S2.round(3).to_string(index=False))

    # S3 tie window
    S3 = []
    for d in [0, 5, 15, 30, 60, 120]:
        Ed = family(C, U, W, delta_min=d)
        n_wu = int((Ed.path == "wu").sum())
        if n_wu >= 10:
            r = order_effect(Ed[Ed.path != "tie"], "days")
        else:
            r = dict(n_uw=int((Ed.path == "uw").sum()), n_wu=n_wu, aipw=np.nan, aipw_lo=np.nan, aipw_hi=np.nan)
        S3.append(dict(delta_min=d, n_uw=r["n_uw"], n_wu=r["n_wu"], n_tie=int((Ed.path == "tie").sum()),
                       aipw=r["aipw"], lo=r["aipw_lo"], hi=r["aipw_hi"]))
    S3 = pd.DataFrame(S3)
    S3.to_csv(os.path.join(OUT, "s3_tie_window.csv"), index=False)
    print(S3.round(3).to_string(index=False))

    # S4 additivity check with IV Liquid as the third action
    third = "IV Liquid"
    t3 = C["t_" + third]
    tu, tw = C["t_" + U], C["t_" + W]
    later = t3.notna() & tu.notna() & tw.notna() & (t3 > tu) & (t3 > tw)
    absent = t3.isna()
    S4 = []
    for name, mask in [("third absent", absent), ("third after both", later)]:
        Ef = family(C[mask], U, W)
        r = order_effect(Ef[Ef.path != "tie"], "days")
        S4.append(dict(stratum=name, n_uw=r["n_uw"], n_wu=r["n_wu"], aipw=r["aipw"], lo=r["aipw_lo"], hi=r["aipw_hi"]))
    S4 = pd.DataFrame(S4)
    # bootstrap difference
    rng = np.random.default_rng(SEED)
    idx_a, idx_l = np.flatnonzero(absent), np.flatnonzero(later)
    diffs = []
    for b in range(300):
        ca = C.iloc[rng.choice(idx_a, len(idx_a))]
        cl = C.iloc[rng.choice(idx_l, len(idx_l))]
        ea, el = family(ca, U, W), family(cl, U, W)
        va = order_effect(ea[ea.path != "tie"], "days", B=0)["aipw"] if (ea.path == "wu").sum() >= 5 else np.nan
        vl = order_effect(el[el.path != "tie"], "days", B=0)["aipw"] if (el.path == "wu").sum() >= 5 else np.nan
        diffs.append(vl - va)
    diffs = np.array(diffs)
    diffs = diffs[~np.isnan(diffs)]
    S4["diff_later_minus_absent"] = S4.aipw.iloc[1] - S4.aipw.iloc[0]
    S4["diff_lo"], S4["diff_hi"] = np.quantile(diffs, [0.025, 0.975])
    S4.to_csv(os.path.join(OUT, "s4_additivity.csv"), index=False)
    print(S4.round(3).to_string(index=False))

    # S6 specification sensitivity of the main effect (days)
    S6 = []
    Em = E[E.path != "tie"]
    for name, kw in [("primary", {}),
                     ("extended covariates", dict(cols=COVARIATES_EXT)),
                     ("weaker regularization (C=1)", dict(C=1.0)),
                     ("clip 0.01", dict(clip=0.01)),
                     ("winsorized at 99\\%", dict(winsor=0.01)),
                     ("extended, C=1, clip 0.01", dict(cols=COVARIATES_EXT, C=1.0, clip=0.01)),
                     ("unadjusted", None)]:
        if kw is None:
            r = order_effect(Em, "days")
            S6.append(dict(spec=name, est=r["naive"], lo=r["naive_lo"], hi=r["naive_hi"], ess_wu=np.nan))
        else:
            r = order_effect(Em, "days", **kw)
            S6.append(dict(spec=name, est=r["aipw"], lo=r["aipw_lo"], hi=r["aipw_hi"], ess_wu=r["ess_wu"]))
    S6 = pd.DataFrame(S6)
    S6.to_csv(os.path.join(OUT, "s6_sensitivity.csv"), index=False)
    print(S6.round(3).to_string(index=False))

    # S5 planning
    L, S5 = planning(E[E.path != "tie"])
    L.to_csv(os.path.join(OUT, "s5_planning_long.csv"), index=False)
    S5.to_csv(os.path.join(OUT, "s5_planning_summary.csv"), index=False)
    print(S5.round(3).to_string(index=False))
    write_tables(support, S1, S2, S3, S4, S5)
    write_sensitivity(S3, S4, S6)


def write_tables(support, S1, S2, S3, S4, S5):
    f = lambda x, d=2: f"{x:.{d}f}"
    lab = {"days": "days to release", "return_er": "return to ER", "icu": "ICU admission"}
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Outcome & naive [95\% CI] & AIPW [95\% CI] & $p$ \\", r"\midrule"]
    for _, r in S1.iterrows():
        L.append(f"{lab[r.outcome]} & {f(r.naive)} [{f(r.naive_lo)}, {f(r.naive_hi)}] & "
                 f"{f(r.aipw)} [{f(r.aipw_lo)}, {f(r.aipw_hi)}] & {f(r.p_aipw)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_sepsis_main.tex"), "w").write("\n".join(L) + "\n")
    ab = {"CRP": "CRP", "Leucocytes": "Leuc", "LacticAcid": "LA", "IV Liquid": "IVL", "IV Antibiotics": "IVA"}
    L = [r"\begin{tabular}{@{}lrrrrrc@{}}", r"\toprule",
         r"Pair & $\{u,v\}$ & $u\to v$ & $v\to u$ & ties & $\hat\kappa$ & 95\% CI \\",
         r"\midrule"]
    for _, r in S2.iterrows():
        est = not pd.isna(r.get("aipw", np.nan))
        k = f(r.aipw) if est else "--"
        ci = f"[{f(r.aipw_lo)}, {f(r.aipw_hi)}]" if est else ""
        L.append(f"{ab[r.u]}, {ab[r.w]} & {int(r.endpoint_uw + r.n_tie)} & {int(r.n_uw)} & {int(r.n_wu)} & {int(r.n_tie)} & {k} & {ci} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_sepsis_pairs.tex"), "w").write("\n".join(L) + "\n")
    names = {"sequence_sensitive": "sequence-sensitive", "pessimistic": "pessimistic",
             "reference_path": "reference-path", "greedy": "greedy", "order_insensitive": "order-insensitive",
             "frequency": "frequency"}
    L = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"Policy & modal plan (freq.) & value & gain vs.\ ref.\ [5\%, 95\%] & win \\", r"\midrule"]
    pl = {"none": r"$\varnothing$", "u": r"$u$", "w": r"$v$", "uw": r"$u\to v$", "wu": r"$v\to u$"}
    for _, r in S5.iterrows():
        L.append(f"{names[r.policy]} & {pl[r.mode_path]} ({f(r.mode_freq)}) & {f(r.value)} & "
                 f"{f(r.gain_vs_ref)} [{f(r.gain_q05)}, {f(r.gain_q95)}] & {f(r.win_vs_ref)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_sepsis_planning.tex"), "w").write("\n".join(L) + "\n")


def write_sensitivity(S3, S4, S6):
    f = lambda x: f"{x:.2f}"
    L = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Specification & AIPW $\hat\kappa$ [95\% CI] \\", r"\midrule"]
    for _, r in S6.iterrows():
        L.append(f"{r.spec} & {f(r.est)} [{f(r.lo)}, {f(r.hi)}] \\\\")
    L += [r"\midrule", r"Tie window $\delta$ (min): $n_{u\to v}/n_{v\to u}$ & \\"]
    for _, r in S3.iterrows():
        eff = f"{f(r.aipw)} [{f(r.lo)}, {f(r.hi)}]" if not pd.isna(r.aipw) else "-- (fewer than 10 cases)"
        L.append(f"$\\delta={int(r.delta_min)}$: {int(r.n_uw)}/{int(r.n_wu)} & {eff} \\\\")
    L += [r"\midrule"]
    for _, r in S4.iterrows():
        L.append(f"IV Liquid {r.stratum.replace('third ', '')}: {int(r.n_uw)}/{int(r.n_wu)} & {f(r.aipw)} [{f(r.lo)}, {f(r.hi)}] \\\\")
    L.append(f"difference (after $-$ absent) & {f(S4.diff_later_minus_absent.iloc[0])} [{f(S4.diff_lo.iloc[0])}, {f(S4.diff_hi.iloc[0])}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_sepsis_sensitivity.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
