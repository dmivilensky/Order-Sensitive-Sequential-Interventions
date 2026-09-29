"""E1. Support separation, quantitatively (Theorem 5.4, Proposition 5.5).

One diamond (empty; u, v). A behaviour policy takes the reverse order v-first with probability q.
After the first action an intermediate signal S is observed; it affects both the decision to take the
second action and the outcome, so naive path means are biased and the history-based g-formula / IPW
is needed. We report the RMSE of the reference score Phi = Q(u,v) and of kappa = Q(v,u) - Q(u,v),
for IPW and naive estimators, and the coverage of the Bernstein bound of Proposition 5.5.
"""
import json
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

P_S = {"u": 0.3, "v": 0.7}           # P(S=1 | first action)
P_CONT = {0: 0.9, 1: 0.3}            # P(take the second action | S)
MU = {("u", "v"): 0.40, ("v", "u"): 0.50}
BETA_S = 0.25
NOISE = 0.10                          # uniform noise half-width; outcomes stay in [0, 1]
B = 1.0
TRUE_Q = {k: MU[k] + BETA_S * P_S[k[0]] for k in MU}
TRUE_PHI = TRUE_Q[("u", "v")]
TRUE_KAPPA = TRUE_Q[("v", "u")] - TRUE_Q[("u", "v")]


def simulate(n: int, q: float, rng: np.random.Generator) -> dict:
    first_v = rng.random(n) < q
    first = np.where(first_v, "v", "u")
    pS = np.where(first_v, P_S["v"], P_S["u"])
    S = (rng.random(n) < pS).astype(int)
    pc = np.where(S == 1, P_CONT[1], P_CONT[0])
    cont = rng.random(n) < pc
    mu = np.where(first_v, MU[("v", "u")], MU[("u", "v")])
    R2 = mu + BETA_S * S + rng.uniform(-NOISE, NOISE, n)
    pi0 = np.where(first_v, q, 1 - q)
    return dict(first_v=first_v, S=S, cont=cont, R2=R2, pi0=pi0, pc=pc)


def estimates(d: dict, n: int) -> dict:
    out = {}
    for side, is_v in (("uv", False), ("vu", True)):
        follow = (d["first_v"] == is_v) & d["cont"]
        W = np.where(follow, 1.0 / (d["pi0"] * d["pc"]), 0.0)
        out[f"ipw_{side}"] = float(np.mean(W * d["R2"]))
        out[f"naive_{side}"] = float(np.mean(d["R2"][follow])) if follow.any() else np.nan
        out[f"n_{side}"] = int(follow.sum())
    return out


def bernstein(omega: float, n: int, delta: float) -> float:
    l = np.log(2 / delta)
    return B * (np.sqrt(2 * omega * l / n) + 2 * omega * l / (3 * n))


def main(n: int = 4000, reps: int = 1000, seed: int = 0, delta: float = 0.05):
    rng = np.random.default_rng(seed)
    qs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    rows = []
    for q in qs:
        omega_uv = 1.0 / ((1 - q) * P_CONT[1])
        omega_vu = 1.0 / (q * P_CONT[1])
        bound_kappa = bernstein(omega_uv, n, delta / 2) + bernstein(omega_vu, n, delta / 2)
        bound_phi = bernstein(omega_uv, n, delta)
        errs = []
        for _ in range(reps):
            e = estimates(simulate(n, q, rng), n)
            errs.append({
                "phi_ipw": e["ipw_uv"] - TRUE_PHI,
                "kappa_ipw": (e["ipw_vu"] - e["ipw_uv"]) - TRUE_KAPPA,
                "phi_naive": e["naive_uv"] - TRUE_PHI,
                "kappa_naive": (e["naive_vu"] - e["naive_uv"]) - TRUE_KAPPA,
                "n_vu": e["n_vu"],
            })
        E = pd.DataFrame(errs)
        rows.append({
            "q": q, "n": n, "reps": reps,
            "omega_uv": omega_uv, "omega_vu": omega_vu,
            "rmse_phi_ipw": float(np.sqrt(np.mean(E.phi_ipw ** 2))),
            "rmse_kappa_ipw": float(np.sqrt(np.mean(E.kappa_ipw ** 2))),
            "bias_phi_naive": float(np.nanmean(E.phi_naive)),
            "bias_kappa_naive": float(np.nanmean(E.kappa_naive)),
            "rmse_kappa_naive": float(np.sqrt(np.nanmean(E.kappa_naive ** 2))),
            "frac_no_reverse_follower": float(np.mean(E.n_vu == 0)),
            "mean_reverse_followers": float(E.n_vu.mean()),
            "bound_phi": bound_phi, "bound_kappa": bound_kappa,
            "coverage_phi": float(np.mean(np.abs(E.phi_ipw) <= bound_phi)),
            "coverage_kappa": float(np.mean(np.abs(E.kappa_ipw) <= bound_kappa)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e1_support_separation.csv"), index=False)
    with open(os.path.join(OUT, "e1_truth.json"), "w") as f:
        json.dump({"phi": TRUE_PHI, "kappa": TRUE_KAPPA}, f)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
