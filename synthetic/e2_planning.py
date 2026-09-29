"""E2. When does modeling order pay off? (Theorem 6.1, Proposition 6.3.)

Random prerequisite posets; true edge field g(K,a) = mu_a + eps * sum_{b in K} J_{ab}, so every
local order effect equals eps*(J_uv - J_vu). A behaviour policy runs H steps, choosing among admissible
actions with probability proportional to exp(-beta * rank_tau(a)); beta > 0 makes reverse orders rare.
Per-step rewards are g + N(0, sigma^2). Planners (all over Gamma_H(empty), with stopping):
  oracle            DP on the true g
  sequence          DP on edge means hat g (edge coverage, Thm 5.4(3)); unvisited edges get value 0
  pessimistic       DP on hat g - c*sd/sqrt(n_e), c from a union bound; unvisited edges excluded
  pessimistic-1     the same with c = 1
  order-insensitive weighted LS fit of a zero-curvature (potential) model to edge means, best
                    endpoint, then its reference path
  endpoint-pooled   best endpoint by pooled mean cumulative reward, then its reference path
  greedy            one-step argmax of hat g
Regret is measured with the true g.
"""
import os

import numpy as np
import pandas as pd

from lattice import dp_plan, plan_value, popcount, random_poset, reference_path

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def make_instance(n_act, p, eps, rng):
    P = random_poset(n_act, p, rng)
    mu = rng.uniform(0.0, 0.5, n_act)
    J = rng.normal(0.0, 1.0, (n_act, n_act))

    def g(K, a):
        s = mu[a]
        Kb = K
        while Kb:
            b = (Kb & -Kb).bit_length() - 1
            s += eps * J[a, b]
            Kb &= Kb - 1
        return s

    return P, g


def simulate_log(P, g, H, n, beta, sigma, rng):
    edge_sum, edge_sq, edge_n = {}, {}, {}
    end_sum, end_n = {}, {}
    for _ in range(n):
        K, cum = 0, 0.0
        for _step in range(H):
            A = P.admissible(K)
            if not A:
                break
            ranks = np.argsort(np.argsort(A))
            w = np.exp(-beta * ranks)
            a = A[rng.choice(len(A), p=w / w.sum())]
            r = g(K, a) + sigma * rng.standard_normal()
            e = (K, a)
            edge_sum[e] = edge_sum.get(e, 0.0) + r
            edge_sq[e] = edge_sq.get(e, 0.0) + r * r
            edge_n[e] = edge_n.get(e, 0) + 1
            cum += r
            K |= 1 << a
            end_sum[K] = end_sum.get(K, 0.0) + cum
            end_n[K] = end_n.get(K, 0) + 1
    return edge_sum, edge_sq, edge_n, end_sum, end_n


def planners(P, g, H, log, sigma, delta=0.05):
    edge_sum, edge_sq, edge_n, end_sum, end_n = log
    ghat = {e: edge_sum[e] / edge_n[e] for e in edge_n}
    n_edges = max(len(edge_n), 1)
    c = np.sqrt(2 * np.log(2 * n_edges / delta))
    plans = {}
    plans["oracle"] = dp_plan(P, 0, H, g)[1]
    plans["sequence"] = dp_plan(P, 0, H, lambda K, a: ghat.get((K, a), 0.0))[1]
    for name, cc in (("pessimistic", c), ("pessimistic-1", 1.0)):
        plans[name] = dp_plan(
            P, 0, H,
            lambda K, a, cc=cc: ghat[(K, a)] - cc * sigma / np.sqrt(edge_n[(K, a)]) if (K, a) in edge_n else -np.inf)[1]
    # order-insensitive (pooled): best endpoint by pooled mean cumulative reward, then its reference path
    best_K, best_v = 0, 0.0
    for K, cnt in end_n.items():
        v = end_sum[K] / cnt
        if v > best_v:
            best_K, best_v = K, v
    plans["endpoint-pooled"] = reference_path(0, best_K)
    # order-insensitive (theory): weighted least-squares fit of a zero-curvature model
    # g(K,a) = Phi(K+a) - Phi(K) to the edge means (projection onto kappa = 0), best endpoint, reference path
    nodes = sorted({0} | {K for (K, a) in edge_n} | {K | (1 << a) for (K, a) in edge_n})
    col = {K: i for i, K in enumerate(nodes)}
    rows_, rhs = [], []
    for (K, a), ne in edge_n.items():
        wgt = np.sqrt(ne)
        r = np.zeros(len(nodes))
        r[col[K | (1 << a)]] += wgt
        r[col[K]] -= wgt
        rows_.append(r)
        rhs.append(wgt * ghat[(K, a)])
    r0 = np.zeros(len(nodes)); r0[col[0]] = 1e3
    A = np.vstack(rows_ + [r0]); bvec = np.array(rhs + [0.0])
    phi = np.linalg.lstsq(A, bvec, rcond=None)[0]
    kbest = max(nodes, key=lambda K: phi[col[K]] if popcount(K) <= H else -np.inf)
    plans["order-insensitive"] = reference_path(0, kbest) if phi[col[kbest]] > 0 else []
    # greedy
    K, plan = 0, []
    for _ in range(H):
        cands = [(ghat[(K, a)], a) for a in P.admissible(K) if (K, a) in ghat]
        if not cands:
            break
        v, a = max(cands)
        if v <= 0:
            break
        plan.append(a)
        K |= 1 << a
    plans["greedy"] = plan
    vals = {k: plan_value(0, p, g) for k, p in plans.items()}
    return {k: vals["oracle"] - v for k, v in vals.items() if k != "oracle"}


def run_setting(eps, beta, n, reps, rng, n_act=10, p=0.2, H=5, sigma=1.0):
    rows = []
    for r in range(reps):
        P, g = make_instance(n_act, p, eps, rng)
        log = simulate_log(P, g, H, n, beta, sigma, rng)
        reg = planners(P, g, H, log, sigma)
        reg.update(eps=eps, beta=beta, n=n, rep=r)
        rows.append(reg)
    return rows


def main(reps=200, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for eps in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        rows += [dict(x, sweep="eps") for x in run_setting(eps, 1.0, 400, reps, rng)]
    for beta in [0.0, 0.5, 1.0, 2.0, 3.0]:
        rows += [dict(x, sweep="beta") for x in run_setting(0.2, beta, 400, reps, rng)]
    for n in [100, 200, 400, 1000, 2500]:
        rows += [dict(x, sweep="n") for x in run_setting(0.2, 1.0, n, reps, rng)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e2_planning_long.csv"), index=False)
    pol = ["sequence", "pessimistic", "pessimistic-1", "order-insensitive", "endpoint-pooled", "greedy"]
    summ = []
    for (sw, key), g in df.groupby(["sweep", df.apply(lambda r: r[r["sweep"]], axis=1)]):
        row = {"sweep": sw, "value": key}
        for p_ in pol:
            row[p_ + "_mean"] = g[p_].mean()
            row[p_ + "_se"] = g[p_].std(ddof=1) / np.sqrt(len(g))
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(OUT, "e2_planning_summary.csv"), index=False)
    print(S.round(3).to_string(index=False))
    return S


if __name__ == "__main__":
    main()
