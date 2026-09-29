"""E3. Cube consistency as a statistical tool (Theorem 4.3, Remark 4.5).

Five pairwise unrelated actions: J(P) is the Boolean lattice on 5 elements (32 ideals, 80 edges,
80 diamonds, 80 three-cubes). The order-effect map g -> kappa_g has rank 80 - 31 = 49, so the
cube-consistent subspace has codimension 31.
(a) Null (C3 holds): kappa = kappa_g for a random g, observed with iid N(0, sigma^2) noise. We report
    the ratio of squared errors after/before least-squares projection (theory: 49/80).
(b) Alternative (C3 fails): the value of an action depends on the ORDER of two earlier actions with
    strength lam. Local order effects are then defined with reference prefixes and are no longer
    cube-consistent. The residual statistic T = ||k - Pk||^2 / sigma^2 is chi^2_31 under the null;
    we report the rejection rate at level 5% as a function of lam / sigma.
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

from lattice import Poset, popcount, reference_path

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

N = 5
P = Poset(N, [])
IDEALS = list(range(1 << N))
EDGES = [(K, a) for K in IDEALS for a in P.admissible(K)]
EIDX = {e: i for i, e in enumerate(EDGES)}
DIAMONDS = [(K, u, v) for K in IDEALS for u, v in itertools.combinations(P.admissible(K), 2)]


def coboundary_matrix():
    M = np.zeros((len(DIAMONDS), len(EDGES)))
    for i, (K, u, v) in enumerate(DIAMONDS):
        M[i, EIDX[(K, v)]] += 1
        M[i, EIDX[(K | 1 << v, u)]] += 1
        M[i, EIDX[(K, u)]] -= 1
        M[i, EIDX[(K | 1 << u, v)]] -= 1
    return M


M = coboundary_matrix()
RANK = np.linalg.matrix_rank(M)
Q_, _ = np.linalg.qr(M)
U, s, _ = np.linalg.svd(M, full_matrices=False)
BASIS = U[:, :RANK]                      # orthonormal basis of the cube-consistent subspace


def project(k):
    return BASIS @ (BASIS.T @ k)


def path_value_nonadditive(path, g, lam, W):
    """Additive part sum g plus lam * sum_c sum_{a<b before c} W[a,b,c] * 1[b before a]."""
    K, v, seen = 0, 0.0, []
    for c in path:
        v += g[EIDX[(K, c)]]
        for a, b in itertools.combinations(sorted(seen), 2):
            if seen.index(b) < seen.index(a):
                v += lam * W[a, b, c]
        seen.append(c)
        K |= 1 << c
    return v


def kappa_field(g, lam, W):
    out = np.zeros(len(DIAMONDS))
    for i, (K, u, v) in enumerate(DIAMONDS):
        pre = reference_path(0, K)
        out[i] = path_value_nonadditive(pre + [v, u], g, lam, W) - path_value_nonadditive(pre + [u, v], g, lam, W)
    return out


def main(reps=2000, sigma=1.0, seed=3):
    rng = np.random.default_rng(seed)
    df_null = len(DIAMONDS) - RANK
    crit = stats.chi2.ppf(0.95, df_null)
    # (a) null: error reduction and size
    ratios, rej0 = [], []
    for _ in range(reps):
        g = rng.normal(size=len(EDGES))
        k = M @ g
        kh = k + sigma * rng.normal(size=len(k))
        pk = project(kh)
        ratios.append(np.sum((pk - k) ** 2) / np.sum((kh - k) ** 2))
        rej0.append(np.sum((kh - pk) ** 2) / sigma ** 2 > crit)
    # (b) alternatives
    rows = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        rej, pop_res = [], []
        for _ in range(reps // 4):
            g = rng.normal(size=len(EDGES))
            W = rng.normal(size=(N, N, N))
            k = kappa_field(g, lam, W)
            pop_res.append(np.sqrt(np.sum((k - project(k)) ** 2)))
            kh = k + sigma * rng.normal(size=len(k))
            rej.append(np.sum((kh - project(kh)) ** 2) / sigma ** 2 > crit)
        rows.append(dict(lam=lam, power=float(np.mean(rej)), pop_residual=float(np.mean(pop_res))))
    A = pd.DataFrame(rows)
    summary = dict(n_ideals=len(IDEALS), n_edges=len(EDGES), n_diamonds=len(DIAMONDS), rank=int(RANK),
                   df=int(df_null), theory_ratio=RANK / len(DIAMONDS),
                   mean_ratio=float(np.mean(ratios)), size=float(np.mean(rej0)))
    pd.DataFrame([summary]).to_csv(os.path.join(OUT, "e3_null.csv"), index=False)
    A.to_csv(os.path.join(OUT, "e3_power.csv"), index=False)
    print(summary)
    print(A.to_string(index=False))


if __name__ == "__main__":
    main()
