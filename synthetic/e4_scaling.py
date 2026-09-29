"""E4. Planning cost (Theorem 6.2).

Layered posets: w chains of length m = 12 with random cross-chain prerequisites (probability 0.3),
so the width is at most w. For horizons H we count the states visited by the dynamic program,
compare with the bound C(H+w, w), count the plans in Gamma_H(empty) that exhaustive search would have
to enumerate, and time the DP. Values: random edge field.
"""
import math
import os
import time

import numpy as np
import pandas as pd

from lattice import dp_plan, layered_poset, popcount

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def count_plans(P, H):
    """Number of admissible paths from the empty ideal with at most H actions (incl. the empty plan)."""
    layer = {0: 1}
    total = 1
    for _ in range(H):
        nxt = {}
        for K, c in layer.items():
            for a in P.admissible(K):
                L = K | (1 << a)
                nxt[L] = nxt.get(L, 0) + c
        total += sum(nxt.values())
        layer = nxt
    return total


def main(seed=5, m=12, reps=5):
    rng = np.random.default_rng(seed)
    rows = []
    for w in [1, 2, 3, 4, 6]:
        for H in [4, 8, 12, 16, 20]:
            for r in range(reps):
                P = layered_poset(w, m, 0.3, rng)
                if H > P.n:
                    continue
                vals = {}

                def g(K, a):
                    key = (K, a)
                    if key not in vals:
                        vals[key] = float(rng.normal())
                    return vals[key]

                states = P.ideals_within(0, H)
                t0 = time.perf_counter()
                dp_plan(P, 0, H, g)
                t_dp = time.perf_counter() - t0
                rows.append(dict(w=w, H=H, rep=r, n_actions=P.n, states=len(states),
                                 bound=math.comb(H + w, w), plans=count_plans(P, H), dp_seconds=t_dp,
                                 edges_evaluated=len(vals)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e4_scaling_long.csv"), index=False)
    S = df.groupby(["w", "H"]).agg(states=("states", "mean"), bound=("bound", "first"),
                                   plans=("plans", "mean"), dp_seconds=("dp_seconds", "mean"),
                                   edges=("edges_evaluated", "mean")).reset_index()
    S["states_le_bound"] = df.groupby(["w", "H"]).apply(lambda g: bool((g.states <= g.bound).all())).values
    S.to_csv(os.path.join(OUT, "e4_scaling_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
