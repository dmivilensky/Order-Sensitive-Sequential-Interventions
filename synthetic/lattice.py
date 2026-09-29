"""Utilities for prerequisite posets and their ideal lattices (bitmask representation)."""
from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np


class Poset:
    """Finite poset on actions 0..n-1. The identity order 0<1<...<n-1 is a linear extension (tau)."""

    def __init__(self, n: int, relations: Sequence[Tuple[int, int]]):
        self.n = n
        less = np.zeros((n, n), dtype=bool)
        for a, b in relations:
            assert a < b, "relations must respect the identity linear extension"
            less[a, b] = True
        # transitive closure
        for k in range(n):
            less |= less[:, [k]] & less[[k], :]
        self.less = less
        self.pred = [int(sum(1 << a for a in range(n) if less[a, b])) for b in range(n)]

    def unrelated(self, a: int, b: int) -> bool:
        return a != b and not self.less[a, b] and not self.less[b, a]

    def admissible(self, K: int) -> List[int]:
        return [a for a in range(self.n) if not (K >> a) & 1 and (self.pred[a] & ~K) == 0]

    def width(self) -> int:
        best = 1
        for r in range(2, self.n + 1):
            found = False
            for A in itertools.combinations(range(self.n), r):
                if all(self.unrelated(a, b) for a, b in itertools.combinations(A, 2)):
                    found = True
                    break
            if not found:
                break
            best = r
        return best

    def ideals_within(self, I: int, H: int) -> List[int]:
        """All ideals K >= I with |K \\ I| <= H, in BFS (depth) order."""
        seen = {I}
        frontier = [I]
        out = [I]
        for _ in range(H):
            nxt = []
            for K in frontier:
                for a in self.admissible(K):
                    L = K | (1 << a)
                    if L not in seen:
                        seen.add(L)
                        nxt.append(L)
                        out.append(L)
            frontier = nxt
        return out

    def diamonds_within(self, I: int, H: int) -> List[Tuple[int, int, int]]:
        """Diamonds (K;u,v), u<v unrelated, with K >= I and |K\\I| <= H-2."""
        res = []
        for K in self.ideals_within(I, max(H - 2, 0)):
            A = self.admissible(K)
            for u, v in itertools.combinations(sorted(A), 2):
                res.append((K, u, v))
        return res


def popcount(x: int) -> int:
    return bin(x).count("1")


def random_poset(n: int, p: float, rng: np.random.Generator) -> Poset:
    rel = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < p]
    return Poset(n, rel)


def layered_poset(w: int, m: int, p_cross: float, rng: np.random.Generator) -> Poset:
    """w chains of length m; action (chain j, level i) has index i*w+j. Chain edges plus random
    cross edges from level i to level i+1. Width is at most w."""
    rel = []
    for i in range(m - 1):
        for j in range(w):
            rel.append((i * w + j, (i + 1) * w + j))
            for k in range(w):
                if k != j and rng.random() < p_cross:
                    rel.append((i * w + j, (i + 1) * w + k))
    return Poset(w * m, rel)


def dp_plan(P: Poset, I: int, H: int, g) -> Tuple[float, List[int]]:
    """Longest-path DP with stopping (Theorem 6.2). g(K, a) -> float (may be -inf)."""
    depth_cap = popcount(I) + H

    @lru_cache(maxsize=None)
    def U(K: int) -> Tuple[float, int]:
        if popcount(K) >= depth_cap:
            return 0.0, -1
        best, arg = 0.0, -1
        for a in P.admissible(K):
            val = g(K, a)
            if val == -np.inf:
                continue
            v = val + U(K | (1 << a))[0]
            if v > best:
                best, arg = v, a
        return best, arg

    plan, K = [], I
    while True:
        _, a = U(K)
        if a < 0:
            break
        plan.append(a)
        K |= 1 << a
    return U(I)[0], plan


def plan_value(I: int, plan: Sequence[int], g) -> float:
    K, v = I, 0.0
    for a in plan:
        v += g(K, a)
        K |= 1 << a
    return v


def reference_path(I: int, J: int) -> List[int]:
    D = J & ~I
    return [a for a in range(D.bit_length()) if (D >> a) & 1]
