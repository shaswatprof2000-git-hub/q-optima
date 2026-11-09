"""Delivery / site-visit routing as a Travelling Salesman Problem (TSP).

QUBO: city 0 is the depot, fixed at the start and end of the tour. For the other
m = n-1 stops, x[c,t] = 1 means "stop c is visited at position t".  m^2 qubits.
    minimise  sum of leg distances
    s.t.      every stop exactly once, every position exactly once (penalties)

Classical baselines: Held-Karp dynamic programming (exact) and
nearest-neighbour + 2-opt (the fast heuristic a real dispatcher would use).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from ..qubo import QUBO
from ..solvers.base import SolveResult


@dataclass
class RoutingProblem:
    names: list[str]
    dist: np.ndarray  # (n, n) km; index 0 = depot

    @classmethod
    def from_latlon(cls, places: dict[str, tuple[float, float]], road_factor: float = 1.3):
        """Haversine distance x road_factor (roads are longer than straight lines)."""
        names = list(places)
        coords = np.radians(np.array([places[k] for k in names]))
        lat, lon = coords[:, 0][:, None], coords[:, 1][:, None]
        a = (np.sin((lat - lat.T) / 2) ** 2
             + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2) ** 2)
        d = 2 * 6371.0 * np.arcsin(np.sqrt(a)) * road_factor
        return cls(names, d)

    @classmethod
    def random(cls, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 20, (n, 2))
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        return cls([f"S{i}" for i in range(n)], d)

    @property
    def n(self) -> int:
        return len(self.names)

    # ---- QUBO ------------------------------------------------------------
    def var(self, c: int, t: int) -> int:
        """Variable index for stop c (1..m) at position t (1..m)."""
        m = self.n - 1
        return (c - 1) * m + (t - 1)

    def to_qubo(self, penalty: float | None = None) -> QUBO:
        m = self.n - 1
        d = self.dist
        A = penalty if penalty is not None else 1.5 * float(d.max())
        names = [f"{self.names[c]}@{t}" for c in range(1, self.n) for t in range(1, m + 1)]
        q = QUBO(m * m, var_names=names)
        stops = range(1, self.n)
        pos = range(1, m + 1)
        for c in stops:
            q.add_exactly_k([self.var(c, t) for t in pos], 1, A)
        for t in pos:
            q.add_exactly_k([self.var(c, t) for c in stops], 1, A)
        for c in stops:  # depot legs
            q.add_linear(self.var(c, 1), d[0, c])
            q.add_linear(self.var(c, m), d[c, 0])
        for t in range(1, m):  # consecutive legs
            for u in stops:
                for v in stops:
                    if u != v:
                        q.add_quadratic(self.var(u, t), self.var(v, t + 1), d[u, v])
        return q

    def decode(self, bits) -> tuple[list[int] | None, float | None]:
        m = self.n - 1
        X = np.asarray(bits).reshape(m, m)  # rows: stops 1..m, cols: positions
        if not (np.all(X.sum(0) == 1) and np.all(X.sum(1) == 1)):
            return None, None
        order = [int(np.argmax(X[:, t])) + 1 for t in range(m)]
        tour = [0] + order + [0]
        return tour, self.tour_length(tour)

    def tour_length(self, tour) -> float:
        return float(sum(self.dist[a, b] for a, b in zip(tour, tour[1:])))

    def describe(self, tour) -> str:
        return " -> ".join(self.names[i] for i in tour)

    # ---- classical baselines --------------------------------------------
    def held_karp(self) -> SolveResult:
        t0 = time.perf_counter()
        n, d = self.n, self.dist
        C = {(1 << k, k): (d[0, k], 0) for k in range(1, n)}
        for size in range(2, n):
            for subset in itertools.combinations(range(1, n), size):
                bits = sum(1 << k for k in subset)
                for k in subset:
                    prev = bits & ~(1 << k)
                    C[(bits, k)] = min((C[(prev, j)][0] + d[j, k], j) for j in subset if j != k)
        full = (1 << n) - 2
        cost, last = min((C[(full, k)][0] + d[k, 0], k) for k in range(1, n))
        tour, bits = [0], full
        for _ in range(n - 1):
            tour.append(last)
            bits, last = bits & ~(1 << last), C[(bits, last)][1]
        tour = [0] + tour[1:][::-1] + [0]
        return SolveResult("classical_exact(held-karp)", np.array(tour), float(cost),
                           time.perf_counter() - t0)

    def nearest_neighbour_2opt(self) -> SolveResult:
        t0 = time.perf_counter()
        d, n = self.dist, self.n
        tour, left = [0], set(range(1, n))
        while left:
            nxt = min(left, key=lambda j: d[tour[-1], j])
            tour.append(nxt)
            left.remove(nxt)
        tour.append(0)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                    if self.tour_length(new) < self.tour_length(tour) - 1e-12:
                        tour, improved = new, True
        return SolveResult("classical_heuristic(nn+2opt)", np.array(tour),
                           self.tour_length(tour), time.perf_counter() - t0)


def qubits_needed(n_places: int) -> int:
    return (n_places - 1) ** 2

