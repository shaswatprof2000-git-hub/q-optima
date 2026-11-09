"""Max-Cut: split a network into two groups so that as many links as possible
cross between the groups. It's the standard benchmark problem for QAOA.

Real-life uses: splitting a social or communication network, circuit-layout
partitioning, and scheduling two shifts so that conflicting jobs are separated.

QUBO (one qubit per node, x_i = side of node i):
    maximise  sum_{(i,j)} w_ij (x_i + x_j - 2 x_i x_j)    ->  minimise its negative
No constraints, so no penalty is needed: every bitstring is a valid cut.

Classical baselines: exhaustive search (exact, n <= ~22) and a greedy local
search (flip any node that improves the cut until none does).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from ..qubo import QUBO
from ..solvers.base import SolveResult


@dataclass
class MaxCutProblem:
    n: int
    edges: list[tuple[int, int, float]]
    names: list[str] | None = None

    @classmethod
    def from_networkx(cls, g, names=None):
        edges = [(int(u), int(v), float(d.get("weight", 1.0))) for u, v, d in g.edges(data=True)]
        return cls(g.number_of_nodes(), edges, names)

    @classmethod
    def random(cls, n: int, p: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        edges = [(i, j, 1.0) for i, j in itertools.combinations(range(n), 2) if rng.random() < p]
        return cls(n, edges)

    @property
    def total_weight(self) -> float:
        return float(sum(w for _, _, w in self.edges))

    def cut_value(self, bits) -> float:
        x = np.asarray(bits)
        return float(sum(w for i, j, w in self.edges if x[i] != x[j]))

    def to_qubo(self, penalty: float | None = None) -> QUBO:  # penalty unused: unconstrained
        q = QUBO(self.n, var_names=self.names or [f"node{i}" for i in range(self.n)])
        for i, j, w in self.edges:
            q.add_linear(i, -w)
            q.add_linear(j, -w)
            q.add_quadratic(i, j, 2 * w)
        return q

    def decode(self, bits):
        x = np.asarray(bits).astype(int)
        side_a = [self._name(i) for i in range(self.n) if x[i] == 0]
        side_b = [self._name(i) for i in range(self.n) if x[i] == 1]
        return (side_a, side_b), self.cut_value(x)

    def _name(self, i):
        return self.names[i] if self.names else i

    def brute_force(self) -> SolveResult:
        t0 = time.perf_counter()
        E = -self.to_qubo().all_energies()  # cut values
        E[2 ** (self.n - 1):] = -np.inf      # symmetry: fix the top node on side 0
        k = int(np.argmax(E))
        bits = QUBO.index_to_bits(k, self.n)
        return SolveResult("classical_exact(enumeration)", bits, float(E[k]), time.perf_counter() - t0)

    def local_search(self, seed: int = 0, restarts: int = 1) -> SolveResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        W = np.zeros((self.n, self.n))
        for i, j, w in self.edges:
            W[i, j] += w
            W[j, i] += w
        best_x, best_v = None, -1.0
        for _ in range(restarts):
            x = rng.integers(0, 2, self.n)
            improved = True
            while improved:
                improved = False
                for i in range(self.n):
                    same = W[i] @ (x == x[i]) - W[i, i]
                    other = W[i] @ (x != x[i])
                    if same > other:          # flipping i gains (same - other)
                        x[i] ^= 1
                        improved = True
            v = self.cut_value(x)
            if v > best_v:
                best_x, best_v = x.copy(), v
        return SolveResult("classical_heuristic(local search)", best_x, best_v, time.perf_counter() - t0)

    def approximation_ratio(self, bits) -> float:
        return self.cut_value(bits) / self.brute_force().energy
