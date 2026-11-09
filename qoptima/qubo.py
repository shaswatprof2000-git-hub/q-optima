"""QUBO model: minimise  E(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j + offset,  x in {0,1}^n.

Bit convention (used everywhere in Q-Optima, and matching Qiskit):
    bitstring index k  <->  x_j = (k >> j) & 1     (qubit 0 is the least significant bit)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_DENSE_VARS = 24  # 2**24 energies = 16.7M floats; above this use heuristics only


@dataclass
class QUBO:
    n: int
    Q: np.ndarray = field(default=None)  # upper-triangular (n, n)
    offset: float = 0.0
    var_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.zeros((self.n, self.n))
        if not self.var_names:
            self.var_names = [f"x{i}" for i in range(self.n)]

    # ---- building -------------------------------------------------------
    def add_linear(self, i: int, value: float) -> None:
        self.Q[i, i] += value

    def add_quadratic(self, i: int, j: int, value: float) -> None:
        if i == j:  # x_i * x_i = x_i
            self.Q[i, i] += value
            return
        a, b = (i, j) if i < j else (j, i)
        self.Q[a, b] += value

    def add_offset(self, value: float) -> None:
        self.offset += value

    def add_exactly_k(self, idx: list[int], k: int, penalty: float) -> None:
        """Add penalty * (sum_{i in idx} x_i - k)^2."""
        # (S - k)^2 = S^2 - 2kS + k^2 ; S^2 = sum x_i + 2 sum_{i<j} x_i x_j
        for a, i in enumerate(idx):
            self.add_linear(i, penalty * (1 - 2 * k))
            for j in idx[a + 1:]:
                self.add_quadratic(i, j, 2 * penalty)
        self.add_offset(penalty * k * k)

    def add_at_most_one(self, idx: list[int], penalty: float) -> None:
        """Add penalty * sum_{i<j} x_i x_j  (zero iff at most one variable is 1)."""
        for a, i in enumerate(idx):
            for j in idx[a + 1:]:
                self.add_quadratic(i, j, penalty)

    # ---- evaluating -----------------------------------------------------
    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def all_energies(self) -> np.ndarray:
        """Energy of every bitstring, index k as in the module docstring."""
        if self.n > MAX_DENSE_VARS:
            raise ValueError(f"{self.n} variables is too many for dense enumeration")
        k = np.arange(2 ** self.n, dtype=np.int64)
        X = ((k[:, None] >> np.arange(self.n)) & 1).astype(np.float64)
        return np.einsum("ki,ij,kj->k", X, self.Q, X) + self.offset

    @staticmethod
    def index_to_bits(k: int, n: int) -> np.ndarray:
        return np.array([(k >> j) & 1 for j in range(n)], dtype=int)

    # ---- Ising form (for quantum circuits) ----------------------------------
    def to_ising(self):
        """Return (h, J, const) with x_i = (1 - z_i)/2 so that
        E = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j,  z in {+1,-1}  (|0> -> z=+1)."""
        n = self.n
        h = np.zeros(n)
        J = {}
        const = self.offset
        for i in range(n):
            q = self.Q[i, i]
            const += q / 2
            h[i] -= q / 2
        for i in range(n):
            for j in range(i + 1, n):
                q = self.Q[i, j]
                if q == 0:
                    continue
                const += q / 4
                h[i] -= q / 4
                h[j] -= q / 4
                J[(i, j)] = q / 4
        return h, J, const

    @property
    def num_interactions(self) -> int:
        return int(np.count_nonzero(np.triu(self.Q, 1)))
