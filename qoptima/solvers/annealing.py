"""Simulated annealing on a QUBO (classical, physics-inspired).

Single-bit-flip Metropolis with a geometric temperature schedule and several
independent restarts ("reads"), the same idea D-Wave's open-source sampler uses.
"""
from __future__ import annotations

import time

import numpy as np

from ..qubo import QUBO
from .base import SolveResult


def solve_annealing(qubo: QUBO, num_reads: int = 20, sweeps: int = 400,
                    t_start: float | None = None, t_end: float | None = None,
                    seed: int | None = 0) -> SolveResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = qubo.n
    diag = np.diag(qubo.Q).copy()
    off = qubo.Q + qubo.Q.T
    np.fill_diagonal(off, 0.0)

    # Temperature range from the typical size of a single flip
    scale = np.abs(diag) + np.abs(off).sum(axis=1)
    scale = scale[scale > 0]
    typical = float(np.median(scale)) if len(scale) else 1.0
    t_start = t_start or 2.0 * typical
    t_end = t_end or 0.001 * typical
    temps = np.geomspace(t_start, t_end, sweeps)

    best_bits, best_e = None, np.inf
    energies = []
    for _ in range(num_reads):
        x = rng.integers(0, 2, n).astype(float)
        field = off @ x  # sum_j off_ij x_j
        e = qubo.energy(x)
        for T in temps:
            for i in rng.permutation(n):
                delta = (1 - 2 * x[i]) * (diag[i] + field[i])
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    d = 1 - 2 * x[i]
                    x[i] += d
                    field += d * off[:, i]
                    e += delta
        energies.append(e)
        if e < best_e:
            best_e, best_bits = e, x.astype(int).copy()

    return SolveResult("simulated_annealing", best_bits, float(qubo.energy(best_bits)),
                       time.perf_counter() - t0,
                       {"num_reads": num_reads, "sweeps": sweeps,
                        "read_energies": np.array(energies)})
