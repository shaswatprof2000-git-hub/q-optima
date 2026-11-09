"""Exhaustive search over all 2^n bitstrings. Ground truth for small problems."""
from __future__ import annotations

import time

import numpy as np

from ..qubo import QUBO
from .base import SolveResult


def solve_exact(qubo: QUBO) -> SolveResult:
    t0 = time.perf_counter()
    E = qubo.all_energies()
    k = int(np.argmin(E))
    bits = QUBO.index_to_bits(k, qubo.n)
    return SolveResult("exact", bits, float(E[k]), time.perf_counter() - t0,
                       {"states_checked": len(E)})
