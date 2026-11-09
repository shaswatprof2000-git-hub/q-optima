"""Raj Sundari plug-in: match buyers to available plots by preference fit.

Each buyer gets exactly one plot; each plot goes to at most one buyer.
Fit score (0-100) combines budget, size, facing, road width and corner preference.

QUBO: x[b,p] = 1 means "show / allocate plot p to buyer b".  B*P qubits.
Classical baseline: the Hungarian algorithm (exact, polynomial time).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..qubo import QUBO
from ..solvers.base import SolveResult


@dataclass
class Plot:
    plot_id: str
    area_sqft: float
    price_lakh: float
    facing: str          # "East", "North", ...
    road_width_ft: float
    corner: bool = False


@dataclass
class Buyer:
    name: str
    budget_lakh: float
    min_area_sqft: float
    preferred_facing: tuple[str, ...] = ("East", "North")
    min_road_ft: float = 20
    wants_corner: bool = False


def fit_score(b: Buyer, p: Plot) -> float:
    score = 0.0
    # budget (40): full marks within budget, falls off quickly above it
    over = (p.price_lakh - b.budget_lakh) / b.budget_lakh
    score += 40 if over <= 0 else max(0.0, 40 * (1 - over / 0.15))
    # size (25)
    short = (b.min_area_sqft - p.area_sqft) / b.min_area_sqft
    score += 25 if short <= 0 else max(0.0, 25 * (1 - short / 0.2))
    # facing (15), road (12), corner (8)
    score += 15 if p.facing in b.preferred_facing else 0
    score += 12 if p.road_width_ft >= b.min_road_ft else 6 * p.road_width_ft / b.min_road_ft
    score += 8 if (p.corner == b.wants_corner or (p.corner and not b.wants_corner)) else 0
    return round(score, 2)


@dataclass
class PlotMatchingProblem:
    buyers: list[Buyer]
    plots: list[Plot]

    def __post_init__(self):
        if len(self.buyers) > len(self.plots):
            raise ValueError("need at least as many plots as buyers")
        self.score = np.array([[fit_score(b, p) for p in self.plots] for b in self.buyers])

    @classmethod
    def sample(cls, n_buyers: int = 3, n_plots: int = 4, seed: int = 0):
        """Illustrative sample data (not real inventory or prices)."""
        rng = np.random.default_rng(seed)
        facings = ["East", "North", "West", "South"]
        plots = [Plot(f"P-{101 + i}", float(rng.choice([1200, 1500, 1800, 2400])),
                      0.0, str(rng.choice(facings)), float(rng.choice([20, 30, 40])),
                      bool(rng.random() < 0.25)) for i in range(n_plots)]
        for p in plots:
            p.price_lakh = round(p.area_sqft * float(rng.uniform(2200, 2800)) / 1e5, 1)
        buyers = [Buyer(f"Buyer-{chr(65 + i)}", round(float(rng.uniform(28, 60)), 1),
                        float(rng.choice([1200, 1500, 1800])),
                        tuple(rng.choice(facings, 2, replace=False)),
                        float(rng.choice([20, 30])), bool(rng.random() < 0.3))
                  for i in range(n_buyers)]
        return cls(buyers, plots)

    @property
    def n(self) -> int:
        return len(self.buyers) * len(self.plots)

    def var(self, b: int, p: int) -> int:
        return b * len(self.plots) + p

    def to_qubo(self, penalty: float | None = None) -> QUBO:
        B, P = len(self.buyers), len(self.plots)
        A = penalty if penalty is not None else 2.0 * float(self.score.max())
        names = [f"{self.buyers[b].name}->{self.plots[p].plot_id}" for b in range(B) for p in range(P)]
        q = QUBO(B * P, var_names=names)
        for b in range(B):
            for p in range(P):
                q.add_linear(self.var(b, p), -self.score[b, p])
            q.add_exactly_k([self.var(b, p) for p in range(P)], 1, A)
        for p in range(P):
            q.add_at_most_one([self.var(b, p) for b in range(B)], A)
        return q

    def decode(self, bits):
        B, P = len(self.buyers), len(self.plots)
        X = np.asarray(bits).reshape(B, P)
        if not (np.all(X.sum(1) == 1) and np.all(X.sum(0) <= 1)):
            return None, None
        pairs = [(self.buyers[b].name, self.plots[int(np.argmax(X[b]))].plot_id) for b in range(B)]
        total = float((X * self.score).sum())
        return pairs, total

    def hungarian(self) -> SolveResult:
        t0 = time.perf_counter()
        rows, cols = linear_sum_assignment(-self.score)
        X = np.zeros_like(self.score, dtype=int)
        X[rows, cols] = 1
        return SolveResult("classical_exact(hungarian)", X.ravel(), float(self.score[rows, cols].sum()),
                           time.perf_counter() - t0)
