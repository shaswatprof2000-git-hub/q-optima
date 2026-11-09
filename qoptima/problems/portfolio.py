"""Portfolio selection: pick exactly k of N stocks, balancing return against risk.

    minimise   q * x^T Sigma x  -  mu^T x      (mean-variance, equal weights)
    s.t.       sum x = k                     (penalty)

One qubit per stock. mu / Sigma come from daily prices (e.g. exported by
Portfolio X-Ray) or from the synthetic generator for demos.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..qubo import QUBO
from ..solvers.base import SolveResult

TRADING_DAYS = 252


@dataclass
class PortfolioProblem:
    tickers: list[str]
    mu: np.ndarray      # expected annual return per stock
    sigma: np.ndarray   # annual covariance
    k: int
    risk_aversion: float = 1.0

    @classmethod
    def from_prices(cls, prices: pd.DataFrame, k: int, risk_aversion: float = 1.0):
        """prices: DataFrame indexed by date, one column per ticker (close prices)."""
        r = np.log(prices / prices.shift(1)).dropna()
        return cls(list(prices.columns), r.mean().values * TRADING_DAYS,
                   r.cov().values * TRADING_DAYS, k, risk_aversion)

    @classmethod
    def from_csv(cls, path: str, k: int, risk_aversion: float = 1.0):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls.from_prices(df, k, risk_aversion)

    @classmethod
    def synthetic(cls, n: int, k: int, seed: int = 0, risk_aversion: float = 1.0):
        """Sector-structured fake market (clearly NOT real data) for demos and tests."""
        rng = np.random.default_rng(seed)
        sectors = ["BANK", "IT", "PHARMA", "AUTO", "FMCG", "ENERGY"]
        tickers = [f"{sectors[i % len(sectors)]}_{i // len(sectors) + 1}" for i in range(n)]
        sector_of = np.array([i % len(sectors) for i in range(n)])
        days = 2 * TRADING_DAYS
        market = rng.normal(0.0004, 0.010, days)
        sector_moves = rng.normal(0, 0.007, (days, len(sectors)))
        alpha = rng.normal(0.0003, 0.0004, n)
        beta = rng.uniform(0.6, 1.4, n)
        r = (alpha + market[:, None] * beta + sector_moves[:, sector_of]
             + rng.normal(0, 0.012, (days, n)))
        prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), columns=tickers,
                              index=pd.bdate_range("2024-01-01", periods=days))
        return cls.from_prices(prices, k, risk_aversion)

    @property
    def n(self) -> int:
        return len(self.tickers)

    def objective(self, x) -> float:
        x = np.asarray(x, dtype=float)
        w = x / self.k
        return float(self.risk_aversion * w @ self.sigma @ w - self.mu @ w)

    def to_qubo(self, penalty: float | None = None) -> QUBO:
        k, q = self.k, self.risk_aversion
        Qfull = q * self.sigma / k ** 2 - np.diag(self.mu) / k
        # Smallest penalty that kept the true optimum feasible in our tests; larger values
        # are "safer" but make the landscape harder for annealing and QAOA.
        # The benchmark's exhaustive check flags it if this ever goes wrong.
        A = penalty if penalty is not None else 1.0 * float(np.abs(Qfull).sum(axis=1).max())
        model = QUBO(self.n, var_names=list(self.tickers))
        for i in range(self.n):
            model.add_linear(i, Qfull[i, i])
            for j in range(i + 1, self.n):
                model.add_quadratic(i, j, Qfull[i, j] + Qfull[j, i])
        model.add_exactly_k(list(range(self.n)), k, A)
        return model

    def decode(self, bits):
        x = np.asarray(bits)
        if x.sum() != self.k:
            return None, None
        return [t for t, b in zip(self.tickers, x) if b], self.objective(x)

    def stats(self, bits) -> dict:
        x = np.asarray(bits, dtype=float)
        w = x / self.k
        ret = float(self.mu @ w)
        vol = float(np.sqrt(w @ self.sigma @ w))
        return {"expected_return": ret, "volatility": vol,
                "return_per_risk": ret / vol if vol else float("nan")}

    def brute_force(self) -> SolveResult:
        t0 = time.perf_counter()
        best, best_x = np.inf, None
        for combo in itertools.combinations(range(self.n), self.k):
            x = np.zeros(self.n, dtype=int)
            x[list(combo)] = 1
            v = self.objective(x)
            if v < best:
                best, best_x = v, x
        return SolveResult("classical_exact(enumeration)", best_x, best,
                           time.perf_counter() - t0)

    def greedy(self) -> SolveResult:
        """Pick the k stocks with the best stand-alone return/risk (common shortcut)."""
        t0 = time.perf_counter()
        score = self.mu / np.sqrt(np.diag(self.sigma))
        x = np.zeros(self.n, dtype=int)
        x[np.argsort(-score)[: self.k]] = 1
        return SolveResult("classical_heuristic(greedy)", x, self.objective(x),
                           time.perf_counter() - t0)
