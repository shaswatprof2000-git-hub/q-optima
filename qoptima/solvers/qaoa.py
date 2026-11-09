"""QAOA (Quantum Approximate Optimization Algorithm) on an exact statevector simulator.

Pure NumPy, so it runs on any laptop without Qiskit. Practical up to ~20 qubits.

Circuit (p layers), starting from |+>^n:
    U(gamma, beta) = prod_l  exp(-i beta_l sum_j X_j) exp(-i gamma_l C)
where C is the QUBO cost, normalised to zero mean / unit std so the same angle
ranges work for every problem. The optimised angles can be replayed on real
hardware with `qoptima.solvers.qiskit_backend` (the scale factor is returned).

Like on a real device, the answer is read out by *sampling* shots and keeping
the lowest-energy bitstring seen, not by peeking at the full statevector.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from ..qubo import QUBO
from .base import SolveResult

MAX_QUBITS = 22


def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    c, s = np.cos(beta), -1j * np.sin(beta)
    for j in range(n):
        v = psi.reshape(2 ** (n - j - 1), 2, 2 ** j)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = c * a + s * b
        v[:, 1, :] = s * a + c * b
    return psi


def qaoa_state(cost: np.ndarray, gammas, betas, n: int) -> np.ndarray:
    psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=np.complex128)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * cost)
        psi = _apply_mixer(psi, b, n)
    return psi


def solve_qaoa(qubo: QUBO, p: int = 2, restarts: int = 4, shots: int = 1024,
               maxiter: int = 250, seed: int | None = 0) -> SolveResult:
    n = qubo.n
    if n > MAX_QUBITS:
        raise ValueError(f"{n} qubits exceeds the simulator limit of {MAX_QUBITS}")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    energies = qubo.all_energies()
    mean, std = energies.mean(), energies.std() or 1.0
    cost = (energies - mean) / std

    def expectation(params):
        psi = qaoa_state(cost, params[:p], params[p:], n)
        return float(np.real(np.vdot(psi, cost * psi)))

    starts = [np.concatenate([np.linspace(0.1, 0.8, p),   # "linear ramp" start,
                              np.linspace(0.8, 0.1, p)])]  # like a slow anneal
    starts += [np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
               for _ in range(restarts - 1)]

    best = None
    evals = 0
    for x0 in starts:
        r = minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter})
        evals += r.nfev
        if best is None or r.fun < best.fun:
            best = r

    gammas, betas = best.x[:p], best.x[p:]
    psi = qaoa_state(cost, gammas, betas, n)
    probs = np.abs(psi) ** 2
    probs /= probs.sum()

    samples = rng.choice(len(probs), size=shots, p=probs)
    uniq = np.unique(samples)
    k = int(uniq[np.argmin(energies[uniq])])
    bits = QUBO.index_to_bits(k, n)

    k_opt = int(np.argmin(energies))
    random_guess = 1 / 2 ** n
    info = {
        "p": p,
        "gammas": gammas,
        "betas": betas,
        "energy_scale": float(std),   # gamma_hardware = gamma / energy_scale
        "expected_energy": float(best.fun * std + mean),
        "prob_optimal": float(probs[k_opt]),
        "prob_optimal_random_guess": random_guess,
        "amplification": float(probs[k_opt] / random_guess),
        "shots": shots,
        "function_evals": evals,
        "qubits": n,
    }
    return SolveResult("qaoa", bits, float(energies[k]), time.perf_counter() - t0, info)
