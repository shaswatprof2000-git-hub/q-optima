"""QAOA (Quantum Approximate Optimization Algorithm) on an exact statevector simulator.

Pure NumPy, so it runs on any laptop or Colab without Qiskit. Practical up to ~20 qubits.

Circuit (p layers):
    |psi> = prod_l  M(beta_l) exp(-i gamma_l C) |psi_0>
C is the QUBO cost, normalised to zero mean / unit std so one angle range fits every
problem. The trained angles can be replayed on real hardware with
`qoptima.solvers.qiskit_backend` (the scale factor is returned in `info`).

Variants (all optional, defaults reproduce standard QAOA):

* **Warm start** (Egger, Marecek & Woerner, Quantum 5, 479, 2021): start from a good
  classical answer c (e.g. simulated annealing). Each qubit starts in
  cos(theta/2)|0> + sin(theta/2)|1> with sin^2(theta/2) = c_i regularised by `warm_eps`,
  and the mixer rotates around that same axis, exp(-i beta (sin theta X + cos theta Z)).
  With theta = pi/2 this is exactly standard QAOA (|+> start, RX mixer).
* **CVaR objective** (Barkoutsos et al., Quantum 4, 256, 2020): train on the average
  energy of the best `alpha` fraction of outcomes instead of the plain expectation.
  That fits the goal better, because we only keep the best sample anyway.
* **Layer-wise training** (INTERP, Zhou et al., PRX 10, 021067, 2020): train p=1, then
  grow p one layer at a time by interpolating the previous optimum. That gives much
  better starting points for deep circuits than random restarts.
* **Noise** (`NoiseModel`): a simple, clearly approximate model of hardware errors,
  applied when sampling. See noise.py.

Like on a real device, the answer is read out by *sampling* shots and keeping the
lowest-energy bitstring seen, not by peeking at the full statevector.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from ..qubo import QUBO
from .base import SolveResult
from .noise import NoiseModel

MAX_QUBITS = 22


# ---------------------------------------------------------------------------
# Statevector primitives
# ---------------------------------------------------------------------------
def _initial_state(n: int, thetas: np.ndarray | None) -> np.ndarray:
    if thetas is None:
        return np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=np.complex128)
    psi = np.ones(1, dtype=np.complex128)
    for j in range(n - 1, -1, -1):  # qubit n-1 is the most significant bit
        amp = np.array([np.cos(thetas[j] / 2), np.sin(thetas[j] / 2)])
        psi = np.kron(psi, amp)
    return psi


def _apply_mixer(psi: np.ndarray, beta: float, n: int, thetas: np.ndarray | None = None) -> np.ndarray:
    """Apply prod_j exp(-i beta n_j.sigma) with n_j = (sin theta_j, 0, cos theta_j).
    thetas=None means theta = pi/2 for all qubits, i.e. the standard RX(2 beta) mixer."""
    c, s = np.cos(beta), np.sin(beta)
    for j in range(n):
        if thetas is None:
            m00 = m11 = c
            m01 = m10 = -1j * s
        else:
            st, ct = np.sin(thetas[j]), np.cos(thetas[j])
            m00, m11 = c - 1j * s * ct, c + 1j * s * ct
            m01 = m10 = -1j * s * st
        v = psi.reshape(2 ** (n - j - 1), 2, 2 ** j)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = m00 * a + m01 * b
        v[:, 1, :] = m10 * a + m11 * b
    return psi


def qaoa_state(cost: np.ndarray, gammas, betas, n: int, thetas: np.ndarray | None = None) -> np.ndarray:
    psi = _initial_state(n, thetas)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * cost)
        psi = _apply_mixer(psi, b, n, thetas)
    return psi


def warm_start_thetas(bits, eps: float = 0.25) -> np.ndarray:
    """Angles for a warm start from a 0/1 solution, regularised so no qubit is frozen."""
    c = np.clip(np.asarray(bits, dtype=float), eps, 1 - eps)
    return 2 * np.arcsin(np.sqrt(c))


def cvar(probs: np.ndarray, cost_sorted_idx: np.ndarray, cost: np.ndarray, alpha: float) -> float:
    """Average cost of the best `alpha` fraction of the probability mass."""
    p = probs[cost_sorted_idx]
    cum = np.cumsum(p)
    k = int(np.searchsorted(cum, alpha))
    k = min(k, len(p) - 1)
    head = cost[cost_sorted_idx[:k]] @ p[:k]
    remaining = alpha - (cum[k - 1] if k > 0 else 0.0)
    return float((head + remaining * cost[cost_sorted_idx[k]]) / alpha)


def interp_next(params: np.ndarray, p: int) -> np.ndarray:
    """INTERP: optimal (gamma, beta) at depth p -> starting guess at depth p+1."""
    out = []
    for vec in (params[:p], params[p:]):
        ext = np.concatenate([[0.0], vec, [0.0]])
        new = [((i - 1) / p) * ext[i - 1] + ((p - i + 1) / p) * ext[i] for i in range(1, p + 2)]
        out.append(new)
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve_qaoa(qubo: QUBO, p: int = 2, restarts: int = 4, shots: int = 1024,
               maxiter: int = 250, seed: int | None = 0, *,
               objective: str = "expectation", alpha: float = 0.1,
               warm_start=None, warm_eps: float = 0.25,
               layerwise: bool = False, noise: NoiseModel | None = None) -> SolveResult:
    """Run QAOA and return the best sampled bitstring.

    objective   "expectation" (standard) or "cvar" (CVaR_alpha)
    warm_start  optional 0/1 array (e.g. annealing result) for warm-start QAOA
    layerwise   train p = 1, 2, ..., p with INTERP initialisation (restarts apply to p = 1)
    noise       optional NoiseModel applied to the sampled shots (training stays noiseless)
    """
    n = qubo.n
    if n > MAX_QUBITS:
        raise ValueError(f"{n} qubits exceeds the simulator limit of {MAX_QUBITS}")
    if objective not in ("expectation", "cvar"):
        raise ValueError("objective must be 'expectation' or 'cvar'")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    energies = qubo.all_energies()
    mean, std = energies.mean(), energies.std() or 1.0
    cost = (energies - mean) / std
    order = np.argsort(cost)
    thetas = None if warm_start is None else warm_start_thetas(warm_start, warm_eps)

    evals = 0

    def make_loss(depth):
        def loss(params):
            nonlocal evals
            evals += 1
            psi = qaoa_state(cost, params[:depth], params[depth:], n, thetas)
            probs = np.abs(psi) ** 2
            if objective == "cvar":
                return cvar(probs, order, cost, alpha)
            return float(probs @ cost)
        return loss

    def train(depth, starts):
        best = None
        for x0 in starts:
            r = minimize(make_loss(depth), x0, method="COBYLA", options={"maxiter": maxiter})
            if best is None or r.fun < best.fun:
                best = r
        return best

    def ramp(depth):
        return np.concatenate([np.linspace(0.1, 0.8, depth), np.linspace(0.8, 0.1, depth)])

    def random_start(depth):
        return np.concatenate([rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / 2, depth)])

    history = []
    if layerwise:
        best = train(1, [ramp(1)] + [random_start(1) for _ in range(restarts - 1)])
        history.append((1, float(best.fun)))
        for depth in range(1, p):
            best = train(depth + 1, [interp_next(best.x, depth)])
            history.append((depth + 1, float(best.fun)))
    else:
        best = train(p, [ramp(p)] + [random_start(p) for _ in range(restarts - 1)])
        history.append((p, float(best.fun)))

    gammas, betas = best.x[:p], best.x[p:]
    psi = qaoa_state(cost, gammas, betas, n, thetas)
    probs = np.abs(psi) ** 2
    probs /= probs.sum()

    fidelity = 1.0
    if noise is not None:
        fidelity = noise.circuit_fidelity(qubo, p, warm=thetas is not None)
        samples = noise.sample(probs, n, shots, rng, fidelity)
    else:
        samples = rng.choice(len(probs), size=shots, p=probs)

    uniq, counts = np.unique(samples, return_counts=True)
    k = int(uniq[np.argmin(energies[uniq])])
    bits = QUBO.index_to_bits(k, n)

    # Problems often have several equally good answers (a route and its reverse,
    # a cut and its mirror image), so "optimal" means any lowest-energy bitstring.
    e_min = energies.min()
    opt_mask = energies <= e_min + 1e-9 * max(1.0, abs(e_min))
    random_guess = float(opt_mask.sum()) / 2 ** n
    p_opt_ideal = float(probs[opt_mask].sum())
    p_opt_sampled = float(np.mean(opt_mask[samples]))
    info = {
        "p": p,
        "objective": objective,
        "alpha": alpha if objective == "cvar" else None,
        "warm_start": thetas is not None,
        "layerwise": layerwise,
        "gammas": gammas,
        "betas": betas,
        "thetas": thetas,
        "energy_scale": float(std),   # gamma_hardware = gamma / energy_scale
        "expected_energy": float(probs @ energies),
        "prob_optimal": p_opt_ideal,
        "prob_optimal_random_guess": random_guess,
        "num_optimal": int(opt_mask.sum()),
        "amplification": float(p_opt_ideal / random_guess),
        "sampled_prob_optimal": p_opt_sampled,
        "noise": None if noise is None else noise.describe(),
        "circuit_fidelity": fidelity,
        "shots": shots,
        "distinct_samples": int(len(uniq)),
        "function_evals": evals,
        "training_history": history,
        "qubits": n,
    }
    label = "qaoa" + ("_ws" if thetas is not None else "") + ("_cvar" if objective == "cvar" else "") \
        + ("_noisy" if noise is not None else "")
    return SolveResult(label, bits, float(energies[k]), time.perf_counter() - t0, info)
