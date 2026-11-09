"""Looking inside a quantum state: per-qubit Bloch vectors and entanglement.

A single qubit's state can be drawn as a point on the Bloch sphere:
    rho = (I + x X + y Y + z Z) / 2,   x = 2 Re(rho_01),  y = -2 Im(rho_01),  z = rho_00 - rho_11
|0> is the north pole (z=+1), |1> the south pole, |+> points along +x.
Inside a many-qubit state, a qubit that is entangled with others has |r| < 1:
it sits *inside* the sphere because its own state is mixed.

Entanglement entropy across a cut (qubits 0..m-1 vs the rest) comes from the
Schmidt (singular) values s_k of the reshaped state:  S = -sum s_k^2 log2 s_k^2.
S = 0 means no entanglement; the maximum is min(m, n-m) bits.
"""
from __future__ import annotations

import numpy as np


def reduced_density_matrix(psi: np.ndarray, n: int, q: int) -> np.ndarray:
    v = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
    a, b = v[:, 0, :], v[:, 1, :]
    r00 = float(np.sum(np.abs(a) ** 2))
    r11 = float(np.sum(np.abs(b) ** 2))
    r01 = complex(np.sum(a * np.conj(b)))
    return np.array([[r00, r01], [np.conj(r01), r11]])


def bloch_vector(psi: np.ndarray, n: int, q: int) -> np.ndarray:
    rho = reduced_density_matrix(psi, n, q)
    return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])


def bloch_vectors(psi: np.ndarray, n: int) -> np.ndarray:
    return np.array([bloch_vector(psi, n, q) for q in range(n)])


def entanglement_entropy(psi: np.ndarray, n: int, m: int | None = None) -> float:
    """Entropy (bits) between qubits 0..m-1 and m..n-1 (default: half/half)."""
    m = n // 2 if m is None else m
    s = np.linalg.svd(psi.reshape(2 ** (n - m), 2 ** m), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def marginal_one_probabilities(psi: np.ndarray, n: int) -> np.ndarray:
    """P(qubit q measured as 1) for every qubit."""
    probs = np.abs(psi) ** 2
    k = np.arange(len(probs))
    return np.array([probs[((k >> q) & 1) == 1].sum() for q in range(n)])
