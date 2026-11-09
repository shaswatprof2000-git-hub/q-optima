"""A deliberately simple model of hardware noise for QAOA sampling.

This is an *approximation* for intuition and planning, not a device simulation:

1. Gate errors: every gate succeeds with probability (1 - e). The whole circuit works
   with fidelity F = (1 - e1)^(single-qubit gates) x (1 - e2)^(two-qubit gates).
   With probability 1 - F the output is replaced by a uniformly random bitstring
   (a global depolarising channel), so  P_noisy = F * P_ideal + (1 - F) / 2^n.
2. Readout errors: each measured bit flips independently with probability r.

It ignores qubit connectivity: on real chips, extra SWAP gates are needed when
interacting qubits aren't neighbours, so real fidelity is usually *lower*. It also
ignores coherent errors and crosstalk. Use the IBM hardware path for real numbers.
Typical published figures for current superconducting devices are around 1e-4 to 1e-3
for single-qubit gates, 1e-3 to 1e-2 for two-qubit gates and around 1e-2 for readout.
Those are the defaults' order of magnitude, not a specific device.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    error_1q: float = 5e-4
    error_2q: float = 5e-3
    readout: float = 1e-2
    cnots_per_zz: int = 2   # an RZZ gate is usually compiled into 2 CNOTs
    fidelity_override: float | None = None  # set by quantum.hardware (adds SWAPs, decoherence)

    def gate_counts(self, qubo, p: int, warm: bool = False) -> tuple[int, int]:
        n = qubo.n
        h, J, _ = qubo.to_ising()
        n_zz = len(J)
        n_z = int(np.count_nonzero(h))
        one_q = n                                    # initial H (or RY) layer
        one_q += p * (n_z + n * (3 if warm else 1))  # RZ cost terms + mixer (RY.RZ.RY if warm)
        two_q = p * n_zz * self.cnots_per_zz
        return one_q, two_q

    def circuit_fidelity(self, qubo, p: int, warm: bool = False) -> float:
        if self.fidelity_override is not None:
            return float(self.fidelity_override)
        one_q, two_q = self.gate_counts(qubo, p, warm)
        return float((1 - self.error_1q) ** one_q * (1 - self.error_2q) ** two_q)

    def sample(self, probs: np.ndarray, n: int, shots: int, rng, fidelity: float) -> np.ndarray:
        noisy = fidelity * probs + (1 - fidelity) / len(probs)
        noisy /= noisy.sum()
        samples = rng.choice(len(noisy), size=shots, p=noisy)
        if self.readout > 0:
            flips = rng.random((shots, n)) < self.readout
            masks = (flips * (1 << np.arange(n))).sum(axis=1)
            samples = samples ^ masks
        return samples

    def describe(self) -> dict:
        return {"error_1q": self.error_1q, "error_2q": self.error_2q,
                "readout": self.readout, "cnots_per_zz": self.cnots_per_zz,
                "fidelity_override": self.fidelity_override}
