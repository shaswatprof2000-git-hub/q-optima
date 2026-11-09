"""Quantum hardware profiles and a rough "will this circuit survive?" estimate.

The profiles hold ORDER-OF-MAGNITUDE figures typical of published devices in recent
years. They are illustrative, NOT the specification of any particular machine.
Real numbers vary by device, qubit and day. For real figures, read the calibration
data of the backend you use (e.g. IBM's `backend.properties()`).

The estimate follows the common "estimated success probability" idea:

    F  ~  (1 - e1)^N1  x  (1 - e2)^N2  x  exp(-n * T / T2)

N1 and N2 are one- and two-qubit gate counts (two-qubit gates include SWAPs, 3 CNOTs
each, when the chip can only couple neighbours). n is the qubit count and T is
circuit duration. It ignores crosstalk, leakage and coherent errors, so treat it
as an optimistic sketch.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..solvers.noise import NoiseModel
from .circuit import Circuit


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    technology: str
    error_1q: float
    error_2q: float
    readout: float
    t2_us: float            # coherence (dephasing) time, microseconds
    gate_1q_ns: float
    gate_2q_ns: float
    connectivity: str       # "line" (sparse, modelled pessimistically) or "all"
    note: str = ""


DEVICES = {
    "ideal": DeviceProfile("Ideal (noiseless)", "simulation", 0, 0, 0, math.inf, 0, 0, "all",
                           "What the statevector simulator shows."),
    "superconducting": DeviceProfile(
        "Superconducting (typical today)", "transmon qubits on a chip at ~10 mK",
        3e-4, 5e-3, 1.5e-2, 100.0, 35.0, 300.0, "line",
        "Fast gates, sparse nearest-neighbour coupling, so SWAPs are needed."),
    "trapped_ion": DeviceProfile(
        "Trapped ion (typical today)", "ions held by electric fields, gates by lasers",
        5e-5, 2e-3, 3e-3, 1e6, 10_000.0, 200_000.0, "all",
        "All-to-all coupling and long coherence, but much slower gates."),
    "future": DeviceProfile(
        "Improved hardware (illustrative target)", "any",
        1e-5, 1e-4, 1e-3, 1e4, 20.0, 100.0, "all",
        "Roughly 10-50x better than today; a planning scenario, not a product."),
}


def estimate(circuit: Circuit, device: DeviceProfile) -> dict:
    d = circuit.decompose()
    counts = d.counts()
    n2 = counts.get("cx", 0)
    swaps = circuit.swap_estimate(device.connectivity)
    n2_total = n2 + 3 * swaps
    n1 = len(d.gates) - n2
    depth = d.depth() + 3 * swaps  # pessimistic: SWAPs serialise
    duration_us = (d.depth() * device.gate_1q_ns + (n2_total) * device.gate_2q_ns / max(1, circuit.n // 2)) / 1000
    gate_f = (1 - device.error_1q) ** n1 * (1 - device.error_2q) ** n2_total
    coh_f = math.exp(-circuit.n * duration_us / device.t2_us) if math.isfinite(device.t2_us) else 1.0
    return {
        "device": device.name, "qubits": circuit.n, "one_qubit_gates": n1,
        "two_qubit_gates": n2, "swaps": swaps, "two_qubit_incl_swaps": n2_total,
        "depth_est": depth, "duration_us": round(duration_us, 2),
        "gate_fidelity": gate_f, "coherence_factor": coh_f,
        "fidelity": gate_f * coh_f, "readout_error": device.readout,
    }


def noise_model(circuit: Circuit, device: DeviceProfile) -> NoiseModel:
    """A NoiseModel whose fidelity includes SWAP overhead and decoherence."""
    e = estimate(circuit, device)
    return NoiseModel(device.error_1q, device.error_2q, device.readout,
                      fidelity_override=e["fidelity"])


def profile_table() -> list[dict]:
    return [asdict(v) | {"key": k} for k, v in DEVICES.items()]
