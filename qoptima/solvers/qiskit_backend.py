"""Run a trained QAOA circuit with Qiskit: local simulator or real IBM Quantum hardware.

Optional. Needs:  pip install "qoptima[qiskit]"   (qiskit, qiskit-ibm-runtime)

Workflow ("parameter transfer"): angles are optimised cheaply on the NumPy
simulator (`solve_qaoa`), then exactly the same gate-level circuit, exported as
OpenQASM 2.0 by `qoptima.quantum.circuit`, is sampled on hardware. That keeps
QPU time to a single sampling job, and warm-start circuits work too.

NOTE: written against Qiskit 1.x/2.x (QuantumCircuit.from_qasm_str, SamplerV2).
It has not been run on real hardware yet, so treat the first run as a test.
"""
from __future__ import annotations

from ..qubo import QUBO
from .base import SolveResult


def build_qiskit_circuit(qubo: QUBO, qaoa_result: SolveResult):
    from qiskit import QuantumCircuit

    from ..quantum.circuit import circuit_from_result
    return QuantumCircuit.from_qasm_str(circuit_from_result(qubo, qaoa_result).to_qasm())


def sample_circuit(qubo: QUBO, qaoa_result: SolveResult, shots: int = 2000,
                   backend_name: str | None = None, use_ibm: bool = False) -> SolveResult:
    """Sample the QAOA circuit. use_ibm=True sends it to IBM Quantum
    (requires a saved account: QiskitRuntimeService.save_account(...))."""
    import time

    qc = build_qiskit_circuit(qubo, qaoa_result)
    t0 = time.perf_counter()

    if use_ibm:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService()
        backend = (service.backend(backend_name) if backend_name
                   else service.least_busy(operational=True, simulator=False))
        isa = generate_preset_pass_manager(backend=backend, optimization_level=1).run(qc)
        job = SamplerV2(mode=backend).run([isa], shots=shots)
        counts = job.result()[0].data.c.get_counts()
        label = f"qaoa_ibm[{backend.name}]"
    else:
        from qiskit.primitives import StatevectorSampler

        job = StatevectorSampler().run([qc], shots=shots)
        counts = job.result()[0].data.c.get_counts()
        label = "qaoa_qiskit_sim"

    # Qiskit bitstrings: rightmost character = classical bit 0 = qubit 0 (our convention)
    E = qubo.all_energies()
    opt = E <= E.min() + 1e-9 * max(1.0, abs(E.min()))
    best_k, hits = None, 0
    for bitstring, c in counts.items():
        k = int(bitstring, 2)
        hits += c if opt[k] else 0
        if best_k is None or E[k] < E[best_k]:
            best_k = k
    return SolveResult(label, QUBO.index_to_bits(best_k, qubo.n), float(E[best_k]),
                       time.perf_counter() - t0,
                       {"shots": shots, "distinct_bitstrings": len(counts), "depth": qc.depth(),
                        "sampled_prob_optimal": hits / shots, "counts": counts})


__all__ = ["build_qiskit_circuit", "sample_circuit"]
