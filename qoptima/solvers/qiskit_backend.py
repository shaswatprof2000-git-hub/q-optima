"""Run a trained QAOA circuit with Qiskit: local simulator or real IBM Quantum hardware.

Optional. Needs:  pip install "qoptima[qiskit]"   (qiskit, qiskit-ibm-runtime)

Workflow ("parameter transfer"): angles are optimised cheaply on the NumPy
simulator (`solve_qaoa`), then the same circuit is sampled on hardware. This
keeps paid/limited QPU time to a single sampling job.

NOTE: written against Qiskit 1.x/2.x primitives (SamplerV2). It was not run in
the environment where this repo was generated, so treat the first run as a test.
"""
from __future__ import annotations

import numpy as np

from ..qubo import QUBO
from .base import SolveResult


def build_qaoa_circuit(qubo: QUBO, gammas, betas, energy_scale: float = 1.0):
    from qiskit import QuantumCircuit

    n = qubo.n
    h, J, _ = qubo.to_ising()
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        g_hw = g / energy_scale
        for (i, j), Jij in J.items():
            qc.rzz(2 * g_hw * Jij, i, j)
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * g_hw * h[i], i)
        qc.rx(2 * b, range(n))
    qc.measure_all()
    return qc


def sample_circuit(qubo: QUBO, qaoa_result: SolveResult, shots: int = 2000,
                   backend_name: str | None = None, use_ibm: bool = False) -> SolveResult:
    """Sample the QAOA circuit. use_ibm=True sends it to IBM Quantum
    (requires a saved account: QiskitRuntimeService.save_account(...))."""
    import time

    info = qaoa_result.info
    qc = build_qaoa_circuit(qubo, info["gammas"], info["betas"], info["energy_scale"])
    t0 = time.perf_counter()

    if use_ibm:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService()
        backend = (service.backend(backend_name) if backend_name
                   else service.least_busy(operational=True, simulator=False))
        isa = generate_preset_pass_manager(backend=backend, optimization_level=1).run(qc)
        job = SamplerV2(mode=backend).run([isa], shots=shots)
        counts = job.result()[0].data.meas.get_counts()
        label = f"qaoa_ibm[{backend.name}]"
    else:
        from qiskit.primitives import StatevectorSampler

        job = StatevectorSampler().run([qc], shots=shots)
        counts = job.result()[0].data.meas.get_counts()
        label = "qaoa_qiskit_sim"

    # Qiskit bitstrings: rightmost character = qubit 0, same as our index convention
    best_k, best_e = None, np.inf
    for bitstring in counts:
        k = int(bitstring, 2)
        e = qubo.energy(QUBO.index_to_bits(k, qubo.n))
        if e < best_e:
            best_k, best_e = k, e
    return SolveResult(label, QUBO.index_to_bits(best_k, qubo.n), float(best_e),
                       time.perf_counter() - t0,
                       {"shots": shots, "distinct_bitstrings": len(counts),
                        "depth": qc.depth()})
