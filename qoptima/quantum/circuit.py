"""Gate-level QAOA circuits: the actual instructions a quantum computer would run.

The fast simulator in `solvers/qaoa.py` applies whole layers at once. This module
writes out the same algorithm gate by gate, the way hardware sees it:

    H on every qubit                       (or RY(theta) for a warm start)
    repeat p times:
        RZZ(2 gamma' J_ij)  for each coupling  -> compiled as CNOT . RZ . CNOT
        RZ(2 gamma' h_i)    for each field
        RX(2 beta)          on every qubit      (warm start: RY(-t) RZ(2 beta) RY(t))
    measure all

where (h, J) are the Ising coefficients of the QUBO and gamma' = gamma / energy_scale.
It can count gates and depth, simulate gate by gate (to cross-check the fast
simulator), export OpenQASM 2.0 for real devices, and estimate SWAP overhead on
hardware where only neighbouring qubits can interact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..qubo import QUBO

SQRT_HALF = 1 / np.sqrt(2)


@dataclass(frozen=True)
class Gate:
    name: str                 # h, x, rx, ry, rz, rzz, cx
    qubits: tuple[int, ...]
    param: float | None = None

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2


@dataclass
class Circuit:
    n: int
    gates: list[Gate] = field(default_factory=list)

    def add(self, name, *qubits, param=None):
        self.gates.append(Gate(name, tuple(qubits), param))

    # ---- structure ---------------------------------------------------------
    def decompose(self) -> "Circuit":
        """Replace RZZ by the standard CNOT . RZ . CNOT (the hardware-native form)."""
        out = Circuit(self.n)
        for g in self.gates:
            if g.name == "rzz":
                i, j = g.qubits
                out.add("cx", i, j)
                out.add("rz", j, param=g.param)
                out.add("cx", i, j)
            else:
                out.gates.append(g)
        return out

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for g in self.gates:
            c[g.name] = c.get(g.name, 0) + 1
        return c

    def depth(self) -> int:
        """Number of time steps if every gate takes one step and gates on
        different qubits run in parallel."""
        level = [0] * self.n
        for g in self.gates:
            t = max(level[q] for q in g.qubits) + 1
            for q in g.qubits:
                level[q] = t
        return max(level) if self.n else 0

    def layers(self) -> list[list[Gate]]:
        level = [0] * self.n
        out: list[list[Gate]] = []
        for g in self.gates:
            t = max(level[q] for q in g.qubits)
            for q in g.qubits:
                level[q] = t + 1
            while len(out) <= t:
                out.append([])
            out[t].append(g)
        return out

    def summary(self) -> dict:
        d = self.decompose()
        c = d.counts()
        two = sum(v for k, v in c.items() if k == "cx")
        return {"qubits": self.n, "gates": len(d.gates), "two_qubit_gates": two,
                "one_qubit_gates": len(d.gates) - two, "depth": d.depth(), "counts": c}

    # ---- simulation ---------------------------------------------------------
    def simulate(self) -> np.ndarray:
        """Gate-by-gate statevector simulation starting from |0...0>."""
        psi = np.zeros(2 ** self.n, dtype=np.complex128)
        psi[0] = 1.0
        for g in self.gates:
            if g.name == "cx":
                psi = _apply_cx(psi, self.n, *g.qubits)
            elif g.name == "rzz":
                i, j = g.qubits
                k = np.arange(2 ** self.n)
                parity = ((k >> i) ^ (k >> j)) & 1           # 0 -> ZZ=+1, 1 -> ZZ=-1
                psi = psi * np.exp(-0.5j * g.param * (1 - 2 * parity))
            else:
                psi = _apply_1q(psi, self.n, g.qubits[0], _matrix(g))
        return psi

    # ---- export ---------------------------------------------------------------
    def to_qasm(self) -> str:
        """OpenQASM 2.0 using only h, rx, ry, rz, cx (accepted by IBM, Braket, etc.)."""
        d = self.decompose()
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{self.n}];", f"creg c[{self.n}];"]
        for g in d.gates:
            q = ",".join(f"q[{i}]" for i in g.qubits)
            lines.append(f"{g.name}({g.param:.10f}) {q};" if g.param is not None else f"{g.name} {q};")
        lines.append("measure q -> c;")
        return "\n".join(lines) + "\n"

    # ---- hardware connectivity -------------------------------------------------
    def interaction_pairs(self) -> set[tuple[int, int]]:
        return {tuple(sorted(g.qubits)) for g in self.gates if g.is_two_qubit and g.name != "cx"} | \
               {tuple(sorted(g.qubits)) for g in self.gates if g.name == "cx"}

    def swap_estimate(self, connectivity: str = "line") -> int:
        """Rough SWAP count to make every interaction nearest-neighbour.

        'all'  : all-to-all connectivity (trapped ions): 0 SWAPs.
        'line' : qubits on a line, identity layout. Each RZZ between qubits i and j
                 needs |i-j|-1 SWAPs there and the same back, a pessimistic upper
                 estimate (a real compiler re-uses moves and picks a better layout).
        """
        if connectivity == "all":
            return 0
        swaps = 0
        for g in self.gates:
            if g.name == "rzz":
                i, j = g.qubits
                swaps += 2 * max(0, abs(i - j) - 1)
        return swaps


def _matrix(g: Gate) -> np.ndarray:
    t = g.param
    if g.name == "h":
        return np.array([[SQRT_HALF, SQRT_HALF], [SQRT_HALF, -SQRT_HALF]], dtype=complex)
    if g.name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if g.name == "rx":
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])
    if g.name == "ry":
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if g.name == "rz":
        return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
    raise ValueError(f"unknown gate {g.name}")


def _apply_1q(psi, n, q, m):
    v = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
    a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
    v[:, 0, :] = m[0, 0] * a + m[0, 1] * b
    v[:, 1, :] = m[1, 0] * a + m[1, 1] * b
    return psi


def _apply_cx(psi, n, control, target):
    k = np.arange(2 ** n)
    on = ((k >> control) & 1) == 1
    out = psi.copy()
    out[k[on]] = psi[k[on] ^ (1 << target)]
    return out


def qaoa_circuit(qubo: QUBO, gammas, betas, energy_scale: float = 1.0,
                 thetas=None) -> Circuit:
    """Build the gate-level QAOA circuit matching `solve_qaoa`'s trained angles.

    Pass `energy_scale` and `thetas` from `result.info` so the circuit reproduces
    the simulator's state exactly (up to a global phase)."""
    h, J, _ = qubo.to_ising()
    n = qubo.n
    c = Circuit(n)
    for q in range(n):
        if thetas is None:
            c.add("h", q)
        else:
            c.add("ry", q, param=float(thetas[q]))
    for g, b in zip(gammas, betas):
        gs = g / energy_scale
        for (i, j), Jij in sorted(J.items()):
            c.add("rzz", i, j, param=float(2 * gs * Jij))
        for q in range(n):
            if h[q] != 0:
                c.add("rz", q, param=float(2 * gs * h[q]))
        for q in range(n):
            if thetas is None:
                c.add("rx", q, param=float(2 * b))
            else:
                c.add("ry", q, param=float(-thetas[q]))
                c.add("rz", q, param=float(2 * b))
                c.add("ry", q, param=float(thetas[q]))
    return c


def circuit_from_result(qubo: QUBO, result) -> Circuit:
    info = result.info
    return qaoa_circuit(qubo, info["gammas"], info["betas"], info["energy_scale"], info.get("thetas"))
