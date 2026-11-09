from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolveResult:
    solver: str
    bits: np.ndarray
    energy: float
    runtime_s: float
    info: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        b = "".join(str(int(v)) for v in self.bits)
        return f"SolveResult({self.solver}, energy={self.energy:.4f}, bits={b}, {self.runtime_s:.3f}s)"
