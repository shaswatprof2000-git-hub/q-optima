# Q-Optima: notes for Claude Code

Hybrid quantum-classical optimisation library. Real problems are turned into QUBOs and solved by
QAOA (simulated, optionally IBM hardware), simulated annealing and classical exact/heuristic
methods. Every result is compared against a proven optimum.

## Commands

```bash
pip install -e ".[dev]"            # first time (add ,qiskit for IBM hardware)
pytest -q                          # 11 tests, ~2 s; run after every change
python -m qoptima benchmark        # ~40 s; regenerates results/benchmark.md + .csv
python -m qoptima benchmark --quick
python -m qoptima route --places examples/places.csv
python -m qoptima portfolio --prices examples/prices_synthetic.csv --k 3
python -m qoptima plots --buyers examples/buyers.csv --plots examples/plots.csv
```

## Layout

- `qoptima/qubo.py`: `QUBO` model (upper-triangular Q + offset), penalty helpers
  (`add_exactly_k`, `add_at_most_one`), `all_energies()`, `to_ising()`
- `qoptima/solvers/`: `exact.py` (brute force, ≤24 vars), `annealing.py`, `qaoa.py`
  (NumPy statevector, ≤22 qubits), `qiskit_backend.py` (optional; replays trained angles on
  Qiskit/IBM, still untested on real hardware)
- `qoptima/problems/`: plug-ins `routing.py`, `portfolio.py`, `plot_matching.py`
- `qoptima/benchmark.py`, `qoptima/cli.py`

## Conventions (keep these)

- **Bit order:** bitstring index `k` ↔ `x_j = (k >> j) & 1`, so qubit 0 is the least significant bit. This matches Qiskit's counts strings, where the rightmost character is qubit 0.
- **Every problem plug-in provides:** `to_qubo(penalty=None)`, `decode(bits) -> (solution, value) | (None, None)` for infeasible bits, and at least one classical baseline returning `SolveResult`. The first baseline must be **exact**, because the benchmark uses it as the optimum.
- **Penalty weights:** use the smallest value that keeps the QUBO's ground state feasible. Add a test asserting `solve_exact(qubo)` decodes to the classical optimum.
- **Honesty rule:** never claim quantum speed-up or advantage in code comments, README or results. Report gap to optimum, feasibility, runtime and QAOA amplification (P(optimal) vs 1/2ⁿ).
- Sample data (buyers, plots, prices) is illustrative. Label anything synthetic.
- Pure NumPy/SciPy/pandas in the core. Qiskit stays an optional extra, imported lazily.

## Adding a new problem

1. Create `qoptima/problems/<name>.py` with a dataclass like the existing plug-ins.
2. Export it in `qoptima/problems/__init__.py`.
3. Add instances in `benchmark.default_instances()` and a branch in `benchmark._classical()`.
4. Add tests: the QUBO optimum equals the classical exact optimum, and decode rejects infeasible bits.
5. Add a CLI subcommand if it's user-facing, and update the README table.

## Roadmap ideas

See README "Roadmap": IBM hardware results table, warm-start QAOA, site-visit scheduling with time windows, Portfolio X-Ray integration, decomposition for larger instances.
