# Q-Optima: notes for Claude Code

Hybrid quantum-classical optimisation and quantum-AI library. Real problems become QUBOs, then
Ising Hamiltonians, then gate-level QAOA circuits. They're solved by QAOA (NumPy statevector,
noise models, optional IBM hardware), simulated annealing and classical exact/heuristic methods,
and every result is compared with a proven optimum. Quantum-AI: quantum-kernel classification on
real data, and learned (transferred) QAOA angles.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # first time (extras: qiskit, data = yfinance, notebook = ipywidgets)
pytest -q                          # 31 tests, ~6 s; run after every change
python -m qoptima benchmark        # ~60 s; regenerates results/benchmark.md + .csv
python -m qoptima figures          # ~30 s; regenerates docs/images/*.png + stats.json
python -m qoptima route --places examples/places.csv
python -m qoptima portfolio --prices examples/prices_synthetic.csv --k 3
python -m qoptima plots --buyers examples/buyers.csv --plots examples/plots.csv
python -m qoptima maxcut [--graph karate --nodes 12]
python -m qoptima odisha --n 8
python -m qoptima circuit --qasm results/route_circuit.qasm
python -m qoptima qml --dataset breast_cancer --qubits 6
```

Slash commands (in `.claude/commands/`): `/run-demo`, `/benchmark`, `/figures`,
`/add-problem <description>`, `/explain <topic>`, `/ibm-hardware`, `/publish`.

## Layout

- `qoptima/qubo.py`: `QUBO` model (upper-triangular Q + offset), penalty helpers, `all_energies()`, `to_ising()`
- `qoptima/solvers/`: `exact.py`, `annealing.py`, `qaoa.py` (statevector, ≤22 qubits; options:
  `objective="cvar"`, `warm_start=bits`, `layerwise=True` (INTERP), `noise=NoiseModel`), `noise.py`,
  `qiskit_backend.py` (optional; replays trained angles on Qiskit/IBM, not yet run on real hardware)
- `qoptima/quantum/`: `circuit.py` (gate list, decompose RZZ→CNOT·RZ·CNOT, depth, gate-by-gate
  simulation, OpenQASM 2.0, SWAP estimate), `hardware.py` (illustrative device profiles, fidelity
  estimate, `noise_model()`), `states.py` (Bloch vectors, entanglement entropy)
- `qoptima/problems/`: plug-ins `routing.py` (also `from_matrix` for real road distances),
  `portfolio.py`, `plot_matching.py`, `maxcut.py`
- `qoptima/qml/`: `kernel.py` (ZZ feature map, quantum-kernel SVM vs classical, bandwidth by CV),
  `angle_transfer.py` (learned QAOA angles)
- `qoptima/datasets.py`: Odisha coordinates, OSRM road distances (online, falls back to haversine),
  NSE prices via yfinance (online), networkx real graphs, scikit-learn real datasets
- `qoptima/viz.py` (all figures), `qoptima/showcase.py` (`make_all`, `compare_devices`, `odisha_tour_demo`)
- `qoptima/benchmark.py`, `qoptima/cli.py`
- `docs/THEORY.md`: every formula, with Mermaid diagrams and references. `docs/images/`: generated figures
- `notebooks/q_optima_colab.ipynb`: interactive Colab notebook. It is generated: edit
  `notebooks/build_notebook.py` and run it, don't hand-edit the JSON. It must work with only
  Colab's preinstalled packages plus `yfinance`, and it clones the repo and uses `sys.path`.
- `.github/workflows/tests.yml`: CI runs pytest + a quick benchmark on Python 3.10 and 3.12

## Conventions (keep these)

- **Bit order:** bitstring index `k` ↔ `x_j = (k >> j) & 1`, so qubit 0 is the least significant bit. This matches Qiskit's counts strings, where the rightmost character is qubit 0.
- **Every problem plug-in provides:** `to_qubo(penalty=None)`, `decode(bits) -> (solution, value) | (None, None)` for infeasible bits, and at least one classical baseline returning `SolveResult`. The first baseline must be **exact**, because the benchmark uses it as the optimum.
- **Penalty weights:** use the smallest value that keeps the QUBO's ground state feasible. Add a test asserting `solve_exact(qubo)` decodes to the classical optimum.
- **Optimal answers are often degenerate** (a route and its reverse, a cut and its mirror image). Count *all* lowest-energy bitstrings when reporting P(optimal) and amplification.
- **Honesty rule:** never claim quantum speed-up or advantage in code comments, README, figures or results. Report gap to optimum, feasibility, runtime and QAOA amplification (P(optimal) vs random). Comparisons must be fair: e.g. warm starts from non-optimal answers, and QML bandwidth chosen on training data only.
- Hardware profiles are illustrative orders of magnitude. Say so wherever they're shown.
- Sample data (buyers, plots, synthetic prices) is illustrative. Label anything synthetic. Live data (OSRM, NSE) must fall back gracefully offline and say which source was used.
- Pure NumPy/SciPy/pandas/networkx/scikit-learn/matplotlib in the core. Qiskit and yfinance are optional extras, imported lazily.
- Figures: use the colours in `viz.py` in fixed order, no dual axes, text in ink colours (not series colours). Look at every regenerated PNG before committing.

## Adding a new problem

Use `/add-problem`, or:
1. Create `qoptima/problems/<name>.py` with a dataclass like the existing plug-ins.
2. Export it in `qoptima/problems/__init__.py`.
3. Add instances in `benchmark.default_instances()` and a branch in `benchmark._classical()`.
4. Add tests: the QUBO optimum equals the classical exact optimum, and decode rejects infeasible bits.
5. Add a CLI subcommand if it's user-facing, and update the README table and `docs/THEORY.md`.

## Roadmap ideas

IBM hardware results table (`/ibm-hardware`), site-visit scheduling with time windows,
Portfolio X-Ray integration, decomposition for larger instances, variational quantum classifier.
