---
description: Set up the environment if needed, run the tests and a tour of every Q-Optima feature
---
Run a full Q-Optima demo and summarise it for me in plain language.

1. If `.venv` doesn't exist, create it (`python3 -m venv .venv`) and run `pip install -e ".[dev]"` inside it. Use `.venv/bin/python` for everything below.
2. Run `python -m pytest -q` and report the pass count.
3. Run these and show the key lines of each output:
   - `python -m qoptima route --places examples/places.csv`
   - `python -m qoptima maxcut`
   - `python -m qoptima circuit --qasm results/route_circuit.qasm`
   - `python -m qoptima qml --qubits 6`
   - `python -m qoptima odisha --n 8`
4. Summarise in a short table: problem, qubits, classical optimum, what QAOA or annealing found, and one honest takeaway. Follow the honesty rule in CLAUDE.md: never claim a quantum speed-up.
