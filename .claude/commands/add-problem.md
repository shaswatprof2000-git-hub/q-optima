---
description: Add a new optimisation problem plug-in following the project conventions
argument-hint: <problem description, e.g. "site-visit scheduling with time windows">
---
Add a new problem plug-in to Q-Optima for: $ARGUMENTS

Follow "Adding a new problem" and "Conventions" in CLAUDE.md exactly:
1. Explain the QUBO formulation first (variables, objective, penalties, qubit count) and wait for my OK if it needs more than 16 qubits for a realistic small instance.
2. Create `qoptima/problems/<name>.py` with `to_qubo(penalty=None)`, `decode(bits)` (returning `(None, None)` when infeasible) and an **exact** classical baseline first, then a heuristic.
3. Use the smallest penalty that keeps the true optimum feasible, and add a test that `solve_exact(qubo)` decodes to the classical optimum.
4. Register it in `qoptima/problems/__init__.py`, `benchmark.default_instances()` and `benchmark._classical()`, and add a CLI subcommand.
5. Add the formula to `docs/THEORY.md` (the four-problems table) and a row to the README table.
6. Run `python -m pytest -q` and `python -m qoptima benchmark --quick`, then show me the results.
