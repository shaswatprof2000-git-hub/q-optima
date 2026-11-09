---
description: Run a small Q-Optima circuit on real IBM quantum hardware and compare it with the simulator
---
Help me run Q-Optima on a real IBM quantum computer.

1. Check that `qiskit` and `qiskit-ibm-runtime` are installed in `.venv`. If not, run `pip install -e ".[qiskit]"`.
2. Check whether an IBM account is saved (`QiskitRuntimeService()`). If not, tell me how to get an API key and how to save it, but **never ask me to paste the token into the chat**. I'll run the save command myself.
3. Use the 4-stop route (9 qubits): train with `solve_qaoa` (p=1 to keep the circuit shallow), then call `qoptima.solvers.qiskit_backend.sample_circuit(..., use_ibm=True)`. This module hasn't been tested on hardware yet, so fix any API mismatches with the installed Qiskit version.
4. Compare: hardware best route and P(best) vs the noiseless simulator vs `compare_devices` estimates. Save a results table to `results/ibm_run.md`.
5. Ask before submitting the job, because it uses my IBM quota.
