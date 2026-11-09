# Q-Optima

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaswatprof2000-git-hub/q-optima/blob/main/notebooks/q_optima_colab.ipynb)
[![tests](https://github.com/shaswatprof2000-git-hub/q-optima/actions/workflows/tests.yml/badge.svg)](https://github.com/shaswatprof2000-git-hub/q-optima/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Hybrid quantum-classical optimisation for real problems, with an honest benchmark.**

Q-Optima turns everyday decisions into QUBO problems (the input format of quantum
optimisers) and solves each one three ways:

| Approach | Method in this repo |
|---|---|
| Quantum (gate model) | **QAOA** on a built-in NumPy statevector simulator; optional replay on **IBM Quantum hardware** via Qiskit |
| Quantum-inspired | **Simulated annealing** (what quantum annealers mimic physically) |
| Classical | Exact solvers (Held-Karp, enumeration, Hungarian) and the quick heuristics people actually use |

Every result is checked against the proven optimum, so you can see where each method stands.
**No quantum speed-up is claimed.** At today's sizes classical methods win, and the benchmark shows it.

## Problems included

| Plug-in | Real-world use | Qubits |
|---|---|---|
| `routing` | Best visiting order for deliveries or site visits (Bhubaneswar–Cuttack sample) | (stops−1)² |
| `portfolio` | Pick *k* of *N* stocks, balancing return against risk (mean-variance) | N |
| `plot_matching` | Match real-estate buyers to plots by budget, size, facing, road width, corner | buyers × plots |

## Run it

### Option 1: in your browser (no installation)

Click **Open in Colab** above, then **Runtime → Run all**. The notebook clones this repo, runs the tests,
solves all three problems, plots the route and shows what QAOA actually does. It ends with an optional
step for real IBM quantum hardware.

### Option 2: on your computer

```bash
git clone https://github.com/shaswatprof2000-git-hub/q-optima && cd q-optima
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # numpy, scipy, pandas, pytest
pytest -q                        # 11 tests

python -m qoptima benchmark      # full comparison -> results/benchmark.md (~40 s)
python -m qoptima route --places examples/places.csv
python -m qoptima plots --buyers examples/buyers.csv --plots examples/plots.csv
python -m qoptima portfolio --prices examples/prices_synthetic.csv --k 3
```

Use your own data by editing the CSVs in `examples/` (same columns).

### Automatic checks

Every push runs the tests on Python 3.10 and 3.12 with GitHub Actions. The **tests** badge above shows the result.

Example output (`route`, 5 places, 16 qubits):

```
classical optimum : Office (Bhubaneswar) -> Khandagiri client -> Patia client -> Cuttack client -> Phulnakhara site -> Office  (66.9 km)
simulated_annealing : ... same route (66.9 km)
qaoa                : ... same route (66.9 km)
```

## Results

See [`results/benchmark.md`](results/benchmark.md). Summary of the current run (6 instances, 8–16 qubits):

- QAOA (p = 2, simulated, noiseless) and simulated annealing both found the proven optimum on all 6 instances.
- Classical exact methods finished in milliseconds; simulated QAOA took up to ~17 s at 16 qubits.
- QAOA's value shows up as **amplification**: after training, the optimal answer is measured 4–92× more often than random guessing.
- The simple heuristics people often use were 1.8–14% worse than optimal on the portfolio problem, which shows why exact optimisation matters even before quantum.

## Running on real quantum hardware (IBM)

```bash
pip install -e ".[qiskit]"
python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='YOUR_TOKEN')"
python -m qoptima route --ibm
```

Angles are trained on the simulator, then the same circuit is sampled once on the least-busy
IBM device ("parameter transfer"), so you use very little QPU time. Comparing hardware results
with the noiseless simulator is itself a useful experiment: it measures how much noise costs.

> The Qiskit adapter (`qoptima/solvers/qiskit_backend.py`) was written against Qiskit 1.x/2.x
> primitives but not executed where this repo was generated. Treat the first run as a test.
> The account-setup `channel` name has changed between versions, so check the current
> qiskit-ibm-runtime docs.

## How the formulations work

**QUBO:** minimise `E(x) = Σ Qᵢᵢxᵢ + Σᵢ<ⱼ Qᵢⱼxᵢxⱼ` over binary `x`. Constraints become penalties:

- "exactly k of these": `A·(Σxᵢ − k)²`
- "at most one of these": `A·Σᵢ<ⱼ xᵢxⱼ`

**Routing:** `x[c,t]=1` means stop *c* is visited at position *t* (depot fixed). Each stop and each position must be used exactly once, and the cost is the sum of leg distances.

**Portfolio:** `x[i]=1` means hold stock *i* (equal weights). Cost is `q·wᵀΣw − μᵀw` subject to exactly *k* stocks. μ and Σ are estimated from daily log returns, annualised.

**Plot matching:** `x[b,p]=1` means plot *p* goes to buyer *b*. The model maximises total fit, with one plot per buyer and at most one buyer per plot.

**Penalty weight:** if *A* is too small, the optimum breaks the constraints. If it's too large, the energy landscape becomes too hard for annealing and QAOA. The defaults were tuned so the exhaustive check confirms the true optimum is still feasible, and the benchmark re-checks this on every run.

**QAOA:** cost normalised to zero mean and unit variance, *p* layers, COBYLA with a linear-ramp start and random restarts. Answers are read by **sampling shots**, as on hardware.

## Project layout

```
qoptima/
  qubo.py                 QUBO model, penalties, Ising conversion
  solvers/                exact, simulated annealing, QAOA (NumPy), Qiskit/IBM adapter
  problems/               routing, portfolio, plot_matching
  benchmark.py, cli.py
examples/                 sample CSVs (illustrative data only)
notebooks/                Google Colab notebook
results/                  generated benchmark tables
tests/
```

## Roadmap

- [ ] Noise model and IBM hardware results table (8–16 qubits)
- [ ] Warm-start QAOA from the annealing solution; compare p = 1…5
- [ ] Site-visit *scheduling* with time windows (sales team calendar)
- [ ] Portfolio plug-in reading holdings from Portfolio X-Ray
- [ ] Larger instances via decomposition (split the problem into sub-QUBOs)

## Notes

- Example buyers, plots and prices are **illustrative**, not real inventory or market data.
- The portfolio tool is educational and not investment advice.
- Coordinates for the Bhubaneswar–Cuttack demo are approximate. Distances are straight-line × 1.3 for roads.

MIT licence · Shaswat Mohanty
