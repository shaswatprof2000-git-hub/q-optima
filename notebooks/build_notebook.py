"""Generates notebooks/q_optima_colab.ipynb (edit this file, not the .ipynb JSON).

Usage:
    python notebooks/build_notebook.py                    # rewrite the notebook
    python notebooks/build_notebook.py --smoke-test       # also run every code cell locally (Agg, stubbed widgets)
"""
import json
import sys

REPO = "https://github.com/shaswatprof2000-git-hub/q-optima"
COLAB = "https://colab.research.google.com/github/shaswatprof2000-git-hub/q-optima/blob/main/notebooks/q_optima_colab.ipynb"
cells = []


def md(t):
    cells.append(("markdown", t.strip("\n")))


def code(t):
    cells.append(("code", t.strip("\n")))


md(f"""
# ⚛️ Q-Optima: quantum computing and quantum-AI, hands-on

**Hybrid quantum–classical optimisation and quantum machine learning on real data, with every answer checked against a proven optimum.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}) · [GitHub repository]({REPO}) · [Theory & formulas]({REPO}/blob/main/docs/THEORY.md)

**How to use:** click **Runtime → Run all** (about 2–4 minutes), or run cells one at a time with **Shift+Enter**. Cells marked 🎛️ have sliders you can play with.

| Part | What you'll see |
|---|---|
| A | Qubits, superposition and the Bloch sphere 🎛️ |
| B | Turning a real decision into qubits: QUBO → Ising |
| C | Real Odisha routes with real road distances, solved by QAOA |
| D | The actual quantum circuit: gates, depth, CNOTs, OpenQASM, entanglement |
| E | QAOA explorer: energy landscape and answer probabilities 🎛️ |
| F | Real hardware: noise, connectivity, fidelity 🎛️ |
| G | Real social network (Florentine families) split by Max-Cut |
| H | Real NSE stock data: quantum portfolio selection |
| I | Quantum-AI: a quantum-kernel classifier on real medical data 🎛️ |
| J | AI for quantum: learned QAOA angles · warm starts |
| K | Full benchmark and infographic |
| L | Optional: a real IBM quantum computer |
| M | Working on this with Claude Code and GitHub |

> **Honesty rule:** these problems are small enough that classical computers solve them exactly. That's what lets us *check* the quantum results. No quantum speed-up is claimed.
""")

md("## Setup")
code(f"""
# Get the code (re-running is safe) and the one extra package for live stock data
import os, sys, subprocess
if not os.path.exists("q-optima"):
    !git clone -q {REPO}.git
else:
    !git -C q-optima pull -q
%cd q-optima
sys.path.insert(0, os.getcwd())
!pip -q install yfinance

%matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
import qoptima
from qoptima import viz, solve_qaoa, solve_annealing, solve_exact, QUBO
print("Q-Optima", qoptima.__version__)
""")
code("""
# Run the automated tests: you should see "31 passed" (or more)
!python -m pytest -q
""")

# ---------------- A
md(r"""
---
## Part A: Qubits and the Bloch sphere

A classical bit is 0 **or** 1. A **qubit** is a superposition of both:

$$|\psi\rangle = \cos\tfrac{\theta}{2}\,|0\rangle + e^{i\varphi}\sin\tfrac{\theta}{2}\,|1\rangle$$

Measuring it gives **1** with probability $\sin^2(\theta/2)$. The angles $(\theta,\varphi)$ place the state on the **Bloch sphere**: north pole $|0\rangle$, south pole $|1\rangle$, equator = 50/50 superpositions.

🎛️ **Move the sliders** to rotate a qubit.
""")
code("""
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown

def one_qubit(theta=1.0, phi=0.6):
    vec = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    fig = viz.plot_bloch([vec], [f"θ={theta:.2f}, φ={phi:.2f}"],
                         title=f"P(measure 1) = sin²(θ/2) = {np.sin(theta / 2) ** 2:.2f}")
    plt.show()

interact(one_qubit, theta=FloatSlider(value=1.0, min=0, max=np.pi, step=0.05),
         phi=FloatSlider(value=0.6, min=0, max=2 * np.pi, step=0.05));
""")
md(r"""
**n qubits need $2^n$ numbers** to describe. That's why quantum computers are interesting, and why simulating them gets impossible fast:
""")
code("""
viz.plot_scaling({"4-stop route": 9, "Florentine Max-Cut": 15, "5-stop route": 16}); plt.show()
""")

# ---------------- B
md(r"""
---
## Part B: From a real decision to qubits

Every problem becomes a **QUBO**: choose binary variables $x_i\in\{0,1\}$ to minimise

$$E(x) = \sum_i Q_{ii}x_i + \sum_{i<j}Q_{ij}x_ix_j .$$

Rules ("visit every stop once") become **penalties** like $A(\sum_i x_i - 1)^2$, which is zero only when the rule holds. Quantum hardware speaks in spins $z_i=\pm1$, so we substitute $x_i=(1-z_i)/2$ and get an **Ising Hamiltonian**:

$$C = \sum_i h_i Z_i + \sum_{i<j}J_{ij}Z_iZ_j .$$

Each $J_{ij}$ becomes a two-qubit gate in the circuit, and each $h_i$ a one-qubit rotation.
""")
code("""
viz.pipeline_flowchart(); plt.show()
""")

# ---------------- C
md(r"""
---
## Part C: Real routes in Odisha, solved by QAOA

We fetch **real driving distances** from the open-source OSRM routing server (falling back to straight-line × 1.3 if offline), then find the best order to visit the stops.

A route with $m$ stops after the depot needs $m^2$ qubits: $x_{c,t}=1$ means "stop $c$ at position $t$".
""")
code("""
from qoptima.datasets import ODISHA_CITIES, odisha_places, road_distance_matrix
from qoptima.problems import RoutingProblem

places = odisha_places(["Bhubaneswar", "Phulnakhara", "Cuttack", "Khordha", "Puri"])
dist, source = road_distance_matrix(places)
print("distance source:", source)
display(pd.DataFrame(dist.round(1), index=list(places), columns=list(places)))

route = RoutingProblem.from_matrix(list(places), dist)
rq = route.to_qubo()
import math
print(f"{route.n} places -> {rq.n} qubits; {2**rq.n:,} possible bitstrings, "
      f"of which only {math.factorial(route.n - 1)} are valid tours")
""")
code("""
best = route.held_karp()
sa = solve_annealing(rq)
qa = solve_qaoa(rq, p=2, restarts=3)          # ~5–10 s on the 16-qubit simulator
for name, bits in [("Classical exact (Held-Karp)", best.bits), ("Simulated annealing", sa.bits), ("QAOA (simulated)", qa.bits)]:
    tour, km = (list(bits), route.tour_length(list(bits))) if name.startswith("Classical") else route.decode(bits)
    print(f"{name:28s}: {route.describe(tour)}  ({km:.1f} km)" if tour else f"{name}: infeasible")

tour = route.decode(qa.bits)[0] or list(best.bits)
viz.plot_route(route.names, list(places.values()), tour, route.tour_length(tour)); plt.show()
""")
md(r"""
**What did QAOA actually achieve?** It started with all $2^{16}$ answers equally likely and *amplified* the good ones. Most single measurements are still invalid, which is why we take many shots and keep the best:
""")
code("""
valid = np.array([route.decode(rq.index_to_bits(k, rq.n))[0] is not None for k in range(2 ** rq.n)])
viz.plot_amplification(rq, qa, valid); plt.show()
print(f"P(best) = {qa.info['prob_optimal']:.4f} vs random {qa.info['prob_optimal_random_guess']:.6f} "
      f"-> {qa.info['amplification']:.0f}x amplification")
""")
md("Bigger real-coordinate routes (8 Odisha towns = 49 qubits) are far beyond exact simulation, so here classical methods and annealing compete:")
code("""
from qoptima.showcase import odisha_tour_demo
r = odisha_tour_demo(8)
print(f"{len(r['places'])} towns, {r['qubits']} qubits as a QUBO, distances from {r['distance_source']}")
print(f"exact      : {r['exact_tour']}  ({r['exact_km']:.1f} km)")
print(f"nn + 2-opt : {r['heuristic_km']:.1f} km")
print(f"annealing  : " + (f"{r['annealing_km']:.1f} km" if r['annealing_feasible'] else "infeasible"))
""")

# ---------------- D
md(r"""
---
## Part D: The actual quantum circuit

QAOA is a sequence of gates:

$$|\gamma,\beta\rangle=\prod_{l=1}^{p}\underbrace{e^{-i\beta_l\sum_j X_j}}_{\text{mixer: }R_X}\;\underbrace{e^{-i\gamma_l C}}_{\text{cost: }R_{ZZ},\,R_Z}\;H^{\otimes n}|0\rangle^{\otimes n}$$

Real chips can't do $R_{ZZ}$ directly. It is compiled into **CNOT · $R_Z$ · CNOT**. Here is a tiny 3-qubit example, then the real 9-qubit route circuit:
""")
code("""
from qoptima.quantum import circuit_from_result, bloch_vectors, entanglement_entropy, DEVICES, estimate
toy = QUBO(3); toy.add_quadratic(0, 1, 2.0); toy.add_quadratic(1, 2, 2.0)
for i in range(3): toy.add_linear(i, -1.0)
toy_c = circuit_from_result(toy, solve_qaoa(toy, p=1, restarts=2))
viz.draw_circuit(toy_c, title="QAOA circuit, 3 qubits, p = 1"); plt.show()
viz.draw_circuit(toy_c.decompose(), title="Compiled to hardware gates"); plt.show()
""")
code("""
small = RoutingProblem.from_matrix(list(places)[:4], dist[:4, :4])   # 4 places -> 9 qubits
sq = small.to_qubo()
sres = solve_qaoa(sq, p=2)
circ = circuit_from_result(sq, sres)
s = circ.summary()
print(f"9-qubit route circuit: {s['gates']} gates, {s['two_qubit_gates']} CNOTs, depth {s['depth']}")
print(pd.Series(s["counts"]).rename("count").to_frame().T.to_string())
open("route_circuit.qasm", "w").write(circ.to_qasm())
print("\\nOpenQASM 2.0 (first lines), runnable on IBM / Amazon Braket / other platforms:\\n")
print("\\n".join(circ.to_qasm().splitlines()[:14]), "\\n...")
# from google.colab import files; files.download("route_circuit.qasm")   # uncomment to download
""")
md(r"""
**Inside the state.** Each qubit's own state is a point on the Bloch sphere. If it's inside the sphere ($|r|<1$), the qubit is **entangled** with the others. Entanglement across a split is measured in bits: $S=-\sum_k s_k^2\log_2 s_k^2$.
""")
code("""
from qoptima.solvers.qaoa import qaoa_state
E = sq.all_energies(); cost = (E - E.mean()) / E.std()
psi = qaoa_state(cost, sres.info["gammas"], sres.info["betas"], sq.n)
viz.plot_bloch(bloch_vectors(psi, sq.n), [n.replace("@", " @ step ") for n in sq.var_names]); plt.show()
print(f"Entanglement between the first 4 and last 5 qubits: {entanglement_entropy(psi, sq.n, 4):.2f} bits (max 4)")
""")

# ---------------- E
md(r"""
---
## Part E: 🎛️ QAOA explorer

The classical optimiser searches the angles $(\gamma,\beta)$ to minimise $\langle C\rangle$. Left: the whole landscape for $p=1$ (darker = better). Right: what you'd measure at the angles you pick. **Drag the sliders** to find the dark valley and watch the best route (blue) grow.
""")
code("""
gs, bs, Z = viz.landscape(sq, n_grid=36)
opt_mask = E <= E.min() + 1e-9
valid_s = np.array([small.decode(sq.index_to_bits(k, sq.n))[0] is not None for k in range(2 ** sq.n)])

def qaoa_explorer(gamma=0.6, beta=0.6):
    probs = np.abs(qaoa_state(cost, [gamma], [beta], sq.n)) ** 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
    im = a1.imshow(Z, origin="lower", aspect="auto", cmap="Blues_r", extent=[gs[0], gs[-1], bs[0], bs[-1]])
    a1.scatter([gamma], [beta], s=120, marker="o", color=viz.ORANGE, edgecolor="white", linewidth=2)
    a1.set_xlabel("γ"); a1.set_ylabel("β"); a1.set_title("⟨C⟩ landscape (p = 1)")
    top = np.argsort(-probs)[:14]
    colors = [viz.BLUE if opt_mask[k] else (viz.AQUA if valid_s[k] else viz.GRAY) for k in top]
    a2.bar(range(len(top)), probs[top] * 100, color=colors)
    a2.set_xticks(range(len(top)), [f"{E[k]:.0f}" for k in top], rotation=60, fontsize=8)
    a2.set_ylabel("probability (%)"); a2.set_xlabel("energy of answer (lower is better)")
    a2.set_title(f"P(best) = {probs[opt_mask].sum():.3f}   ·   P(valid) = {probs[valid_s].sum():.3f}   "
                 f"(random: {opt_mask.mean():.3f} / {valid_s.mean():.3f})", fontsize=10)
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=viz.BLUE, label="best route"), Patch(color=viz.AQUA, label="valid route"),
                       Patch(color=viz.GRAY, label="invalid")], frameon=False, fontsize=8)
    plt.tight_layout(); plt.show()

interact(qaoa_explorer, gamma=FloatSlider(value=0.6, min=0, max=np.pi, step=0.02),
         beta=FloatSlider(value=0.6, min=-np.pi / 2, max=np.pi / 2, step=0.02));
""")

# ---------------- F
md(r"""
---
## Part F: Real quantum hardware: noise, connectivity, fidelity

Real qubits make errors. A simple, transparent model:

$$F\approx(1-\varepsilon_1)^{N_1}(1-\varepsilon_2)^{N_2+3N_\text{SWAP}}\,e^{-nT/T_2},\qquad P_\text{noisy}=F\,P_\text{ideal}+\frac{1-F}{2^n}$$

Superconducting chips connect only neighbouring qubits, so they need extra SWAP gates. Trapped ions connect everything but are slower. *Profiles are illustrative orders of magnitude, not a specific machine.*
""")
code("""
from qoptima.showcase import compare_devices
rows = compare_devices(sq, sres)
display(pd.DataFrame([{**estimate(circ, DEVICES[r["key"]]), "P(best) %": round(100 * r["p_opt"], 2)} for r in rows])
        [["device", "two_qubit_gates", "swaps", "duration_us", "fidelity", "P(best) %"]].round(4))
viz.plot_devices(rows); plt.show()
""")
md("🎛️ **Design your own quantum computer:** how good must the two-qubit gates be for this circuit to work?")
code("""
from qoptima.quantum import DeviceProfile
from qoptima.quantum.hardware import noise_model
probs_ideal = np.abs(psi) ** 2

def hardware_explorer(two_qubit_error_pct=0.5, readout_error_pct=1.0, connectivity="all"):
    dev = DeviceProfile("custom", "custom", 1e-4, two_qubit_error_pct / 100, readout_error_pct / 100,
                        1e9, 30, 200, connectivity)
    nm = noise_model(circ, dev)
    samples = nm.sample(probs_ideal, sq.n, 20000, np.random.default_rng(0), nm.fidelity_override)
    p = opt_mask[samples].mean()
    e = estimate(circ, dev)
    print(f"two-qubit gates incl. SWAPs: {e['two_qubit_incl_swaps']}   circuit fidelity: {e['fidelity']:.3f}")
    print(f"P(best route) per shot: {100 * p:.2f}%   (noiseless {100 * probs_ideal[opt_mask].sum():.2f}%, "
          f"random {100 * opt_mask.mean():.2f}%)")

interact(hardware_explorer,
         two_qubit_error_pct=FloatSlider(value=0.5, min=0.01, max=3, step=0.01),
         readout_error_pct=FloatSlider(value=1.0, min=0, max=5, step=0.1),
         connectivity=Dropdown(options=["all", "line"], value="all"));
""")

# ---------------- G
md(r"""
---
## Part G: A real social network: Max-Cut on the Florentine families

The marriage network of 15 Renaissance Florentine families (Padgett & Ansell, 1993) is real historical data. **Max-Cut** splits the network into two groups so that as many ties as possible cross between them:

$$\max_x \sum_{(i,j)\in\text{edges}} w_{ij}\,(x_i + x_j - 2x_ix_j)$$

It's the standard QAOA benchmark problem. Here it uses 15 qubits, one per family.
""")
code("""
from qoptima.datasets import real_graph
from qoptima.problems import MaxCutProblem
g, fam = real_graph("florentine")
mc = MaxCutProblem.from_networkx(g, fam)
mres = solve_qaoa(mc.to_qubo(), p=2, restarts=3)
best_cut = mc.brute_force().energy
print(f"QAOA cut: {mc.cut_value(mres.bits):.0f}  |  best possible: {best_cut:.0f}  |  "
      f"P(best) {mres.info['prob_optimal']:.3f} = {mres.info['amplification']:.0f}x random")
viz.plot_maxcut(mc, mres.bits, g); plt.show()
""")

# ---------------- H
md(r"""
---
## Part H: Real stock data: quantum portfolio selection

Pick $k$ stocks to balance return $\mu$ against risk $\Sigma$ (Markowitz, 1952):

$$\min_x\; q\,w^\top\Sigma w-\mu^\top w,\quad w=x/k,\quad \textstyle\sum_i x_i=k$$

Prices are **2 years of real NSE daily closes** (via Yahoo Finance). If that fails (offline), the cell says so and uses clearly labelled synthetic data. *Educational only, not investment advice.*
""")
code("""
from qoptima.datasets import nse_prices, NIFTY_SAMPLE
from qoptima.problems import PortfolioProblem
try:
    prices = nse_prices(NIFTY_SAMPLE[:8], period="2y")
    label = "REAL NSE data"
except Exception as e:
    print("Live data unavailable:", e)
    prices = pd.read_csv("examples/prices_synthetic.csv", index_col=0, parse_dates=True)
    label = "SYNTHETIC demo data"
print(label, "|", prices.index[0].date(), "to", prices.index[-1].date(), "|", ", ".join(prices.columns))

pf = PortfolioProblem.from_prices(prices, k=3)
pq = pf.to_qubo()
rows = []
for name, res in [("Classical exact", pf.brute_force()), ("Greedy shortcut", pf.greedy()),
                  ("Annealing", solve_annealing(pq)), ("QAOA (simulated)", solve_qaoa(pq, p=2))]:
    picks, _ = pf.decode(res.bits)
    st = pf.stats(res.bits)
    rows.append({"method": name, "picks": ", ".join(picks) if picks else "infeasible",
                 "expected return": f"{st['expected_return']:.1%}", "volatility": f"{st['volatility']:.1%}"})
display(pd.DataFrame(rows))
""")

# ---------------- I
md(r"""
---
## Part I: Quantum-AI: a quantum-kernel classifier on real medical data

**Breast Cancer Wisconsin**: 569 real tumours, 30 measurements each, malignant vs benign. Each sample $x$ (compressed to $n$ features by PCA) is encoded into an $n$-qubit state by the **ZZ feature map**:

$$|\phi(x)\rangle=\big(U_Z(x)H^{\otimes n}\big)^r|0\rangle,\quad U_Z=\exp\Big(i\Big[\sum_i b\,x_iZ_i+\sum_{(i,j)}b^2(\pi-x_i)(\pi-x_j)Z_iZ_j\Big]\Big)$$

The **quantum kernel** $k(x,x')=|\langle\phi(x)|\phi(x')\rangle|^2$ (what a quantum computer estimates by sampling) feeds a support-vector machine. We compare it fairly with classical models on held-out test data. *A learning tool, not a diagnostic.*
""")
code("""
from qoptima.qml import run_classification
qml = run_classification("breast_cancer", n_qubits=6)
print(f"{qml['n_qubits']} qubits | train {qml['train']} / test {qml['test']} | PCA keeps {qml['pca_variance_kept']:.0%} "
      f"of the variance | bandwidth b = {qml['bandwidth']} chosen by cross-validation")
viz.plot_qml(qml); plt.show()
""")
md(r"🎛️ **The bandwidth $b$ matters.** Too large and all states look unrelated (the kernel concentrates near 0, called *exponential concentration*). Too small and everything looks the same. Slide it and watch the accuracy:")
code("""
def kernel_explorer(bandwidth=0.5, qubits=4):
    r = run_classification("breast_cancer", n_qubits=qubits, bandwidths=(bandwidth,))
    K = r["kernel_train"]; off = K[~np.eye(len(K), dtype=bool)]
    print(f"b = {bandwidth:.2f}: quantum-kernel test accuracy {r['results']['Quantum kernel SVM']['accuracy']:.1%}"
          f"   |   mean overlap {off.mean():.3f}  (1/2^n = {1 / 2 ** qubits:.3f})")
    order = np.argsort(r["y_train"], kind="stable")
    plt.figure(figsize=(4.5, 4)); plt.imshow(K[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1)
    plt.axis("off"); plt.title(f"kernel matrix, b = {bandwidth:.2f}"); plt.show()

interact(kernel_explorer, bandwidth=FloatSlider(value=0.5, min=0.05, max=2.0, step=0.05),
         qubits=IntSlider(value=4, min=2, max=8));
""")

# ---------------- J
md(r"""
---
## Part J: AI for quantum, and the warm-start lesson

**Learned angles.** Training QAOA costs hundreds of circuit runs. For one problem family the best angles *concentrate*, so angles learned from a few small graphs work on new, bigger graphs **with zero training**:
""")
code("""
from qoptima.qml import learn_angles, evaluate_transfer
learned = learn_angles(n_nodes=8, p=2, n_train=6)
transfer = evaluate_transfer(learned, n_test=4, n_nodes=10)
display(pd.DataFrame(transfer).round(3))
viz.plot_angle_transfer(transfer); plt.show()
""")
md(r"**Warm starts** begin QAOA from a classical answer $c$, using $R_Y(\theta_i)$ with $\sin^2(\theta_i/2)=c_i$. They lock onto that answer, so they only help when it's already good:")
code("""
import itertools
tours = sorted(([0, *perm, 0] for perm in itertools.permutations(range(1, small.n))), key=small.tour_length)
def tour_bits(t):
    b = np.zeros(sq.n, dtype=int)
    for pos, c in enumerate(t[1:-1], start=1): b[small.var(c, pos)] = 1
    return b
rows = [{"label": "random guessing", "p_opt": float(opt_mask.mean()), "baseline": True},
        {"label": "standard QAOA (p = 2)", "p_opt": sres.info["prob_optimal"], "standard": True}]
for t in (tours[-1], tours[0]):
    r = solve_qaoa(sq, p=2, warm_start=tour_bits(t))
    rows.append({"label": f"warm start from a {small.tour_length(t):.0f} km tour", "p_opt": r.info["prob_optimal"]})
viz.plot_warm_start(rows); plt.show()
""")

# ---------------- K
md("""
---
## Part K: Full benchmark and infographic
Every problem, every method, checked against the proven optimum (under a minute).
""")
code("""
from qoptima import benchmark
df = benchmark.run(verbose=False)
display(Markdown(open("results/benchmark.md").read()))
""")
code("""
from qoptima.showcase import make_all
stats = make_all("figures", quick=True, verbose=False)     # all charts as PNG files in ./figures
display(Image("figures/00_infographic.png"))
""")

# ---------------- L
md("""
---
## Part L: Optional: a real IBM quantum computer

1. Create a free account at [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com) and copy your API key.
2. Click the 🔑 **Secrets** icon on the left, add `IBM_QUANTUM_TOKEN`, and allow notebook access.
3. Set `RUN_ON_IBM = True` and run. The circuit is trained on the simulator, then sampled once on hardware.

> Not yet tested end to end, and IBM's account options change between library versions. If it errors, check the `qiskit-ibm-runtime` docs.
""")
code("""
RUN_ON_IBM = False
if RUN_ON_IBM:
    !pip -q install qiskit qiskit-ibm-runtime
    from google.colab import userdata
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qoptima.solvers.qiskit_backend import sample_circuit
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                      token=userdata.get("IBM_QUANTUM_TOKEN"), overwrite=True)
    hw = sample_circuit(sq, sres, shots=2000, use_ibm=True)
    tour, km = small.decode(hw.bits)
    print(hw.solver, "->", small.describe(tour) if tour else "infeasible", f"({km:.1f} km)" if km else "")
    print(f"classical optimum: {small.held_karp().energy:.1f} km")
else:
    print("Skipped. Set RUN_ON_IBM = True to try real hardware.")
""")

# ---------------- M
md(f"""
---
## Part M: Working on Q-Optima with Claude Code and GitHub

The repository is set up for **Claude Code** (in VS Code or the terminal):

- `CLAUDE.md` tells Claude the commands, conventions and honesty rules.
- Custom slash commands in `.claude/commands/`: `/run-demo`, `/benchmark`, `/figures`, `/add-problem`, `/explain`.
- GitHub Actions re-runs the tests on every push to `main` and on every pull request.

Typical loop on your computer:
```bash
git clone {REPO} && cd q-optima
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
claude            # then try:  /run-demo   or   /add-problem site-visit scheduling with time windows
```
Theory, formulas and diagrams: [docs/THEORY.md]({REPO}/blob/main/docs/THEORY.md)

---
Built by **Shaswat Mohanty** · MIT licence
""")


def to_ipynb():
    out = []
    for kind, src in cells:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        c = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        out.append(c)
    return {"cells": out, "metadata": {"colab": {"provenance": [], "name": "q_optima_colab.ipynb", "toc_visible": True},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


STUB = '''
import matplotlib; matplotlib.use("Agg")
import sys, types
def _interact(f, **kw):
    args = {k: (v.value if hasattr(v, "value") else v) for k, v in kw.items()}
    f(**args)
class _W:
    def __init__(self, value=None, **kw): self.value = value
ipw = types.ModuleType("ipywidgets"); ipw.interact = _interact
ipw.FloatSlider = ipw.IntSlider = ipw.Dropdown = _W
sys.modules["ipywidgets"] = ipw
ipd = types.ModuleType("IPython.display")
ipd.display = lambda x: print(x if not hasattr(x, "data") else "[rich output]")
class _Rich:
    def __init__(self, *a, **k): self.data = a
ipd.Markdown = ipd.Image = _Rich
_ip = types.ModuleType("IPython"); _ip.get_ipython = lambda: None; _ip.display = ipd; _ip.version_info = (9, 0)
sys.modules["IPython"] = _ip; sys.modules["IPython.display"] = ipd
import time; _T0 = time.time()
'''


def to_script(repo_dir):
    out = [STUB, f"import os; os.chdir({repo_dir!r})"]
    for i, (kind, src) in enumerate(cells):
        if kind != "code":
            continue
        out.append(f"print('--- cell {i} ---', round(time.time() - _T0, 1), flush=True)")
        for line in src.split("\n"):
            s = line.strip()
            out.append(line[: len(line) - len(line.lstrip())] + "pass" if s.startswith(("!", "%")) else line)
    return "\n".join(out)


if __name__ == "__main__":
    import os
    import subprocess
    import tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    nb_path = os.path.join(here, "q_optima_colab.ipynb")
    with open(nb_path, "w") as f:
        json.dump(to_ipynb(), f, indent=1, ensure_ascii=False)
    print("wrote", nb_path)
    if "--smoke-test" in sys.argv:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["cp", "-r", repo, os.path.join(tmp, "repo")], check=True)
            script = os.path.join(tmp, "smoke.py")
            with open(script, "w") as f:
                f.write(to_script(os.path.join(tmp, "repo")))
            sys.exit(subprocess.run([sys.executable, "-I", script], cwd=tmp).returncode)
