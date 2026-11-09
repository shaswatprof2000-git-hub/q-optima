"""Honest benchmark: quantum (QAOA, simulated) vs simulated annealing vs classical.

For every instance we record the solution quality relative to the proven optimum,
whether the answer is even feasible, and wall-clock time. Results go to
results/benchmark.md and results/benchmark.csv.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .datasets import real_graph
from .problems import BBSR_CUTTACK_STOPS, MaxCutProblem, PlotMatchingProblem, PortfolioProblem, RoutingProblem
from .solvers import solve_annealing, solve_exact, solve_qaoa
from .solvers.noise import NoiseModel

QAOA_MAX_QUBITS = 16
NOISY_MAX_QUBITS = 12   # the noisy run retrains QAOA, so keep it to the quicker instances


@dataclass
class Instance:
    family: str
    label: str
    problem: object
    sense: str  # "min" or "max"
    unit: str


def default_instances(quick: bool = False) -> list[Instance]:
    stops4 = dict(list(BBSR_CUTTACK_STOPS.items())[:4])
    inst = [
        Instance("routing", "4 stops (BBSR-Cuttack)", RoutingProblem.from_latlon(stops4), "min", "km"),
        Instance("portfolio", "8 stocks, pick 3", PortfolioProblem.synthetic(8, 3, seed=7), "min", "objective"),
        Instance("plot_matching", "3 buyers x 4 plots", PlotMatchingProblem.sample(3, 4, seed=3), "max", "fit score"),
    ]
    if not quick:
        inst += [
            Instance("routing", "5 stops (BBSR-Cuttack)", RoutingProblem.from_latlon(BBSR_CUTTACK_STOPS), "min", "km"),
            Instance("portfolio", "12 stocks, pick 4", PortfolioProblem.synthetic(12, 4, seed=11), "min", "objective"),
            Instance("plot_matching", "4 buyers x 4 plots", PlotMatchingProblem.sample(4, 4, seed=5), "max", "fit score"),
            Instance("maxcut", "Florentine families network (real data)",
                     MaxCutProblem.from_networkx(*real_graph("florentine")), "max", "links cut"),
        ]
    return inst


def _classical(inst: Instance):
    p = inst.problem
    if inst.family == "routing":
        return [p.held_karp(), p.nearest_neighbour_2opt()]
    if inst.family == "portfolio":
        return [p.brute_force(), p.greedy()]
    if inst.family == "maxcut":
        return [p.brute_force(), p.local_search()]
    return [p.hungarian()]


def run(quick: bool = False, out_dir: str | Path = "results", seed: int = 0, verbose: bool = True):
    rows = []
    for inst in default_instances(quick):
        p = inst.problem
        qubo = p.to_qubo()
        classical = _classical(inst)
        optimum = classical[0].energy  # first baseline is exact
        quantum_style = [solve_exact(qubo), solve_annealing(qubo, seed=seed)]
        if qubo.n <= QAOA_MAX_QUBITS:
            quantum_style.append(solve_qaoa(qubo, p=2, restarts=3, seed=seed))
        if qubo.n <= NOISY_MAX_QUBITS:
            quantum_style.append(solve_qaoa(qubo, p=2, restarts=3, seed=seed, noise=NoiseModel()))

        for r in quantum_style:
            _, value = p.decode(r.bits)
            rows.append(_row(inst, qubo.n, r.solver, value, optimum, r.runtime_s, r.info))
        for r in classical:
            rows.append(_row(inst, qubo.n, r.solver, r.energy, optimum, r.runtime_s, {}))
        if verbose:
            print(f"done: {inst.family} / {inst.label} ({qubo.n} qubits)")

    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False)
    (out / "benchmark.md").write_text(to_markdown(df))
    return df


def _row(inst, qubits, solver, value, optimum, runtime, info):
    feasible = value is not None
    if feasible:
        gap = (value - optimum) if inst.sense == "min" else (optimum - value)
        gap_pct = 100 * gap / abs(optimum) if optimum else 0.0
    else:
        gap_pct = None
    note = ""
    if solver == "exact":
        solver = "exhaustive QUBO check"
    if "prob_optimal" in info:
        note = (f"p={info['p']}; P(optimal)={info['prob_optimal']:.3f} "
                f"({info['amplification']:.0f}x random guessing)")
        if info.get("noise"):
            note += f"; noisy sampling, circuit fidelity {info['circuit_fidelity']:.2f}"
    return {
        "problem": inst.family, "instance": inst.label, "qubits": qubits, "solver": solver,
        "value": None if value is None else round(value, 4), "unit": inst.unit,
        "feasible": feasible, "gap_to_optimum_pct": None if gap_pct is None else round(gap_pct, 2),
        "runtime_s": round(runtime, 3), "notes": note,
    }


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["# Q-Optima benchmark", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. QAOA runs on an exact statevector "
             "simulator (no noise). Gap is measured against the proven optimum from the "
             "classical exact method; 0.00 means the optimum was found.", ""]
    for (fam, label), g in df.groupby(["problem", "instance"], sort=False):
        lines += [f"## {fam}: {label} ({g.qubits.iloc[0]} qubits)", "",
                  "| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |",
                  "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            gap = "—" if r.gap_to_optimum_pct is None or pd.isna(r.gap_to_optimum_pct) else f"{r.gap_to_optimum_pct:.2f}%"
            val = "infeasible" if not r.feasible else f"{r.value} {r.unit}"
            lines.append(f"| {r.solver} | {val} | {'yes' if r.feasible else 'no'} | {gap} | {r.runtime_s} | {r.notes} |")
        lines.append("")
    lines += ["## Reading these results honestly", ""]
    lines += _honest_summary(df)
    lines.append("")
    return "\n".join(lines)


def _honest_summary(df: pd.DataFrame) -> list[str]:
    out = []
    for solver, label in [("qaoa", "QAOA (simulated, noiseless)"), ("qaoa_noisy", "QAOA with gate & readout noise"),
                          ("simulated_annealing", "Simulated annealing")]:
        g = df[df.solver == solver]
        if g.empty:
            continue
        hits = int((g.gap_to_optimum_pct == 0).sum())
        worst = g.gap_to_optimum_pct.max()
        out.append(f"- {label} found the proven optimum in {hits} of {len(g)} instances"
                   + (f" (worst gap {worst:.2f}%)." if hits < len(g) else "."))
    q = df[df.solver == "qaoa"]
    ex = df[df.solver.str.startswith("classical_exact")]
    if not q.empty and not ex.empty:
        out.append(f"- Classical exact methods took at most {ex.runtime_s.max():.3f}s; simulated "
                   f"QAOA took up to {q.runtime_s.max():.1f}s. No quantum speed-up is claimed: at "
                   "these sizes classical methods win easily.")
    out += ["- What QAOA does show is *amplification*: after optimisation, the optimal answer is "
            "measured many times more often than random guessing (see Notes).",
            "- The research question is how gap and P(optimal) change as qubits grow, and how much "
            "real-hardware noise erodes them (sample on IBM Quantum with `--ibm`)."]
    return out
