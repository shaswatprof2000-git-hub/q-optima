"""Command line:  python -m qoptima <command>

  benchmark [--quick] [--out results]     run the full comparison table
  route     [--places file.csv]           best visiting order (CSV: name,lat,lon; first row = start)
  portfolio [--prices file.csv] --k 4     pick k stocks (CSV: date + one close-price column per ticker)
  plots     [--buyers b.csv --plots p.csv] match buyers to plots
  maxcut    [--graph florentine|karate]   split a real social network (Max-Cut)
  odisha    [--n 8]                        bigger real-coordinate route across Odisha (classical vs annealing)
  circuit   [--qasm out.qasm]              build the gate-level QAOA circuit, gate counts, hardware estimates
  qml       [--dataset breast_cancer|wine] [--qubits 4]   quantum-kernel classifier vs classical models
  figures   [--out docs/images] [--quick]  regenerate every figure and the infographic
  Add --ibm to route/portfolio/plots/maxcut to also sample the QAOA circuit on IBM Quantum hardware.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import benchmark
from .problems import BBSR_CUTTACK_STOPS, Buyer, Plot, PlotMatchingProblem, PortfolioProblem, RoutingProblem
from .solvers import solve_annealing, solve_qaoa
from .solvers.qaoa import MAX_QUBITS


def _solve_all(problem, qubo, use_ibm: bool):
    results = [solve_annealing(qubo)]
    if qubo.n <= MAX_QUBITS:
        qa = solve_qaoa(qubo)
        results.append(qa)
        if use_ibm:
            from .solvers.qiskit_backend import sample_circuit
            results.append(sample_circuit(qubo, qa, use_ibm=True))
    else:
        print(f"({qubo.n} qubits: too many for the simulator, QAOA skipped)")
    return results


def cmd_route(a):
    if a.places:
        df = pd.read_csv(a.places)
        places = {r["name"]: (r["lat"], r["lon"]) for _, r in df.iterrows()}
    else:
        places = BBSR_CUTTACK_STOPS
    p = RoutingProblem.from_latlon(places)
    q = p.to_qubo()
    print(f"{p.n} places -> {q.n} qubits")
    best = p.held_karp()
    print(f"classical optimum : {p.describe(best.bits)}  ({best.energy:.1f} km)")
    for r in _solve_all(p, q, a.ibm):
        tour, km = p.decode(r.bits)
        print(f"{r.solver:20s}: " + (f"{p.describe(tour)}  ({km:.1f} km)" if tour else "infeasible"))


def cmd_portfolio(a):
    p = (PortfolioProblem.from_csv(a.prices, a.k, a.risk) if a.prices
         else PortfolioProblem.synthetic(10, a.k, seed=1, risk_aversion=a.risk))
    q = p.to_qubo()
    best = p.brute_force()
    print(f"{p.n} stocks, pick {p.k} -> {q.n} qubits" + ("" if a.prices else "  [synthetic demo data]"))
    s = p.stats(best.bits)
    print(f"classical optimum : {p.decode(best.bits)[0]}  return {s['expected_return']:.1%}, vol {s['volatility']:.1%}")
    for r in _solve_all(p, q, a.ibm):
        picks, _ = p.decode(r.bits)
        if picks:
            s = p.stats(r.bits)
            print(f"{r.solver:20s}: {picks}  return {s['expected_return']:.1%}, vol {s['volatility']:.1%}")
        else:
            print(f"{r.solver:20s}: infeasible")
    print("Educational tool only, not investment advice.")


def cmd_plots(a):
    if a.buyers and a.plots:
        bdf, pdf = pd.read_csv(a.buyers), pd.read_csv(a.plots)
        buyers = [Buyer(r["name"], r["budget_lakh"], r["min_area_sqft"],
                        tuple(str(r["preferred_facing"]).split("|")), r["min_road_ft"],
                        bool(r["wants_corner"])) for _, r in bdf.iterrows()]
        plots = [Plot(r["plot_id"], r["area_sqft"], r["price_lakh"], r["facing"],
                      r["road_width_ft"], bool(r["corner"])) for _, r in pdf.iterrows()]
        p = PlotMatchingProblem(buyers, plots)
    else:
        p = PlotMatchingProblem.sample(3, 4, seed=3)
        print("[sample data]")
    q = p.to_qubo()
    best = p.hungarian()
    print(f"{len(p.buyers)} buyers x {len(p.plots)} plots -> {q.n} qubits")
    print(f"classical optimum : {p.decode(best.bits)[0]}  (fit {best.energy:.1f})")
    for r in _solve_all(p, q, a.ibm):
        pairs, total = p.decode(r.bits)
        print(f"{r.solver:20s}: " + (f"{pairs}  (fit {total:.1f})" if pairs else "infeasible"))


def cmd_maxcut(a):
    from .datasets import karate_subgraph, real_graph
    from .problems import MaxCutProblem
    if a.graph == "karate":
        g, names = karate_subgraph(a.nodes)
    else:
        g, names = real_graph("florentine")
    p = MaxCutProblem.from_networkx(g, names)
    q = p.to_qubo()
    best = p.brute_force()
    print(f"{p.n} nodes, {len(p.edges)} links -> {q.n} qubits   (real network data)")
    print(f"classical optimum : cut {best.energy:.0f} of {p.total_weight:.0f} links")
    for r in _solve_all(p, q, a.ibm):
        (side_a, side_b), cut = p.decode(r.bits)
        print(f"{r.solver:20s}: cut {cut:.0f}  (ratio {cut / best.energy:.3f})")
    print("Group A:", ", ".join(map(str, p.decode(best.bits)[0][0])))
    print("Group B:", ", ".join(map(str, p.decode(best.bits)[0][1])))


def cmd_odisha(a):
    from .showcase import odisha_tour_demo
    r = odisha_tour_demo(a.n)
    print(f"{len(r['places'])} places, {r['qubits']} qubits as a QUBO (too many for QAOA simulation)")
    print(f"distances from: {r['distance_source']}")
    print(f"classical exact : {r['exact_tour']}  ({r['exact_km']:.1f} km)")
    print(f"nn + 2-opt      : {r['heuristic_km']:.1f} km")
    print(f"annealing       : " + (f"{r['annealing_km']:.1f} km" if r["annealing_feasible"] else "infeasible"))


def cmd_circuit(a):
    from .quantum import DEVICES, circuit_from_result, estimate
    p = RoutingProblem.from_latlon(dict(list(BBSR_CUTTACK_STOPS.items())[:4]))
    q = p.to_qubo()
    res = solve_qaoa(q, p=a.p)
    c = circuit_from_result(q, res)
    s = c.summary()
    print(f"QAOA circuit for the 4-stop route: {s['qubits']} qubits, p = {a.p}")
    print(f"  gates {s['gates']}  (two-qubit CNOTs {s['two_qubit_gates']}, one-qubit {s['one_qubit_gates']}), depth {s['depth']}")
    print("  hardware estimates (illustrative profiles):")
    for d in DEVICES.values():
        e = estimate(c, d)
        print(f"   {d.name:42s} SWAPs {e['swaps']:4d}  duration {e['duration_us']:9.1f} us  fidelity {e['fidelity']:.4f}")
    if a.qasm:
        with open(a.qasm, "w") as f:
            f.write(c.to_qasm())
        print(f"OpenQASM 2.0 written to {a.qasm}")


def cmd_qml(a):
    from .qml import run_classification
    r = run_classification(a.dataset, n_qubits=a.qubits)
    print(r["dataset"])
    print(f"{r['n_qubits']} qubits, {r['train']} training / {r['test']} test samples, "
          f"PCA keeps {r['pca_variance_kept']:.0%} of variance, bandwidth {r['bandwidth']} (chosen by CV)")
    for k, v in r["results"].items():
        print(f"  {k:22s} {v['accuracy']:.1%}")


def cmd_figures(a):
    from .showcase import make_all
    s = make_all(a.out, quick=a.quick)
    print(f"{len(s['files'])} figures written to {a.out} in {s['seconds']} s")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qoptima", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--quick", action="store_true")
    b.add_argument("--out", default="results")
    r = sub.add_parser("route")
    r.add_argument("--places")
    r.add_argument("--ibm", action="store_true")
    pf = sub.add_parser("portfolio")
    pf.add_argument("--prices")
    pf.add_argument("--k", type=int, default=3)
    pf.add_argument("--risk", type=float, default=1.0)
    pf.add_argument("--ibm", action="store_true")
    pl = sub.add_parser("plots")
    pl.add_argument("--buyers")
    pl.add_argument("--plots")
    pl.add_argument("--ibm", action="store_true")
    mc = sub.add_parser("maxcut")
    mc.add_argument("--graph", choices=["florentine", "karate"], default="florentine")
    mc.add_argument("--nodes", type=int, default=12, help="karate: keep the n best-connected members")
    mc.add_argument("--ibm", action="store_true")
    od = sub.add_parser("odisha")
    od.add_argument("--n", type=int, default=8)
    ci = sub.add_parser("circuit")
    ci.add_argument("--p", type=int, default=2)
    ci.add_argument("--qasm")
    qm = sub.add_parser("qml")
    qm.add_argument("--dataset", choices=["breast_cancer", "wine"], default="breast_cancer")
    qm.add_argument("--qubits", type=int, default=4)
    fg = sub.add_parser("figures")
    fg.add_argument("--out", default="docs/images")
    fg.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "benchmark":
        df = benchmark.run(quick=a.quick, out_dir=a.out)
        print(benchmark.to_markdown(df))
    else:
        {"route": cmd_route, "portfolio": cmd_portfolio, "plots": cmd_plots, "maxcut": cmd_maxcut,
         "odisha": cmd_odisha, "circuit": cmd_circuit, "qml": cmd_qml, "figures": cmd_figures}[a.cmd](a)


if __name__ == "__main__":
    main()
