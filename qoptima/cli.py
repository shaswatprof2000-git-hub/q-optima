"""Command line:  python -m qoptima <command>

  benchmark [--quick] [--out results]     run the full comparison table
  route     [--places file.csv]           best visiting order (CSV: name,lat,lon; first row = start)
  portfolio [--prices file.csv] --k 4     pick k stocks (CSV: date + one close-price column per ticker)
  plots     [--buyers b.csv --plots p.csv] match buyers to plots
  Add --ibm to route/portfolio/plots to also sample the QAOA circuit on IBM Quantum hardware.
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
    a = ap.parse_args(argv)

    if a.cmd == "benchmark":
        df = benchmark.run(quick=a.quick, out_dir=a.out)
        print(benchmark.to_markdown(df))
    elif a.cmd == "route":
        cmd_route(a)
    elif a.cmd == "portfolio":
        cmd_portfolio(a)
    else:
        cmd_plots(a)


if __name__ == "__main__":
    main()
