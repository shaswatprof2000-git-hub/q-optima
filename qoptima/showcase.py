"""Run the headline experiments and save every figure (used by `python -m qoptima figures`).

All numbers in the README gallery and infographic come from this file, so they
are reproducible: same seeds, same data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import viz
from .datasets import karate_subgraph, odisha_places, real_graph, road_distance_matrix
from .problems import BBSR_CUTTACK_STOPS, MaxCutProblem, RoutingProblem
from .qml import evaluate_transfer, learn_angles, run_classification
from .quantum import DEVICES, bloch_vectors, circuit_from_result, entanglement_entropy, qaoa_circuit
from .quantum.hardware import noise_model
from .solvers import solve_annealing, solve_exact
from .solvers.qaoa import qaoa_state, solve_qaoa


def compare_devices(qubo, result, shots: int = 20000, seed: int = 0) -> list[dict]:
    """Sample the trained circuit under each hardware profile; P(best answer) per device."""
    rng = np.random.default_rng(seed)
    E = qubo.all_energies()
    cost = (E - E.mean()) / (E.std() or 1.0)
    info = result.info
    probs = np.abs(qaoa_state(cost, info["gammas"], info["betas"], qubo.n, info.get("thetas"))) ** 2
    probs /= probs.sum()
    opt = E <= E.min() + 1e-9 * max(1.0, abs(E.min()))
    circ = circuit_from_result(qubo, result)
    rows = []
    for key in ("ideal", "future", "trapped_ion", "superconducting"):
        dev = DEVICES[key]
        nm = noise_model(circ, dev)
        samples = nm.sample(probs, qubo.n, shots, rng, nm.circuit_fidelity(qubo, info["p"]))
        rows.append({"device": dev.name, "key": key, "fidelity": float(nm.fidelity_override),
                     "p_opt": float(np.mean(opt[samples])),
                     "random_guess": float(opt.mean()),
                     "best_found": bool(opt[samples].any())})
    return rows


def _save(fig, out: Path, name: str, saved: list):
    path = out / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    saved.append(str(path))


def make_all(out_dir: str | Path = "docs/images", quick: bool = False, verbose: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    stats: dict = {}
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t0 = time.time()

    # 1. Pipeline schematic
    _save(viz.pipeline_flowchart(), out, "01_pipeline", saved)

    # 2. A 2-qubit toy circuit, readable gate by gate
    from .qubo import QUBO
    toy = QUBO(3)
    toy.add_quadratic(0, 1, 2.0); toy.add_quadratic(1, 2, 2.0)
    toy.add_linear(0, -1.0); toy.add_linear(1, -1.0); toy.add_linear(2, -1.0)
    toy_res = solve_qaoa(toy, p=1, restarts=2)
    toy_circ = circuit_from_result(toy, toy_res)
    _save(viz.draw_circuit(toy_circ, title="QAOA circuit for a 3-qubit problem (p = 1)"), out, "02_circuit_toy", saved)
    _save(viz.draw_circuit(toy_circ.decompose(), title="Same circuit compiled to hardware gates (CNOT + rotations)"),
          out, "03_circuit_compiled", saved)
    log("circuits done")

    # 3. Route on Bhubaneswar-Cuttack stops (16 qubits) + amplification + Bloch spheres
    places = BBSR_CUTTACK_STOPS
    route = RoutingProblem.from_latlon(places)
    rq = route.to_qubo()
    rres = solve_qaoa(rq, p=2, restarts=3)
    tour, km = route.decode(rres.bits)
    tour = tour or list(route.held_karp().bits)
    _save(viz.plot_route(route.names, list(places.values()), tour, km), out, "04_route", saved)
    valid = np.array([route.decode(rq.index_to_bits(k, rq.n))[0] is not None for k in range(2 ** rq.n)])
    _save(viz.plot_amplification(rq, rres, valid, title="QAOA on the 5-stop route: amplified, but most single shots still invalid"),
          out, "05_amplification", saved)
    stats["route_amplification"] = rres.info["amplification"]
    log("route done")

    small = RoutingProblem.from_latlon(dict(list(places.items())[:4]))
    sq = small.to_qubo()
    std_res = solve_qaoa(sq, p=2)
    # Warm starts, compared fairly: from the second-best and the worst tour, plus from a
    # best tour for reference (that one is circular: it starts from the answer).
    import itertools
    tours = sorted(([0, *perm, 0] for perm in itertools.permutations(range(1, small.n))),
                   key=small.tour_length)
    def tour_bits(tour):
        b = np.zeros(sq.n, dtype=int)
        for t, c in enumerate(tour[1:-1], start=1):
            b[small.var(c, t)] = 1
        return b

    lengths = sorted({round(small.tour_length(t), 6) for t in tours})
    by_len = {L: next(t for t in tours if round(small.tour_length(t), 6) == L) for L in lengths}
    opt = sq.all_energies()
    opt_mask = opt <= opt.min() + 1e-9 * max(1.0, abs(opt.min()))
    warm_rows = [{"label": "random guessing", "p_opt": float(opt_mask.mean()), "baseline": True},
                 {"label": "standard QAOA (p = 2)", "p_opt": std_res.info["prob_optimal"], "standard": True}]
    for L, name in [(lengths[1], "second-best"), (lengths[-1], "worst")]:
        r = solve_qaoa(sq, p=2, warm_start=tour_bits(by_len[L]))
        warm_rows.append({"label": f"warm start from the {name} tour ({L:.0f} km)", "p_opt": r.info["prob_optimal"]})
    r = solve_qaoa(sq, p=2, warm_start=tour_bits(by_len[lengths[0]]))
    warm_rows.append({"label": f"warm start from a best tour ({lengths[0]:.0f} km, circular)",
                      "p_opt": r.info["prob_optimal"]})
    _save(viz.plot_warm_start(warm_rows), out, "13_warm_start", saved)
    stats["warm_start"] = warm_rows
    E = sq.all_energies()
    cost = (E - E.mean()) / E.std()
    psi = qaoa_state(cost, std_res.info["gammas"], std_res.info["betas"], sq.n)
    labels = [n.replace("@", " @ step ") for n in sq.var_names]
    _save(viz.plot_bloch(bloch_vectors(psi, sq.n), labels), out, "06_bloch", saved)
    stats["entanglement_bits"] = entanglement_entropy(psi, sq.n)

    # 4. Hardware comparison on the 9-qubit route circuit
    rows = compare_devices(sq, std_res)
    _save(viz.plot_devices(rows, title="9-qubit route circuit on different hardware (illustrative profiles)"),
          out, "07_devices", saved)
    stats["devices"] = rows
    stats["superconducting_fidelity"] = next(r["fidelity"] for r in rows if r["key"] == "superconducting")
    stats["ion_fidelity"] = next(r["fidelity"] for r in rows if r["key"] == "trapped_ion")
    stats["circuit_9q"] = circuit_from_result(sq, std_res).summary()
    log("hardware done")

    # 5. Landscape
    _save(viz.plot_landscape(sq, n_grid=24 if quick else 40,
                             title="QAOA energy landscape for the 4-stop route (p = 1)"), out, "08_landscape", saved)

    # 6. Real network Max-Cut (Florentine families, 15 qubits)
    g, names = real_graph("florentine")
    mc = MaxCutProblem.from_networkx(g, names)
    mres = solve_qaoa(mc.to_qubo(), p=2, restarts=3)
    best_cut = mc.brute_force().energy
    _save(viz.plot_maxcut(mc, mres.bits, g, title=f"Real network: Florentine marriage ties split by QAOA "
                                                  f"(cut {mc.cut_value(mres.bits):.0f}, best possible {best_cut:.0f})"),
          out, "09_maxcut_florentine", saved)
    stats["maxcut_amplification"] = mres.info["amplification"]
    stats["maxcut_cut"] = mc.cut_value(mres.bits)
    stats["maxcut_best"] = best_cut
    log("maxcut done")

    # 7. Scaling
    _save(viz.plot_scaling({"4-stop route": 9, "5-stop route": 16, "Florentine Max-Cut": 15}),
          out, "10_scaling", saved)

    # 8. Quantum-AI: kernel classifier on real medical data
    qml = run_classification("breast_cancer", n_qubits=4 if quick else 6)
    _save(viz.plot_qml(qml), out, "11_quantum_kernel", saved)
    stats["qml"] = {k: v["accuracy"] for k, v in qml["results"].items()}
    stats["qml_bandwidth"] = qml["bandwidth"]
    log("qml done")

    # 9. Quantum-AI: learned angles
    learned = learn_angles(n_nodes=8, p=2, n_train=4 if quick else 6)
    transfer = evaluate_transfer(learned, n_test=3 if quick else 4, n_nodes=10)
    _save(viz.plot_angle_transfer(transfer), out, "12_angle_transfer", saved)
    gap = float(np.mean([100 * (r["ratio_trained"] - r["ratio_transferred"]) for r in transfer]))
    stats["transfer_gap_points"] = gap
    log("transfer done")

    # 10. Infographic
    info_stats = {
        "problems": 4, "max_qubits": 16,
        "best_amplification": max(stats["route_amplification"], stats["maxcut_amplification"]),
        "cnots": stats["circuit_9q"]["two_qubit_gates"],
        "qml_quantum": stats["qml"]["Quantum kernel SVM"],
        "qml_classical": max(v for k, v in stats["qml"].items() if not k.startswith(("Quantum", "Majority"))),
        "transfer_gap": gap,
        "superconducting_fidelity": stats["superconducting_fidelity"],
        "ion_fidelity": stats["ion_fidelity"],
    }
    _save(viz.infographic(info_stats), out, "00_infographic", saved)

    stats["seconds"] = round(time.time() - t0, 1)
    stats["files"] = saved
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
    return stats


def odisha_tour_demo(n: int = 8, seed: int = 0) -> dict:
    """A bigger, real-coordinate route (too big for QAOA simulation): classical vs annealing."""
    places = odisha_places(n)
    d, source = road_distance_matrix(places)
    prob = RoutingProblem.from_matrix(list(places), d)
    exact = prob.held_karp()
    heur = prob.nearest_neighbour_2opt()
    sa = solve_annealing(prob.to_qubo(), num_reads=30, sweeps=800, seed=seed)
    tour, km = prob.decode(sa.bits)
    return {"places": list(places), "distance_source": source, "qubits": prob.to_qubo().n,
            "exact_km": exact.energy, "exact_tour": prob.describe(exact.bits),
            "heuristic_km": heur.energy, "annealing_km": km, "annealing_feasible": tour is not None}


__all__ = ["make_all", "compare_devices", "odisha_tour_demo", "karate_subgraph", "qaoa_circuit", "solve_exact"]
