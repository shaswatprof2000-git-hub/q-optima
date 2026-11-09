"""Tests for the v0.2 quantum, data, quantum-AI and graphics layers."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from qoptima import QUBO, solve_annealing, solve_exact, solve_qaoa
from qoptima.problems import BBSR_CUTTACK_STOPS, MaxCutProblem, RoutingProblem
from qoptima.quantum import DEVICES, Circuit, bloch_vectors, circuit_from_result, entanglement_entropy, estimate
from qoptima.solvers.noise import NoiseModel
from qoptima.solvers.qaoa import cvar, interp_next, qaoa_state, warm_start_thetas


@pytest.fixture(scope="module")
def small_route():
    r = RoutingProblem.from_latlon(dict(list(BBSR_CUTTACK_STOPS.items())[:4]))
    return r, r.to_qubo()


def _cost(q):
    E = q.all_energies()
    return (E - E.mean()) / E.std()


# ---- circuits ------------------------------------------------------------------
@pytest.mark.parametrize("warm", [False, True])
def test_gate_level_circuit_matches_fast_simulator(small_route, warm):
    r, q = small_route
    ws = None
    if warm:
        ws = np.zeros(q.n, dtype=int)
        ws[[0, 4, 8]] = 1
    res = solve_qaoa(q, p=2, restarts=1, warm_start=ws)
    psi_fast = qaoa_state(_cost(q), res.info["gammas"], res.info["betas"], q.n, res.info["thetas"])
    c = circuit_from_result(q, res)
    assert abs(np.vdot(psi_fast, c.simulate())) == pytest.approx(1, abs=1e-9)
    assert abs(np.vdot(psi_fast, c.decompose().simulate())) == pytest.approx(1, abs=1e-9)


def test_counts_depth_and_qasm(small_route):
    _, q = small_route
    res = solve_qaoa(q, p=1, restarts=1)
    c = circuit_from_result(q, res)
    s = c.summary()
    h, J, _ = q.to_ising()
    assert s["two_qubit_gates"] == 2 * len(J)          # each RZZ -> 2 CNOTs
    assert 0 < s["depth"] <= s["gates"]
    qasm = c.to_qasm()
    assert qasm.startswith("OPENQASM 2.0;")
    assert qasm.count("\ncx ") == s["two_qubit_gates"]
    assert "measure q -> c;" in qasm


def test_depth_parallelism():
    c = Circuit(3)
    for qb in range(3):
        c.add("h", qb)
    c.add("cx", 0, 1)
    c.add("rz", 2, param=0.3)
    assert c.depth() == 2


def test_swap_estimate_and_hardware_profiles(small_route):
    _, q = small_route
    c = circuit_from_result(q, solve_qaoa(q, p=1, restarts=1))
    assert c.swap_estimate("all") == 0
    assert c.swap_estimate("line") > 0
    ideal = estimate(c, DEVICES["ideal"])
    assert ideal["fidelity"] == pytest.approx(1.0)
    for d in DEVICES.values():
        f = estimate(c, d)["fidelity"]
        assert 0.0 <= f <= 1.0


# ---- states --------------------------------------------------------------------
def test_bloch_vectors_and_entanglement():
    zero = np.array([1, 0], dtype=complex)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert bloch_vectors(zero, 1)[0] == pytest.approx([0, 0, 1])
    assert bloch_vectors(plus, 1)[0] == pytest.approx([1, 0, 0])
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert np.linalg.norm(bloch_vectors(bell, 2)[0]) == pytest.approx(0, abs=1e-12)
    assert entanglement_entropy(bell, 2) == pytest.approx(1.0)
    product = np.kron(plus, zero)
    assert entanglement_entropy(product, 2) == pytest.approx(0.0, abs=1e-12)


# ---- QAOA variants ----------------------------------------------------------------
def test_warm_start_with_equator_angles_is_standard_qaoa(small_route):
    _, q = small_route
    cost = _cost(q)
    thetas = np.full(q.n, np.pi / 2)
    a = qaoa_state(cost, [0.4, 0.7], [0.3, 0.2], q.n)
    b = qaoa_state(cost, [0.4, 0.7], [0.3, 0.2], q.n, thetas)
    assert abs(np.vdot(a, b)) == pytest.approx(1, abs=1e-12)
    assert warm_start_thetas([0, 1], eps=0.25) == pytest.approx(2 * np.arcsin(np.sqrt([0.25, 0.75])))


def test_cvar_limits():
    rng = np.random.default_rng(0)
    cost = rng.normal(size=16)
    probs = rng.random(16)
    probs /= probs.sum()
    order = np.argsort(cost)
    assert cvar(probs, order, cost, 1.0) == pytest.approx(probs @ cost)
    assert cvar(probs, order, cost, 1e-9) == pytest.approx(cost.min())


def test_interp_shapes_and_layerwise(small_route):
    x = interp_next(np.array([0.5, 0.2]), 1)
    assert len(x) == 4
    _, q = small_route
    r = solve_qaoa(q, p=3, layerwise=True, restarts=1)
    assert [d for d, _ in r.info["training_history"]] == [1, 2, 3]


def test_degenerate_optima_are_counted(small_route):
    _, q = small_route
    r = solve_qaoa(q, p=1, restarts=1)
    assert r.info["num_optimal"] == 2            # a route and its reverse
    assert r.info["prob_optimal_random_guess"] == pytest.approx(2 / 2 ** q.n)


def test_noise_reduces_fidelity(small_route):
    _, q = small_route
    nm = NoiseModel()
    assert 0 < nm.circuit_fidelity(q, 2) < 1
    r = solve_qaoa(q, p=1, restarts=1, noise=nm)
    assert r.info["circuit_fidelity"] < 1 and r.solver.endswith("_noisy")
    assert NoiseModel(fidelity_override=0.3).circuit_fidelity(q, 2) == 0.3


# ---- Max-Cut and datasets ---------------------------------------------------------------
def test_maxcut_qubo_matches_enumeration():
    p = MaxCutProblem.random(9, 0.5, seed=4)
    groups, value = p.decode(solve_exact(p.to_qubo()).bits)
    assert value == pytest.approx(p.brute_force().energy)
    assert len(groups[0]) + len(groups[1]) == 9
    assert p.local_search(restarts=3).energy <= p.brute_force().energy


def test_real_datasets_offline():
    from qoptima.datasets import (ODISHA_CITIES, classification_dataset, haversine_matrix, karate_subgraph,
                                  odisha_places, real_graph)
    g, names = real_graph("florentine")
    assert g.number_of_nodes() == 15 and g.number_of_edges() == 20 and "Medici" in names
    k, _ = karate_subgraph(10)
    assert k.number_of_nodes() == 10
    X, y, feats, _ = classification_dataset("breast_cancer")
    assert X.shape == (569, 30) and set(y) == {0, 1}
    places = odisha_places(5)
    assert list(places)[0] == "Bhubaneswar" and len(ODISHA_CITIES) == 15
    d = haversine_matrix(places)
    assert d.shape == (5, 5) and np.allclose(d, d.T) and 15 < d[0, 1] < 35  # Bhubaneswar-Cuttack


def test_road_distances_fall_back_offline(monkeypatch):
    import requests

    from qoptima.datasets import odisha_places, road_distance_matrix

    def boom(*a, **k):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", boom)
    d, src = road_distance_matrix(odisha_places(4))
    assert src == "haversine" and d.shape == (4, 4)


def test_routing_from_matrix_symmetrises():
    p = RoutingProblem.from_matrix(["a", "b", "c"], [[0, 1, 2], [3, 0, 4], [2, 4, 0]])
    assert np.allclose(p.dist, p.dist.T) and p.dist[0, 1] == 2


# ---- quantum-AI ------------------------------------------------------------------------
def test_quantum_kernel_is_valid():
    from qoptima.qml import ZZFeatureMap
    X = np.random.default_rng(1).random((6, 3)) * np.pi
    K = ZZFeatureMap(3).kernel(X)
    assert np.allclose(np.diag(K), 1)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_quantum_kernel_classifier_beats_guessing():
    from qoptima.qml import run_classification
    r = run_classification("breast_cancer", n_qubits=4)
    acc = r["results"]["Quantum kernel SVM"]["accuracy"]
    assert acc > r["results"]["Majority-class guess"]["accuracy"] + 0.2


def test_learned_angles_transfer():
    from qoptima.qml import evaluate_transfer, learn_angles
    learned = learn_angles(n_nodes=6, p=1, n_train=3)
    rows = evaluate_transfer(learned, n_test=2, n_nodes=7)
    for r in rows:
        assert r["ratio_transferred"] > 0.7
        assert r["ratio_transferred"] <= r["ratio_trained"] + 0.05


# ---- graphics smoke tests -------------------------------------------------------------
def test_figures_render(small_route):
    import matplotlib.pyplot as plt

    from qoptima import viz
    r, q = small_route
    res = solve_qaoa(q, p=1, restarts=1)
    figs = [viz.pipeline_flowchart(), viz.draw_circuit(circuit_from_result(q, res)),
            viz.plot_landscape(q, n_grid=6), viz.plot_amplification(q, res),
            viz.plot_scaling({"demo": 9}),
            viz.plot_bloch(bloch_vectors(qaoa_state(_cost(q), res.info["gammas"], res.info["betas"], q.n), q.n)[:2])]
    mc = MaxCutProblem.random(6, 0.6, seed=1)
    figs.append(viz.plot_maxcut(mc, mc.brute_force().bits))
    for f in figs:
        assert f is not None
        plt.close(f)


def test_compare_devices(small_route):
    from qoptima.showcase import compare_devices
    _, q = small_route
    rows = compare_devices(q, solve_qaoa(q, p=1, restarts=1), shots=4000)
    by = {r["key"]: r for r in rows}
    assert by["ideal"]["fidelity"] == pytest.approx(1)
    assert by["ideal"]["p_opt"] >= by["superconducting"]["p_opt"]
