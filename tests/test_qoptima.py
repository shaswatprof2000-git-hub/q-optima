import numpy as np
import pytest

from qoptima import QUBO, solve_annealing, solve_exact, solve_qaoa
from qoptima.problems import PlotMatchingProblem, PortfolioProblem, RoutingProblem


def random_qubo(n, seed=0):
    rng = np.random.default_rng(seed)
    q = QUBO(n)
    q.Q = np.triu(rng.normal(size=(n, n)))
    q.offset = 0.3
    return q


def test_all_energies_match_pointwise():
    q = random_qubo(6)
    E = q.all_energies()
    for k in range(64):
        assert E[k] == pytest.approx(q.energy(QUBO.index_to_bits(k, 6)))


def test_exactly_k_penalty_is_zero_only_when_satisfied():
    q = QUBO(4)
    q.add_exactly_k([0, 1, 2, 3], 2, penalty=5.0)
    for k in range(16):
        x = QUBO.index_to_bits(k, 4)
        assert q.energy(x) == pytest.approx(5.0 * (x.sum() - 2) ** 2)


def test_ising_conversion_preserves_energy():
    q = random_qubo(5, seed=3)
    h, J, c = q.to_ising()
    for k in range(32):
        x = QUBO.index_to_bits(k, 5)
        z = 1 - 2 * x
        e = c + h @ z + sum(v * z[i] * z[j] for (i, j), v in J.items())
        assert e == pytest.approx(q.energy(x))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_annealing_and_qaoa_find_ground_state_small(seed):
    q = random_qubo(8, seed)
    ex = solve_exact(q)
    assert solve_annealing(q, seed=seed).energy == pytest.approx(ex.energy)
    qa = solve_qaoa(q, p=2, restarts=2, seed=seed)
    assert qa.energy == pytest.approx(ex.energy)
    assert qa.info["amplification"] > 1  # better than random guessing


def test_routing_qubo_optimum_equals_held_karp():
    p = RoutingProblem.random(4, seed=5)
    tour, km = p.decode(solve_exact(p.to_qubo()).bits)
    assert tour is not None
    assert km == pytest.approx(p.held_karp().energy)
    assert p.nearest_neighbour_2opt().energy >= p.held_karp().energy - 1e-9


def test_held_karp_matches_brute_force_on_6_cities():
    import itertools
    p = RoutingProblem.random(6, seed=9)
    brute = min(p.tour_length([0, *perm, 0]) for perm in itertools.permutations(range(1, 6)))
    assert p.held_karp().energy == pytest.approx(brute)


def test_portfolio_qubo_optimum_equals_enumeration():
    p = PortfolioProblem.synthetic(9, 3, seed=2)
    picks, value = p.decode(solve_exact(p.to_qubo()).bits)
    assert picks is not None and len(picks) == 3
    assert value == pytest.approx(p.brute_force().energy)


def test_plot_matching_qubo_optimum_equals_hungarian():
    p = PlotMatchingProblem.sample(3, 4, seed=1)
    pairs, total = p.decode(solve_exact(p.to_qubo()).bits)
    assert pairs is not None
    assert total == pytest.approx(p.hungarian().energy)
    assert len({plot for _, plot in pairs}) == 3  # no plot given twice


def test_decode_rejects_infeasible():
    p = RoutingProblem.random(4)
    assert p.decode(np.zeros(9, dtype=int)) == (None, None)
