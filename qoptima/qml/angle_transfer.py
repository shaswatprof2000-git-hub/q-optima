"""Quantum-AI #2: machine learning *for* quantum, i.e. learned QAOA angles.

Training QAOA means many calls to the quantum computer, one per optimiser step.
For problems of the same family, good angles turn out to be very similar
("parameter concentration": Brandao et al., arXiv:1812.04170; Galda et al.,
IEEE QCS 2021). So we can *learn* angles once from a few small training instances
and reuse them on new instances with zero training calls.

This module trains QAOA on random Max-Cut graphs, keeps the angle set that works
best across all of them (a simple learned prior), and measures how close those
transferred angles get compared with full training on each new instance.
"""
from __future__ import annotations

import numpy as np

from ..problems.maxcut import MaxCutProblem
from ..solvers.qaoa import qaoa_state, solve_qaoa


def expected_ratio(problem: MaxCutProblem, gammas, betas) -> float:
    """Expected cut / max cut for given angles (cost normalised as in solve_qaoa)."""
    q = problem.to_qubo()
    E = q.all_energies()
    cost = (E - E.mean()) / (E.std() or 1.0)
    probs = np.abs(qaoa_state(cost, gammas, betas, q.n)) ** 2
    return float(probs @ (-E)) / problem.brute_force().energy


def learn_angles(n_nodes: int = 8, p: int = 2, n_train: int = 6, edge_prob: float = 0.5,
                 seed: int = 0) -> dict:
    """Train on n_train random graphs, then pick the one angle set that does best
    on *all* training graphs (more robust than averaging, which can blend two
    different good solutions into a bad one)."""
    rng = np.random.default_rng(seed)
    problems, candidates = [], []
    for _ in range(n_train):
        prob = MaxCutProblem.random(n_nodes, edge_prob, seed=int(rng.integers(1e9)))
        if not prob.edges:
            continue
        r = solve_qaoa(prob.to_qubo(), p=p, restarts=3, seed=int(rng.integers(1e9)))
        problems.append(prob)
        candidates.append(np.concatenate([r.info["gammas"], r.info["betas"]]))
    candidates.append(np.mean(candidates, axis=0))
    scores = [np.mean([expected_ratio(pr, c[:p], c[p:]) for pr in problems]) for c in candidates]
    best = candidates[int(np.argmax(scores))]
    A = np.array(candidates[:-1])
    return {"p": p, "gammas": best[:p], "betas": best[p:], "train_score": float(max(scores)),
            "spread": A.std(0), "n_train": len(problems)}


def evaluate_transfer(learned: dict, n_test: int = 4, n_nodes: int = 10, edge_prob: float = 0.5,
                      seed: int = 100) -> list[dict]:
    """Compare transferred angles (no training) with full training, on bigger unseen graphs."""
    rng = np.random.default_rng(seed)
    p = learned["p"]
    out = []
    for t in range(n_test):
        prob = MaxCutProblem.random(n_nodes, edge_prob, seed=int(rng.integers(1e9)))
        transferred = expected_ratio(prob, learned["gammas"], learned["betas"])
        r = solve_qaoa(prob.to_qubo(), p=p, restarts=3, seed=t)
        trained = expected_ratio(prob, r.info["gammas"], r.info["betas"])
        out.append({"instance": t, "nodes": n_nodes, "edges": len(prob.edges),
                    "ratio_transferred": transferred, "ratio_trained": trained,
                    "training_calls_saved": r.info["function_evals"]})
    return out
