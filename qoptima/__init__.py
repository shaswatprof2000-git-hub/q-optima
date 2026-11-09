"""Q-Optima: hybrid quantum-classical optimisation with honest benchmarks."""
from .qubo import QUBO
from .solvers import SolveResult, solve_annealing, solve_exact, solve_qaoa

__version__ = "0.2.0"
__all__ = ["QUBO", "SolveResult", "solve_exact", "solve_annealing", "solve_qaoa"]
