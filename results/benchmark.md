# Q-Optima benchmark

Generated 2026-11-09 14:26. QAOA runs on an exact statevector simulator (no noise). Gap is measured against the proven optimum from the classical exact method; 0.00 means the optimum was found.

## routing: 4 stops (BBSR-Cuttack) (9 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 60.1803 km | yes | 0.00% | 0.0 |  |
| simulated_annealing | 60.1803 km | yes | 0.00% | 0.121 |  |
| qaoa | 60.1803 km | yes | 0.00% | 0.75 | p=2; P(optimal)=0.044 (22x random guessing) |
| classical_exact(held-karp) | 60.1803 km | yes | 0.00% | 0.0 |  |
| classical_heuristic(nn+2opt) | 60.1803 km | yes | 0.00% | 0.0 |  |

## portfolio: 8 stocks, pick 3 (8 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 0.0305 objective | yes | 0.00% | 0.0 |  |
| simulated_annealing | 0.0305 objective | yes | 0.00% | 0.115 |  |
| qaoa | 0.0305 objective | yes | 0.00% | 0.723 | p=2; P(optimal)=0.015 (4x random guessing) |
| classical_exact(enumeration) | 0.0305 objective | yes | 0.00% | 0.0 |  |
| classical_heuristic(greedy) | 0.0348 objective | yes | 14.07% | 0.0 |  |

## plot_matching: 3 buyers x 4 plots (12 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 264.17 fit score | yes | 0.00% | 0.002 |  |
| simulated_annealing | 264.17 fit score | yes | 0.00% | 0.145 |  |
| qaoa | 264.17 fit score | yes | 0.00% | 1.159 | p=2; P(optimal)=0.009 (38x random guessing) |
| classical_exact(hungarian) | 264.17 fit score | yes | 0.00% | 0.0 |  |

## routing: 5 stops (BBSR-Cuttack) (16 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 71.6945 km | yes | 0.00% | 0.038 |  |
| simulated_annealing | 71.6945 km | yes | 0.00% | 0.187 |  |
| qaoa | 71.6945 km | yes | 0.00% | 16.726 | p=2; P(optimal)=0.001 (79x random guessing) |
| classical_exact(held-karp) | 71.6945 km | yes | 0.00% | 0.0 |  |
| classical_heuristic(nn+2opt) | 71.6945 km | yes | 0.00% | 0.0 |  |

## portfolio: 12 stocks, pick 4 (12 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | -0.4905 objective | yes | 0.00% | 0.001 |  |
| simulated_annealing | -0.4905 objective | yes | 0.00% | 0.213 |  |
| qaoa | -0.4905 objective | yes | 0.00% | 1.175 | p=2; P(optimal)=0.001 (5x random guessing) |
| classical_exact(enumeration) | -0.4905 objective | yes | 0.00% | 0.002 |  |
| classical_heuristic(greedy) | -0.4816 objective | yes | 1.83% | 0.0 |  |

## plot_matching: 4 buyers x 4 plots (16 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 297.0 fit score | yes | 0.00% | 0.033 |  |
| simulated_annealing | 297.0 fit score | yes | 0.00% | 0.226 |  |
| qaoa | 297.0 fit score | yes | 0.00% | 16.646 | p=2; P(optimal)=0.001 (92x random guessing) |
| classical_exact(hungarian) | 297.0 fit score | yes | 0.00% | 0.0 |  |

## Reading these results honestly

- QAOA (simulated) found the proven optimum in 6 of 6 instances.
- Simulated annealing found the proven optimum in 6 of 6 instances.
- Classical exact methods took at most 0.002s; simulated QAOA took up to 16.7s. No quantum speed-up is claimed: at these sizes classical methods win easily.
- What QAOA does show is *amplification*: after optimisation, the optimal answer is measured many times more often than random guessing (see Notes).
- The research question is how gap and P(optimal) change as qubits grow, and how much real-hardware noise erodes them (sample on IBM Quantum with `--ibm`).
