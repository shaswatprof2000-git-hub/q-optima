# Q-Optima benchmark

Generated 2026-11-09 18:51. QAOA runs on an exact statevector simulator (no noise). Gap is measured against the proven optimum from the classical exact method; 0.00 means the optimum was found.

## routing: 4 stops (BBSR-Cuttack) (9 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 60.1803 km | yes | 0.00% | 0.0 |  |
| simulated_annealing | 60.1803 km | yes | 0.00% | 0.112 |  |
| qaoa | 60.1803 km | yes | 0.00% | 0.634 | p=2; P(optimal)=0.088 (22x random guessing) |
| qaoa_noisy | 60.1803 km | yes | 0.00% | 0.632 | p=2; P(optimal)=0.088 (22x random guessing); noisy sampling, circuit fidelity 0.54 |
| classical_exact(held-karp) | 60.1803 km | yes | 0.00% | 0.0 |  |
| classical_heuristic(nn+2opt) | 60.1803 km | yes | 0.00% | 0.0 |  |

## portfolio: 8 stocks, pick 3 (8 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 0.0305 objective | yes | 0.00% | 0.0 |  |
| simulated_annealing | 0.0305 objective | yes | 0.00% | 0.109 |  |
| qaoa | 0.0305 objective | yes | 0.00% | 0.772 | p=2; P(optimal)=0.015 (4x random guessing) |
| qaoa_noisy | 0.0305 objective | yes | 0.00% | 0.655 | p=2; P(optimal)=0.015 (4x random guessing); noisy sampling, circuit fidelity 0.56 |
| classical_exact(enumeration) | 0.0305 objective | yes | 0.00% | 0.0 |  |
| classical_heuristic(greedy) | 0.0348 objective | yes | 14.07% | 0.0 |  |

## plot_matching: 3 buyers x 4 plots (12 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 264.17 fit score | yes | 0.00% | 0.002 |  |
| simulated_annealing | 264.17 fit score | yes | 0.00% | 0.14 |  |
| qaoa | 264.17 fit score | yes | 0.00% | 1.127 | p=2; P(optimal)=0.008 (31x random guessing) |
| qaoa_noisy | 264.17 fit score | yes | 0.00% | 1.123 | p=2; P(optimal)=0.008 (31x random guessing); noisy sampling, circuit fidelity 0.53 |
| classical_exact(hungarian) | 264.17 fit score | yes | 0.00% | 0.0 |  |

## routing: 5 stops (BBSR-Cuttack) (16 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 71.6945 km | yes | 0.00% | 0.035 |  |
| simulated_annealing | 71.6945 km | yes | 0.00% | 0.2 |  |
| qaoa | 71.6945 km | yes | 0.00% | 14.381 | p=2; P(optimal)=0.002 (80x random guessing) |
| classical_exact(held-karp) | 71.6945 km | yes | 0.00% | 0.0 |  |
| classical_heuristic(nn+2opt) | 71.6945 km | yes | 0.00% | 0.0 |  |

## portfolio: 12 stocks, pick 4 (12 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | -0.4905 objective | yes | 0.00% | 0.001 |  |
| simulated_annealing | -0.4905 objective | yes | 0.00% | 0.16 |  |
| qaoa | -0.4905 objective | yes | 0.00% | 1.157 | p=2; P(optimal)=0.001 (5x random guessing) |
| qaoa_noisy | -0.4905 objective | yes | 0.00% | 1.125 | p=2; P(optimal)=0.001 (5x random guessing); noisy sampling, circuit fidelity 0.26 |
| classical_exact(enumeration) | -0.4905 objective | yes | 0.00% | 0.002 |  |
| classical_heuristic(greedy) | -0.4816 objective | yes | 1.83% | 0.0 |  |

## plot_matching: 4 buyers x 4 plots (16 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 297.0 fit score | yes | 0.00% | 0.029 |  |
| simulated_annealing | 297.0 fit score | yes | 0.00% | 0.18 |  |
| qaoa | 297.0 fit score | yes | 0.00% | 14.318 | p=2; P(optimal)=0.001 (93x random guessing) |
| classical_exact(hungarian) | 297.0 fit score | yes | 0.00% | 0.0 |  |

## maxcut: Florentine families network (real data) (15 qubits)

| Solver | Value | Feasible | Gap to optimum | Time (s) | Notes |
|---|---|---|---|---|---|
| exhaustive QUBO check | 17.0 links cut | yes | 0.00% | 0.012 |  |
| simulated_annealing | 17.0 links cut | yes | 0.00% | 0.178 |  |
| qaoa | 17.0 links cut | yes | 0.00% | 4.132 | p=2; P(optimal)=0.074 (243x random guessing) |
| classical_exact(enumeration) | 17.0 links cut | yes | 0.00% | 0.012 |  |
| classical_heuristic(local search) | 17.0 links cut | yes | 0.00% | 0.0 |  |

## Reading these results honestly

- QAOA (simulated, noiseless) found the proven optimum in 7 of 7 instances.
- QAOA with gate & readout noise found the proven optimum in 4 of 4 instances.
- Simulated annealing found the proven optimum in 7 of 7 instances.
- Classical exact methods took at most 0.012s; simulated QAOA took up to 14.4s. No quantum speed-up is claimed: at these sizes classical methods win easily.
- What QAOA does show is *amplification*: after optimisation, the optimal answer is measured many times more often than random guessing (see Notes).
- The research question is how gap and P(optimal) change as qubits grow, and how much real-hardware noise erodes them (sample on IBM Quantum with `--ibm`).
