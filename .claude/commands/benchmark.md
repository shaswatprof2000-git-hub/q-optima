---
description: Re-run the full benchmark and report changes versus the committed results
---
1. Run `python -m qoptima benchmark` (about 1 minute). It rewrites `results/benchmark.md` and `results/benchmark.csv`.
2. Run `git diff --stat results/` and compare the new table with the committed one.
3. Report any instance where a solver's gap to the optimum, feasibility or QAOA P(optimal) got worse, and suggest why (penalty weights, seeds, restarts).
4. Don't commit unless I ask. Never describe results as a quantum speed-up.
