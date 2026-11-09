---
description: Regenerate every figure and the infographic, inspect them, and update README numbers
---
1. Run `python -m qoptima figures --out docs/images` (about 30 seconds).
2. Open each PNG in `docs/images/` and check for overlapping labels, clipped text or empty charts. Fix the relevant function in `qoptima/viz.py` if needed and regenerate.
3. Compare `docs/images/stats.json` with the numbers quoted in `README.md` (amplification, accuracies, transfer gap, fidelities). Update the README where they changed.
4. Run `python -m pytest -q`. Show me what changed before committing.
