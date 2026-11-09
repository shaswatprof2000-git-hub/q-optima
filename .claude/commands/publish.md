---
description: Test, commit and push the current work to GitHub
---
1. Run `python -m pytest -q`. Stop and show me the failures if any test fails.
2. Show `git status` and a short summary of the changes, then propose a commit message.
3. After I confirm, commit and `git push`. Then report the commit hash and the GitHub Actions status (`gh run list --limit 1` if `gh` is available).
