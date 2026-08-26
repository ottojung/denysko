# Continuous integration

Denysko's required GitHub Actions workflow is `CI`. Its stable test job is `test`, so the status check to require in branch protection is `CI / test`.

The workflow runs on pull requests targeting `master` or `release/**` and on pushes to those branches. It installs the locked project environment with `uv sync --locked` on Python 3.11 and runs the complete suite with `uv run pytest -q`.
