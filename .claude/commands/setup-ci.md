Set up CI and do final code polish:

1. **Create `.github/workflows/ci.yml`:**
   - Trigger on push and pull_request to main
   - Runner: ubuntu-latest
   - Python 3.10
   - Install CPU-only PyTorch and torch-geometric (no CUDA in CI)
   - Install project dependencies from pyproject.toml
   - Steps: `black --check src/ tests/ scripts/`, `ruff check src/ tests/ scripts/`, `pytest tests/ --cov=src --cov-fail-under=80`
   - Add comments explaining the CPU-only PyTorch install

2. **Final formatting pass:**
   - Run `black src/ tests/ scripts/`
   - Run `ruff check src/ tests/ scripts/ --fix`
   - Fix any remaining issues

3. **Final test run:**
   - `pytest tests/ -v --tb=short --cov=src`
   - Confirm all pass, coverage ≥ 80%

4. **Update CLAUDE.md:**
   - Change "Current Phase" to "Phase 4 — Complete"
   - Add any conventions learned during development
