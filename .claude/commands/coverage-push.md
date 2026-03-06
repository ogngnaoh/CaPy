Push test coverage to ≥80% across src/:

1. Run `pytest tests/ --cov=src --cov-report=term-missing` and identify modules below 80%
2. For each under-covered module (lowest coverage first):
   - Read the source file and identify untested functions/branches
   - Write tests using synthetic data (no real data downloads)
   - Cover edge cases: empty inputs, NaN values, single-element batches, mismatched dimensions
   - Run the test file with `-x` to verify tests pass before moving on
3. Re-run full coverage report after each module
4. Repeat until total coverage ≥ 80%
5. Final verification: `pytest tests/ --cov=src --cov-report=term-missing -v`

Test conventions:
- Test files: tests/test_{module_name}.py
- Use pytest fixtures for shared synthetic data
- Use @pytest.mark.parametrize for testing multiple input sizes
- All tests must run on CPU without real data
