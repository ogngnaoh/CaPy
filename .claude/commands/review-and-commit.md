Run a complete quality check before committing:

1. Run black src/ tests/ scripts/ and fix any formatting issues
2. Run ruff check src/ tests/ scripts/ --fix and fix any linting issues
3. Run pytest tests/ -v --tb=short and confirm all tests pass
4. Use the code-reviewer agent to review all files changed since last commit
5. Fix any FIX items from the review
6. Generate a conventional commit message (feat/fix/test/chore prefix)
7. Stage all changes and create the commit

Show me the proposed commit message before committing. Do not push.
