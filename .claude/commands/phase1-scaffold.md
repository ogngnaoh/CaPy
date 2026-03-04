Set up the complete CaPy repository structure following PRD section 10.1.

Create:
- All directories: src/data, src/models, src/training, src/evaluation, src/utils, scripts, tests, notebooks, configs, data
- All __init__.py files
- pyproject.toml with exact dependencies from PRD section 10.3
- Dockerfile from PRD section 10.4
- configs/default.yaml with all hyperparameters from PRD section 7.2 (with inline comments)
- src/utils/config.py for YAML loading and seed management
- src/utils/logging.py for wandb/console logging wrapper
- data/README.md explaining data provenance

After creating everything, verify that Python can import the src package.

Then use the architect-explainer agent to explain why this project structure follows ML research best practices: why src/ vs scripts/? Why YAML configs? Why data/raw vs data/processed?
