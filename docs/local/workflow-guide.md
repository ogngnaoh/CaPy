# CaPy — Claude Code Workflow Guide

Personal reference for using Claude Code's automation features in this project.

---

## Quick Reference: What to Type

| I want to...                          | Type this                                      |
|---------------------------------------|-------------------------------------------------|
| Build the next phase                  | `/phase1-model`, `/phase2-training`, etc.       |
| Develop a feature with guided review  | `/feature-dev <description>`                    |
| Explain a file or concept             | `/explain-this src/models/encoders.py`          |
| Commit with full QA checks            | `/review-and-commit`                            |
| Export to Colab notebook              | `/colab-export`                                 |
| Look up library docs                  | "Check the context7 docs for PyG Batch"         |
| Run data quality checks               | "Run the data-detective agent on processed data"|

---

## Hooks (automatic — no action needed)

Configured in `.claude/settings.json`. Active after every Claude Code restart.

### Auto-format on edit

Every time Claude edits or writes a `.py` file, `black` and `ruff --fix` run
silently on that file. No manual formatting step needed.

- **Trigger:** `Edit` or `Write` on any `.py` file
- **Action:** `black --quiet` then `ruff check --fix --quiet`
- **Result:** Every file is always formatted. The `/review-and-commit` formatting
  step becomes a no-op verification rather than a fix-up.

### Path protection

Prevents accidental edits to gitignored output directories.

- **Trigger:** `Edit` or `Write` targeting `data/raw/`, `data/processed/`, or `checkpoints/`
- **Action:** Blocks the edit with an error message
- **Why:** CLAUDE.md says "Do NOT commit data/ or checkpoints/". This prevents
  Claude from accidentally writing to those directories.

---

## Plugins

### 1. context7 — Live documentation lookup

Gives Claude access to up-to-date docs for any library (torch-geometric, rdkit,
omegaconf, wandb, etc.) instead of relying on training data.

**How it activates:** Automatically when Claude needs library docs. You can also
prompt it directly:

```
Look up the PyG global_mean_pool docs using context7
What does omegaconf OmegaConf.merge do? Check the latest docs
How does wandb.init work with config dicts? Use context7
```

### 2. feature-dev — Guided feature development

A 7-phase workflow for building features with exploration, architecture choices,
and parallel code review.

**Invoke with:** `/feature-dev`

Then describe what to build. The phases:

| Phase | What happens | Your role |
|-------|-------------|-----------|
| 1. Discovery | Clarifies the request, creates tasks | Answer questions |
| 2. Exploration | 2-3 `code-explorer` agents scan the codebase in parallel | Wait |
| 3. Questions | Asks clarifying questions | Answer — don't skip this |
| 4. Architecture | `code-architect` agents propose approaches | Pick one or suggest changes |
| 5. Implementation | Writes code (only after your approval) | Review as it goes |
| 6. Quality Review | 3 `code-reviewer` agents check in parallel | Decide what to fix |
| 7. Summary | Reports what was built | Verify |

**Best for:** Non-trivial features where you want architectural exploration.
Example prompts:

```
/feature-dev implement the MolecularEncoder with 5-layer GIN
/feature-dev add cross-modal retrieval evaluation with Recall@k and MRR
/feature-dev build the training loop with cosine annealing and early stopping
```

### 3. claude-code-setup — Automation recommender

Analyzes the codebase and suggests hooks, MCP servers, skills, subagents.
Already used — run again if you add new tools or libraries.

```
/claude-code-setup:claude-automation-recommender
```

---

## Slash Commands (Phase-based)

These are your main implementation drivers. Each corresponds to a session in
`capy-guide.md`.

| Command | Creates | Status |
|---------|---------|--------|
| `/phase1-scaffold` | Repo structure, pyproject.toml, configs, utils | Done |
| `/phase1-data` | download.py, preprocess.py, featurize.py, dataset.py, tests | Done |
| `/phase1-model` | encoders.py, losses.py, capy.py, test_models.py, test_losses.py | **Next** |
| `/phase2-training` | trainer.py, scheduler.py, retrieval.py, train.py | Pending |
| `/phase2-ablations` | Ablation configs, run_ablations.py, clustering.py | Pending |
| `/phase3-interpret` | interpretability.py, notebooks/03_interpretability.ipynb | Pending |
| `/review-and-commit` | Runs black, ruff, pytest, code-reviewer, then commits | Use before every commit |
| `/explain-this` | Architect-explainer explains a file or concept | Use anytime |
| `/colab-export` | Self-contained Colab notebook | Use after training works |

---

## Agents

These run automatically when invoked by commands or plugins. You can also
request them directly.

| Agent | How to invoke | What it does |
|-------|--------------|-------------|
| `architect-explainer` | `/explain-this <file>` or ask directly | Explains code for ML learners: Math, PyTorch patterns, pitfalls |
| `code-reviewer` | `/review-and-commit` or ask directly | Checks correctness, ML bugs, types, docstrings, test coverage |
| `data-detective` | Ask: "run data-detective on the processed data" | NaN checks, scaffold leakage, compound ID alignment, distributions |

**Asking directly examples:**

```
Use the code-reviewer agent to review src/models/encoders.py
Use the data-detective agent to check the processed parquet files
Use the architect-explainer to explain why GIN is provably as powerful as the WL test
```

---

## Knowledge Skills (Claude-only, automatic)

These are reference documents Claude loads automatically when working on
relevant code. You never invoke them directly.

| Skill | Loaded when | Key content |
|-------|------------|-------------|
| `contrastive-learning-reference` | Working on losses or training | InfoNCE formula, temperature tuning, numerical stability, batch size |
| `pytorch-geometric-patterns` | Working on GIN encoder or graph data | SMILES-to-graph, PyG batching, GINConv patterns, common pitfalls |

---

## Recommended Workflows

### A. Building the next phase (follow the guide)

```
1. Open capy-guide.md to the current session
2. Start a fresh Claude Code session
3. Type the slash command:  /phase1-model
4. Claude builds everything
5. Follow the guide's supervision prompts to ask questions
6. Type:  /review-and-commit
```

The guide is your **reading companion** — it tells you what to watch for and
what questions to ask. The slash command is **what you actually type**.

### B. Building a feature with exploration

```
1. Type:  /feature-dev implement the training loop with early stopping
2. Answer discovery questions
3. Wait for codebase exploration
4. Answer clarifying questions
5. Review architecture proposal, approve or adjust
6. Wait for implementation
7. Review quality report, fix what matters
```

Best when you want Claude to explore options and present trade-offs before
committing to an approach.

### C. Understanding code

```
/explain-this src/data/featurize.py
```

Or ask a specific question:

```
Why does the InfoNCE loss need to be symmetric? Show me the math.
```

### D. Committing clean code

```
/review-and-commit
```

Runs the full QA pipeline: format, lint, test, code review, then creates a
conventional commit. Use this instead of manual `git add/commit`.

### E. Data quality audit

```
Run the data-detective agent on the processed pipeline
```

Use after any changes to preprocessing, or before training to verify data
integrity.

---

## When to Use feature-dev vs Slash Commands

| Situation | Use |
|-----------|-----|
| Following the PRD spec step by step | Slash command (`/phase1-model`) |
| Building something with multiple valid approaches | `/feature-dev` |
| Need architectural exploration before coding | `/feature-dev` |
| Want parallel code review after implementation | `/feature-dev` |
| Quick, well-defined implementation | Slash command |
| Unsure how to approach a problem | `/feature-dev` |

You can mix them: use `/phase1-model` for the bulk, then `/feature-dev` for
tricky additions that need architectural decisions.

---
