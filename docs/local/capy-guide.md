# CaPy × Claude Code: Complete Implementation Guide

### From Empty Directory to Portfolio-Ready ML Research Project

**Project:** CaPy — Tri-Modal Contrastive Learning for Drug Discovery
**Tool:** Claude Code with Opus 4.6 (Max plan)
**Timeline:** 8 weeks, ~20 sessions
**Author's ML Level:** Beginner (strong CS/math background)

---

## How to Use This Guide

This guide is structured as a session-by-session walkthrough. Each session represents one sitting with Claude Code (typically 30–90 minutes). For every session you will find:

- **Goal** — what you'll have by the end
- **Model** — whether to use Opus or Sonnet for that session
- **Exact prompts** — text you type into Claude Code, in blockquotes
- **Adjustment notes** — when and how to modify the prompts
- **Checkpoints** — how to verify the session succeeded
- **Learning moments** — concepts the architect-explainer will teach you

Read the guide end-to-end once before starting. Then use it as a reference, one session at a time.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Session 0 — Install Claude Code and Dependencies](#2-session-0--install-claude-code-and-dependencies)
3. [Session 1 — Bootstrap the Workspace](#3-session-1--bootstrap-the-workspace)
4. [Session 2 — Install Plugins](#4-session-2--install-plugins)
5. [Session 3 — Scaffold the Repository](#5-session-3--scaffold-the-repository)
6. [Session 4 — Build the Data Pipeline](#6-session-4--build-the-data-pipeline)
7. [Session 5 — Build the Model Architecture](#7-session-5--build-the-model-architecture)
8. [Session 6 — Integration Test and First Commit](#8-session-6--integration-test-and-first-commit)
9. [Session 7 — Build the Training Infrastructure](#9-session-7--build-the-training-infrastructure)
10. [Session 8 — Export to Colab and First Training Run](#10-session-8--export-to-colab-and-first-training-run)
11. [Session 9 — Debug, Tune, and Stabilize Training](#11-session-9--debug-tune-and-stabilize-training)
12. [Session 10 — Ablation Study Infrastructure](#12-session-10--ablation-study-infrastructure)
13. [Session 11 — Run Ablations and Analyze Results](#13-session-11--run-ablations-and-analyze-results)
14. [Session 12 — MOA Clustering and Retrieval Evaluation](#14-session-12--moa-clustering-and-retrieval-evaluation)
15. [Session 13 — UMAP Visualizations and Results Notebook](#15-session-13--umap-visualizations-and-results-notebook)
16. [Session 14 — Interpretability Analysis](#16-session-14--interpretability-analysis)
17. [Session 15 — Biology Notebook and GSEA](#17-session-15--biology-notebook-and-gsea)
18. [Session 16 — Test Coverage Push](#18-session-16--test-coverage-push)
19. [Session 17 — README, Documentation, and Docker](#19-session-17--readme-documentation-and-docker)
20. [Session 18 — GitHub Actions CI and Final Polish](#20-session-18--github-actions-ci-and-final-polish)
21. [Session 19 — Stretch Goals](#21-session-19--stretch-goals)
22. [Session 20 — Final Review and Ship](#22-session-20--final-review-and-ship)
23. [Appendix A — All Configuration Files](#appendix-a--all-configuration-files)
24. [Appendix B — Troubleshooting Common Issues](#appendix-b--troubleshooting-common-issues)
25. [Appendix C — Prompt Patterns and Techniques](#appendix-c--prompt-patterns-and-techniques)

---

## 1. Prerequisites

Before Session 0, make sure you have the following on your local machine:

- **Node.js 18 or later** — required to install Claude Code. Check with `node --version`.
- **Python 3.10 or later** — the project runtime. Check with `python3 --version`.
- **Git** — for version control. Check with `git --version`.
- **A Claude Max subscription** — this gives you extended Opus 4.6 access, which is essential for the subagent-heavy workflow.
- **A Google account** — for Google Colab (GPU training). Colab Pro is recommended but not required.
- **A GitHub account** — for hosting the repository.
- **A Weights & Biases account** — free tier at wandb.ai. You'll need your API key.

Optional but recommended:
- **VS Code** — useful for browsing files alongside Claude Code in the terminal.
- **pyright** — `npm install -g pyright` — the Python language server for the LSP plugin.

---

## 2. Session 0 — Install Claude Code and Dependencies

**Goal:** Claude Code installed, authenticated, and working in your terminal.
**Duration:** 15 minutes.
**Model:** N/A (installation only).

### Step 1: Install Claude Code

Open your terminal and run:

```
npm install -g @anthropic-ai/claude-code
```

Verify it installed:

```
claude --version
```

### Step 2: Authenticate

```
claude auth login
```

This opens a browser window. Log in with your Anthropic account (the one with Max). Follow the prompts to complete authentication.

### Step 3: Install pyright for the LSP plugin

```
npm install -g pyright
```

### Step 4: Install Python dependencies you'll need globally

```
pip install black ruff pytest --break-system-packages
```

> **Adjustment note:** If you use a Python version manager (pyenv, conda), install these into your active environment instead.

### Step 5: Create the project directory

```
mkdir capy && cd capy && git init
```

### Checkpoint

Run `claude` from inside the `capy/` directory. You should see Claude Code start up with a prompt. Type "hello" and confirm you get a response. Then type `exit` to close the session.

---

## 3. Session 1 — Bootstrap the Workspace

**Goal:** All Claude Code configuration files created — CLAUDE.md, subagents, slash commands, skills, hooks.
**Duration:** 30 minutes.
**Model:** Sonnet (this is file creation, not architecture thinking).

### Step 1: Start Claude Code

```
cd capy
claude
```

### Step 2: Switch to Sonnet to conserve Opus quota

> `/model sonnet`

### Step 3: Create the directory structure

Type this prompt exactly:

> Create the following directory structure for my Claude Code configuration. Create every file listed below with the exact content I provide. Do not modify or add anything — use my content verbatim.
>
> First, create these directories:
> .claude/agents/
> .claude/commands/
> .claude/skills/

Claude will create the directories. Then proceed to create each file one by one. For each file below, type:

> Create the file `[path]` with exactly this content:

Then paste the content from Appendix A of this guide. Create the files in this order:

1. `CLAUDE.md` (project root) — paste from Appendix A, Section A1
2. `.claude/agents/architect-explainer.md` — paste from Section A2
3. `.claude/agents/code-reviewer.md` — paste from Section A3
4. `.claude/agents/data-detective.md` — paste from Section A4
5. `.claude/commands/phase1-scaffold.md` — paste from Section A5
6. `.claude/commands/phase1-data.md` — paste from Section A6
7. `.claude/commands/phase1-model.md` — paste from Section A7
8. `.claude/commands/phase2-training.md` — paste from Section A8
9. `.claude/commands/phase2-ablations.md` — paste from Section A9
10. `.claude/commands/phase3-interpret.md` — paste from Section A10
11. `.claude/commands/review-and-commit.md` — paste from Section A11
12. `.claude/commands/explain-this.md` — paste from Section A12
13. `.claude/commands/colab-export.md` — paste from Section A13
14. `.claude/skills/pytorch-geometric-patterns.md` — paste from Section A14
15. `.claude/skills/contrastive-learning-reference.md` — paste from Section A15
16. `SCRATCHPAD.md` — paste from Section A16
17. `.gitignore` — paste from Section A17

> **Adjustment note:** If you want to go faster, you can paste the entire bootstrap shell script from the previous deliverable into your terminal instead. But doing it file-by-file lets you read and understand each piece as you go — recommended for a first-time setup.

### Step 4: Verify the workspace

> Show me the directory structure of .claude/ and confirm all files exist. Also cat the CLAUDE.md and confirm it starts with "# CaPy Project".

### Checkpoint

Claude should show a tree with 3 agent files, 9 command files, 2 skill files, plus the root CLAUDE.md, SCRATCHPAD.md, and .gitignore. If anything is missing, ask Claude to create it.

### Step 5: Initial commit

> Initialize git, stage all files, and commit with the message "chore: bootstrap Claude Code workspace configuration"

---

## 4. Session 2 — Install Plugins

**Goal:** All marketplace plugins installed and verified.
**Duration:** 15 minutes.
**Model:** Sonnet (administrative task).

### Step 1: Start a fresh Claude Code session

```
cd capy
claude
```

### Step 2: Add the scientific skills marketplace

> /plugin marketplace add K-Dense-AI/claude-scientific-skills

> **Adjustment note:** If this marketplace has changed its name or location by the time you read this, search for it with `/plugin marketplace add` and look for "scientific skills" or "RDKit" in the descriptions.

### Step 3: Install official plugins

Run each of these one at a time, waiting for confirmation between each:

> /plugin install pyright@claude-plugin-directory

> /plugin install commit-commands@claude-plugin-directory

> /plugin install code-review@claude-plugin-directory

> /plugin install explanatory@claude-plugin-directory

> /plugin install feature-dev@claude-plugin-directory

> /plugin install claudemd-tools@claude-plugin-directory

> **Adjustment note:** Plugin names may have changed since this guide was written. If a name doesn't resolve, run `/plugin` and go to the Discover tab to browse available plugins. Look for the functionality described (Python LSP, git commits, code review, educational explanations, feature development workflow, CLAUDE.md maintenance).

### Step 4: Verify installation

> /plugin

Navigate to the Installed tab using Tab. Confirm you see all 6 plugins plus whichever scientific skills you installed. Check the Errors tab — if any plugins show errors, note the error message and address it (usually a missing binary like pyright).

### Step 5: Test the LSP

Create a quick test file to verify pyright is working:

> Create a file called test_lsp.py with a simple Python function that has a deliberate type error, like passing a string where an int is expected. Then check if pyright catches it.

If Claude reports a diagnostic from pyright, the LSP is working. Delete the test file afterward.

### Checkpoint

Six plugins installed with no errors. Pyright LSP responding to type issues. You're ready for real development.

---

## 5. Session 3 — Scaffold the Repository

**Goal:** Complete project directory structure, pyproject.toml, Dockerfile, all __init__.py files, default config YAML.
**Duration:** 45 minutes.
**Model:** Start with Sonnet for boilerplate, switch to Opus for the config and Dockerfile.

### Step 1: Start a fresh session

```
cd capy
claude
```

Claude will read CLAUDE.md automatically. You should see it acknowledge the CaPy project context.

### Step 2: Run the scaffold command

> /phase1-scaffold

> **What this does:** Claude reads your slash command file, which instructs it to create the full repo structure from PRD section 10.1, set up pyproject.toml, Dockerfile, configs, and all init files. It will also invoke the architect-explainer subagent at the end.

> **Adjustment note:** If Claude asks clarifying questions (like "what Python version?"), answer with: Python 3.10+, PyTorch 2.1+, use the exact dependencies from the PRD section 10.3.

### Step 3: While Claude works, watch for these specific files

Claude should create all of the following. If it misses any, prompt it:

- `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/training/__init__.py`, `src/evaluation/__init__.py`, `src/utils/__init__.py`
- `src/utils/config.py` — YAML config loading + seed setting
- `src/utils/logging.py` — wandb/console logging wrapper
- `pyproject.toml` — with all dependencies from the PRD
- `Dockerfile` — based on pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
- `configs/default.yaml` — all hyperparameters with comments
- `scripts/` directory with empty placeholder files
- `tests/` directory with empty placeholder files
- `notebooks/` directory
- `data/README.md` — explaining data provenance and download steps

### Step 4: Review the config file

This is critical — the config file defines every hyperparameter. Switch to Opus for this review:

> /model opus

> Show me the contents of configs/default.yaml. I want to verify every hyperparameter matches the PRD section 7.2. Walk me through each parameter and what it controls.

> **What to look for:** Learning rate 1e-3 for encoders and 1e-4 for GIN, batch size 128, epochs 200, patience 20, embedding dim 256, dropout 0.1, gradient clip 1.0, temperature init 0.07, lambda weights all 1.0.

### Step 5: Learning moment — the architect explains the scaffold

The slash command should have triggered the architect-explainer. If it didn't, ask directly:

> Use the architect-explainer agent to explain why this project structure follows ML research best practices. Specifically: why separate src/ from scripts/? Why configs as YAML? Why the data/raw vs data/processed split?

### Step 6: Verify imports work

> Run this command in bash: `cd /path/to/capy && python -c "import src; print('src imports OK')"` and make sure there are no errors.

> **Adjustment note:** Replace `/path/to/capy` with your actual path, or just use the relative path if Claude is already in the right directory.

### Checkpoint

- Complete directory tree matching PRD section 10.1
- pyproject.toml with all dependencies
- Dockerfile that references the correct base image
- configs/default.yaml with all hyperparameters matching the PRD
- All __init__.py files present
- Python imports work without errors

### Step 7: Commit

> /review-and-commit

Or if you installed the commit-commands plugin:

> /commit-commands:commit

---

## 6. Session 4 — Build the Data Pipeline

**Goal:** Complete data download, preprocessing, featurization, and dataset classes with tests.
**Duration:** 90 minutes (the longest session — this is the foundation).
**Model:** Opus throughout (complex implementation with many design decisions).

This is the most important session. The data pipeline determines whether everything downstream works. Take your time.

### Step 1: Start a fresh session with Opus

```
cd capy
claude
```

> /model opus

### Step 2: Run the data pipeline command

> /phase1-data

> **What this does:** Claude reads your slash command which specifies all 5 files to create (download.py, preprocess.py, featurize.py, dataset.py, test_data.py) and the architect-explainer follow-up.

### Step 3: Supervise the download module

Claude will write `src/data/download.py` first. Watch for:

- It should attempt S3 download with a fallback to direct HTTP links
- It should use tqdm for progress bars
- It should save to `data/raw/`
- It should handle the case where data already exists (skip re-download)

If Claude tries to actually download the data (which will fail in a local CPU-only environment without AWS CLI), interrupt:

> Don't actually run the download now — just write the module. We'll run it later when we have the right environment set up. For now, make sure the download function has both S3 and direct HTTP fallback paths, and include a --dry-run flag.

### Step 4: Pay attention to preprocessing

This is where the most design decisions happen. Claude will write `src/data/preprocess.py`. Watch for these critical steps:

**Compound matching:** Claude needs to join morphology and expression tables by compound ID. Ask it to explain:

> Before you implement the compound matching, explain to me: how are compounds identified in the Rosetta dataset? What column do we join on? What happens to compounds that appear in one modality but not the other?

**Feature QC:** The PRD specifies removing zero-variance features and features with >50% NaN. Make sure Claude implements both.

**Scaffold splitting:** This is the most important data decision. After Claude implements it, demand an explanation:

> Stop. Use the architect-explainer to explain scaffold splitting to me. Why can't we do a random train/test split? What is a Bemis-Murcko scaffold? What happens if structurally similar molecules end up in both train and test?

> **What you should learn:** Random splitting would let the model "cheat" by memorizing molecular structure patterns that appear in both sets. Scaffold splitting ensures that molecules in the test set have different core structures than training molecules, forcing the model to generalize.

### Step 5: Supervise featurization

Claude will write `src/data/featurize.py`. This converts SMILES strings to PyTorch Geometric graph objects. Key things to verify:

> After writing featurize.py, explain to me: what are the 9 node features for each atom? Why do we need edge features? Why does edge_index need to be a Long tensor and not Float?

> **What you should learn:** Node features encode what each atom IS (element, charge, ring membership, etc.). Edge features encode what each bond IS (single/double/triple, aromatic). edge_index must be Long because it contains integer indices, not continuous values.

### Step 6: Supervise the dataset class

Claude will write `src/data/dataset.py`. The tricky part is batching — PyG molecular graphs are variable-size, so they can't be stacked like regular tensors.

> After writing dataset.py, explain to me: how does PyG's DataLoader handle graphs of different sizes? What is the `batch` vector? Why can't we just use a regular PyTorch DataLoader?

> **What you should learn:** PyG creates one big disconnected graph from the entire batch, with a `batch` tensor that maps each node back to its original graph. `global_mean_pool(x, batch)` then averages node features per graph. This is much more efficient than padding all graphs to the same size.

### Step 7: Verify tests pass

Claude should have written `tests/test_data.py`. Ask it to run:

> Run pytest tests/test_data.py -v and show me the results. If any tests require actual data (which we haven't downloaded yet), make sure those tests are marked with @pytest.mark.skipif or use synthetic fixture data.

> **Adjustment note:** Tests should use small synthetic data (10 fake compounds, 50 features each) rather than requiring the real Rosetta download. If Claude wrote tests that need real data, ask it to refactor them to use fixtures.

### Step 8: Use the data detective

> Use the data-detective agent to review everything we just built. Have it check: does the preprocessing pipeline handle edge cases? Are there any places where NaN could slip through? Is the scaffold split deterministic (seeded)?

### Checkpoint

- `src/data/download.py` — download with S3 + HTTP fallback
- `src/data/preprocess.py` — QC, normalization, scaffold split
- `src/data/featurize.py` — SMILES to PyG graphs
- `src/data/dataset.py` — CaPyDataset returning triples
- `tests/test_data.py` — all tests pass (on synthetic data)
- You can explain scaffold splitting and PyG batching in your own words

### Step 9: Commit

> /review-and-commit

Expected commit message: `feat: implement data pipeline with scaffold splitting and PyG featurization`

---

## 7. Session 5 — Build the Model Architecture

**Goal:** All three encoders, InfoNCE loss, and the full CaPy model with tests.
**Duration:** 75 minutes.
**Model:** Opus (critical architecture decisions).

### Step 1: Fresh session, Opus model

```
cd capy
claude
```

> /model opus

### Step 2: Run the model command

> /phase1-model

### Step 3: Supervise the molecular encoder

Claude will write the GIN-based MolecularEncoder in `src/models/encoders.py`. This is the most complex encoder. Key verification points:

After Claude writes it, ask:

> Walk me through the forward pass of the MolecularEncoder step by step. For a batch of 32 molecules, where the largest molecule has 45 atoms, what is the shape of the tensor at each stage? Start from the input Data object and end at the 256-dim output.

> **What you should learn:** Input node features are [total_atoms_in_batch, feat_dim], not [32, max_atoms, feat_dim]. After GIN layers and global_mean_pool with the batch vector, you get [32, hidden_dim]. After the projection MLP and L2 normalization, you get [32, 256] on the unit hypersphere.

### Step 4: Verify the tabular encoders

The morphology and expression encoders should be instances of the same TabularEncoder class, just with different input dimensions. Verify:

> Show me the TabularEncoder class. Confirm it has: 3 MLP layers with LayerNorm and ReLU, residual connections on layers 2 and 3 (not layer 1, since the dimensions change), dropout of 0.1, and a projection head that outputs 256-dim L2-normalized vectors.

> **Adjustment note:** If Claude puts residual connections on all layers including the first (where input_dim ≠ hidden_dim), ask it to fix this. Residual connections require matching dimensions. The first layer needs to project from input_dim to hidden_dim, so it can't have a skip connection unless a linear projection is added to the skip path.

### Step 5: Deep dive on InfoNCE loss

This is the mathematical heart of the project. After Claude writes `src/models/losses.py`:

> Use the architect-explainer to give me a thorough explanation of InfoNCE. Specifically: what is the similarity matrix? Why do we divide by temperature? Why is the loss symmetric? What does "log(N) is the minimum, not zero" mean? And show me where numerical stability matters.

> **What you should learn:** For a batch of N=128 samples, the similarity matrix is [128, 128] where entry [i,j] is the cosine similarity between modality A's embedding for compound i and modality B's embedding for compound j. The diagonal entries are the correct matches. Temperature sharpens or smooths this distribution. Symmetry means we compute the loss in both directions (A→B and B→A) and average. The minimum loss is log(N) because even a perfect model faces a 1/N chance per sample in the cross-entropy.

### Step 6: Verify the full CaPy model

After Claude writes `src/models/capy.py`:

> Show me the CaPy model. Confirm: it has all 3 encoders, a learnable log_temperature parameter initialized to log(1/0.07), and a compute_loss method that returns the weighted sum of 3 pairwise InfoNCE losses plus a dictionary of individual losses. Also confirm temperature is clamped to prevent collapse.

> **Critical check:** The temperature should be clamped (e.g., `temp = self.log_temperature.exp().clamp(min=0.01, max=10.0)`). Without this, training can become unstable.

### Step 7: Run shape tests

> Run pytest tests/test_models.py -v. The tests should verify: forward pass output shapes are [batch_size, 256] for all encoders, all outputs are L2-normalized (torch.norm ≈ 1.0), gradients flow through all parameters, and the model works on both CPU and (mock) CUDA.

### Step 8: Run loss tests

> Run pytest tests/test_losses.py -v. The tests should verify: loss for perfect matching is approximately log(batch_size), loss for random embeddings is higher than perfect matching, loss is symmetric (swapping z_a and z_b gives the same value), and the function handles batch_size=1 gracefully.

### Checkpoint

- `src/models/encoders.py` — MolecularEncoder (GIN), TabularEncoder, MorphologyEncoder, ExpressionEncoder
- `src/models/losses.py` — symmetric InfoNCE with numerical stability
- `src/models/capy.py` — full CaPy model with compute_loss
- `tests/test_models.py` — all shape and gradient tests pass
- `tests/test_losses.py` — all loss sanity checks pass
- You can explain GIN, InfoNCE, and why temperature is learnable

### Step 9: Commit

> /review-and-commit

Expected message: `feat: implement tri-modal CaPy model with GIN encoder and InfoNCE loss`

---

## 8. Session 6 — Integration Test and First Commit

**Goal:** Verify end-to-end data→model→loss→backward works. Clean up any issues. Tag the Phase 1 milestone.
**Duration:** 45 minutes.
**Model:** Opus for the integration test, Sonnet for cleanup.

### Step 1: Fresh session

```
cd capy
claude
```

> /model opus

### Step 2: End-to-end integration test

> Write a test in tests/test_integration.py that does the following: create a small synthetic dataset with 32 samples (fake morphology vectors of dim 100, fake expression vectors of dim 50, and fake molecular graphs with 5-10 atoms each). Create a CaPy model configured for these dimensions. Run one forward pass. Compute the loss. Call loss.backward(). Verify gradients exist on all parameters. Verify the loss is a finite scalar. This tests the entire pipeline from data to gradient.

> **Adjustment note:** The synthetic dimensions (100 morph, 50 expr) are deliberately smaller than real data (~1000 morph, 978 expr) so the test runs fast. The test should use the same CaPy model class but with input_dim overrides.

### Step 3: Run the integration test

> Run pytest tests/test_integration.py -v and show me the full output.

If it fails, Claude will debug. Common issues at this stage:

- **Shape mismatches** between the dataset and model — usually the input_dim config doesn't match the synthetic data dimensions
- **Device mismatches** — some tensors on CPU, some on CUDA
- **PyG batching issues** — the collate function needs to handle the triple of (graph, morph_vector, expr_vector)

Let Claude fix these. Each fix is a learning opportunity:

> When you fix this, explain what went wrong and why. I want to understand the root cause, not just the fix.

### Step 4: Run the full test suite

> Run pytest tests/ -v --tb=short and show me a summary. How many tests total, how many pass, how many fail?

Target: all tests pass. If any fail, fix them now.

### Step 5: Code review

> Use the code-reviewer agent to review all files in src/models/ and src/data/. Focus on ML-specific bugs.

Address any ❌ FIX items. ⚠️ WARN items can be noted in the SCRATCHPAD for later.

### Step 6: Update the scratchpad

> Update SCRATCHPAD.md with notes about what we've built so far. Include: the number of synthetic features used in tests, any tricky bugs we hit, and any decisions we made that differed from the PRD.

### Step 7: Tag the milestone

> /review-and-commit

Then manually (or ask Claude):

> Create a git tag called "phase1-complete" with the message "Phase 1: Foundation complete — data pipeline + model architecture"

### Checkpoint — Phase 1 Complete

At this point you should have:
- Complete data pipeline (download, preprocess, featurize, dataset)
- Complete model (3 encoders, InfoNCE loss, CaPy wrapper)
- Integration test proving end-to-end forward/backward works
- All tests passing
- Clean git history with conventional commits
- Understanding of: scaffold splitting, GIN, InfoNCE, PyG batching, learnable temperature

---

## 9. Session 7 — Build the Training Infrastructure

**Goal:** Training loop with wandb logging, scheduler, retrieval metrics, and the train.py entrypoint.
**Duration:** 90 minutes.
**Model:** Opus (training loops are subtle and bug-prone).

### Step 1: Fresh session

```
cd capy
claude
```

> /model opus

### Step 2: Run the training command

> /phase2-training

### Step 3: Supervise the trainer

Claude will write `src/training/trainer.py`. This is the core training loop. Key things to verify as it works:

**Separate parameter groups:**

> After writing the trainer, show me how the optimizer is configured. I need to see separate parameter groups: one for the GIN encoder with lr=1e-4, and one for everything else with lr=1e-3. Both should use AdamW with weight_decay=1e-4.

> **What you should learn:** GNNs are more sensitive to learning rate than MLPs. A lower LR for the GIN prevents the molecular encoder from diverging early in training while the MLP encoders are still finding a good representation.

**Validation loop:**

> Show me the validation step. Confirm: model.eval() is called, torch.no_grad() context is used, retrieval metrics are computed on the entire validation set (not per-batch), and model.train() is restored after.

> **Critical check:** Forgetting model.eval() during validation is one of the most common ML bugs. It means dropout and batch norm behave differently than they should during evaluation.

**Early stopping:**

> Explain the early stopping logic. What metric are we monitoring? What is the patience? What happens when we save a checkpoint?

### Step 4: Supervise the retrieval metrics

Claude will write `src/evaluation/retrieval.py`. This computes the primary evaluation metrics.

> After writing retrieval.py, explain to me: for the direction mol→morph, what exactly happens? Walk me through: we take the molecular embedding for compound i, compute cosine similarity against ALL morphology embeddings, rank them, and check if compound i's morphology embedding is in the top-k. Do this for all compounds and average.

> **What you should learn:** Retrieval Recall@10 means "what fraction of the time is the correct match in the top 10 results?" MRR (Mean Reciprocal Rank) gives a smoother signal — if the correct answer is rank 3, MRR contributes 1/3.

### Step 5: Verify the scheduler

> Show me the cosine annealing scheduler with warmup. For a training run of 200 epochs with 10 epochs of warmup, plot (describe) what the learning rate looks like over time.

> **What you should learn:** Warmup linearly increases LR from 0 to the target over 10 epochs, preventing large gradients early when the model is randomly initialized. Then cosine annealing slowly decays the LR, which helps fine-tune in later epochs.

### Step 6: Verify the train.py entrypoint

> Show me scripts/train.py. Confirm it: loads the config from a YAML file specified by --config, seeds all random sources (torch, numpy, random, torch_geometric), initializes wandb with the config, builds the dataset and dataloaders, builds the model and optimizer, calls the trainer, and saves final results.

### Step 7: Dry-run the training script

> Run a quick sanity check: can you execute `python scripts/train.py --config configs/default.yaml --dry-run` (or equivalent) that loads the config, builds the model, creates one fake batch, and runs one forward+backward pass? We don't need real data — use the synthetic data approach from our integration test.

> **Adjustment note:** If the train.py doesn't have a --dry-run flag, ask Claude to add one. This is invaluable for verifying the training pipeline works without needing the full dataset.

### Checkpoint

- `src/training/trainer.py` — full training loop with wandb, early stopping, checkpointing
- `src/training/scheduler.py` — cosine annealing with warmup
- `src/evaluation/retrieval.py` — 6-direction retrieval metrics
- `scripts/train.py` — main entrypoint
- `configs/default.yaml` — updated with any new parameters
- Dry-run training works end-to-end

### Step 8: Commit

> /review-and-commit

Expected message: `feat: implement training loop with wandb logging and retrieval evaluation`

---

## 10. Session 8 — Export to Colab and First Training Run
**Goal:** Self-contained Colab notebook for GPU training. | **Model:** Sonnet

1. `/model sonnet`
2. `/colab-export`
3. Verify the generated notebook has: GPU check, pip installs, repo clone, data download, wandb login, training run, checkpoint save
4. Add a GPU memory monitoring cell (`torch.cuda.max_memory_allocated()`)
5. Upload to Colab → Runtime → T4 GPU → Run all cells
6. Record observations in SCRATCHPAD.md (GPU type, time/epoch, peak memory, final loss, errors)
7. `/review-and-commit`

**Checkpoint:**
- `notebooks/colab_training.ipynb` exists and runs
- First training run observed (even if not converged)
- Observations in SCRATCHPAD.md

---

## 11. Session 9 — Debug, Tune, and Stabilize Training

**Goal:** A training run that converges. Stable loss curves. Reasonable retrieval metrics.
**Duration:** Variable — this might take multiple sub-sessions over 2–3 days.
**Model:** Opus (debugging ML training requires deep reasoning).

This session is inherently iterative. You'll go back and forth between Colab (running training) and Claude Code (diagnosing and fixing issues).

### Step 1: Diagnose the first run

Start Claude Code and share your observations:

> Here's what happened in my first training run on Colab: [paste your observations from the scratchpad — loss values, any NaN, convergence behavior, errors, wandb screenshots if relevant]. Help me diagnose what's going wrong and fix it.

> **Common issues and their typical fixes:**
>
> - **Loss is NaN after a few epochs:** Temperature collapse. Add a clamp to the temperature (min=0.01). Also check for NaN in the data.
> - **Loss doesn't decrease:** Learning rate too low, or the batch size is too small for contrastive learning (need enough negatives). Try batch_size=256 if memory allows.
> - **Loss decreases then plateaus very early:** Model capacity might be too low, or the projection head is too small. Try hidden_dim=512 for the projection.
> - **One pairwise loss converges but others don't:** The modalities have very different signal strength. Try adjusting lambda weights (e.g., increase lambda for the struggling pair).
> - **Out of memory on Colab:** Reduce batch size to 64, or reduce GIN hidden_dim from 300 to 256.

### Step 2: Apply fixes

For each fix Claude suggests:

> Before you make this change, explain why this fix addresses the root cause. I want to understand the mechanism, not just the patch.

### Step 3: Re-run on Colab

After applying fixes, re-export the notebook and run again:

> Update the Colab notebook with the changes we just made. Make sure the config values in the notebook match what we changed in configs/default.yaml.

Repeat Steps 1–3 until you see:
- Loss decreasing smoothly over 50+ epochs
- No NaN values
- Temperature stabilizing in the range 0.02–0.20
- All three pairwise losses decreasing (not just one or two)

### Step 4: Record the successful configuration

> Update SCRATCHPAD.md with the final configuration that works. Include: learning rates, batch size, any clamp values, training time, and the approximate loss values at convergence.

> Update configs/default.yaml with the tuned values.

### Checkpoint

- Training converges reliably
- Loss curves look healthy in wandb
- You understand what each hyperparameter does and why
- Updated config reflects the tuned values

### Step 5: Commit

> /review-and-commit

Expected message: `fix: tune hyperparameters for stable training convergence`

---

## 12. Session 10 — Ablation Study Infrastructure
**Goal:** Ablation configs, runner script, and clustering evaluation. | **Model:** Opus

1. `/model opus`
2. `/phase2-ablations`
3. Verify: 3 ablation configs have correct lambda weights (only one non-zero each)
4. Verify: `scripts/run_ablations.py` uses same random seeds across all 4 variants
5. Learn: `/explain-this` → AMI, ARI, and why "adjusted" metrics matter
6. Run `pytest tests/ -v --tb=short` — clustering tests pass
7. `/review-and-commit`

**Checkpoint:**
- `configs/ablation_*.yaml` (3 files)
- `scripts/run_ablations.py`
- `src/evaluation/clustering.py`

---

## 13. Session 11 — Run Ablations and Analyze Results

**Goal:** All 4 training variants completed. Results table generated. Core scientific claim verified (or invalidated).
**Duration:** 60 minutes in Claude Code + several hours of Colab training time.
**Model:** Opus for analysis.

### Step 1: Run ablations on Colab

This happens outside Claude Code. Create an ablation notebook (or extend the existing one):

> /model sonnet

> Create a notebook called notebooks/run_ablations.ipynb that runs all 4 training variants sequentially on Colab. Each variant should: load its config, train for the full epoch count, evaluate retrieval metrics on test set, save results. At the end, display the comparison table.

Upload and run on Colab. This will take several hours (4 training runs × 10–30 minutes each).

### Step 2: Analyze results

Once the ablation runs complete, bring the results back to Claude Code:

> /model opus

> Here are my ablation results: [paste the comparison table or JSON from the Colab notebook]. Analyze these results. Which variant won? Does tri-modal outperform all bi-modal pairs? Which bi-modal pair is strongest? What does this tell us about the complementarity of the modalities?

### Step 3: Have the architect explain the biological implications

> Use the architect-explainer to interpret these results biologically. If morph↔expr is the strongest bi-modal pair, what does that mean about what Cell Painting captures versus L1000? If adding the molecular modality improves results, what does that tell us about the relationship between chemical structure and cellular response?

### Step 4: Handle the case where tri-modal doesn't win

This is a real possibility. If tri-modal doesn't clearly outperform all bi-modal pairs:

> The tri-modal model didn't outperform [specific bi-modal variant] on [specific metric]. This is an honest scientific result, not a failure. Help me write an analysis of why this might be the case. Consider: dataset size (only ~2K compounds), modality signal strength, potential for the third modality to add noise.

> **Important mindset:** Negative results are valid results. The PRD's goal G2 says "at least 2 of 3 bi-modal pairs are outperformed." If you achieve this, the project succeeds. If you don't, you still have a rigorous analysis of why not.

### Step 5: Generate the results table for the README

> Generate a markdown table comparing all variants across all retrieval metrics and MOA clustering metrics. This will go in the README. Format it clearly with the winning values bolded.

### Step 6: Update the scratchpad and commit

> Update SCRATCHPAD.md with the ablation results and your interpretation. Save the results table to results/ablation_results.md.

> /review-and-commit

### Checkpoint — Phase 2 Complete

At this point you should have:
- A trained tri-modal model
- Three trained bi-modal baselines
- A comparison table showing which variant wins on which metric
- An interpretation of what the results mean biologically
- Understanding of AMI/ARI, retrieval metrics, and ablation methodology

---

## 14. Session 12 — MOA Clustering and Retrieval Evaluation
**Goal:** Full evaluation: MOA clustering, k-NN accuracy, 6-direction retrieval. | **Model:** Opus

1. `/model opus`
2. `/evaluate-model`
3. Run on Colab (or locally if CPU-sufficient for inference)
4. Verify: `results/evaluation_results.json` contains AMI, ARI, silhouette, k-NN accuracy, all retrieval metrics
5. Learn: `/explain-this` → hub points in retrieval, why retrieval can be asymmetric
6. `/review-and-commit`

**Checkpoint:**
- `scripts/evaluate.py`
- `results/evaluation_results.json`
- Interpretation of retrieval asymmetries

---

## 15. Session 13 — UMAP Visualizations and Results Notebook
**Goal:** Paper-quality visualizations in the main results notebook. | **Model:** Sonnet + Opus

1. `/model sonnet`
2. `/results-notebook`
3. `/model opus` — review UMAP plot: do MOA clusters form? Which are tightest?
4. Run notebook on Colab, save figures
5. Verify: `results/figures/` has `embedding_umap.png`, `training_curves.png`, `retrieval_heatmap.png`, `ablation_comparison.png` (PNG + SVG)
6. `/review-and-commit`

**Checkpoint:**
- `notebooks/02_results_analysis.ipynb`
- Figures in `results/figures/`

---

## 16. Session 14 — Interpretability Analysis
**Goal:** Feature importance, gene-morphology mapping, modality disagreement. | **Model:** Opus

1. `/model opus`
2. `/phase3-interpret`
3. Learn: `/explain-this` → what "morphology-visible genes" means, why modality disagreement is novel
4. Verify: `src/evaluation/interpretability.py` has all 4 analysis functions
5. `/review-and-commit`

**Checkpoint:**
- `src/evaluation/interpretability.py`
- Feature importance, gene-morphology, modality disagreement results

---

## 17. Session 15 — Biology Notebook and GSEA
**Goal:** Interpretability notebook with GSEA and biology figures. | **Model:** Opus

1. `/model opus`
2. `/biology-notebook`
3. Learn: `/explain-this` → how to read GSEA enrichment plots, what Hallmark gene sets represent
4. Run notebook on Colab
5. Bring GSEA results back: paste top enriched pathways and ask Claude for biological interpretation
6. `/review-and-commit`

**Checkpoint:**
- `notebooks/03_interpretability.ipynb`
- Biology-aware interpretation you can articulate

---

## 18. Session 16 — Test Coverage Push
**Goal:** Hit 80% test coverage across src/. | **Model:** Sonnet

1. `/model sonnet`
2. `/coverage-push`
3. Verify: `pytest tests/ --cov=src --cov-report=term-missing` shows ≥80% total
4. `/review-and-commit`

**Checkpoint:**
- Total coverage ≥ 80%
- All tests pass
- Edge cases covered (empty data, NaN, single-element batches)

---

## 19. Session 17 — README, Documentation, and Docker
**Goal:** Comprehensive README, data docs, technical summary, Docker. | **Model:** Opus + Sonnet

1. `/model opus`
2. `/polish-docs`
3. Verify: README has results table, UMAP figure, architecture diagram, quick start
4. Verify: `data/README.md` is standalone
5. Verify: `docs/technical_summary.md` is ~2 pages
6. `/model sonnet` — verify Dockerfile matches current dependencies
7. `/review-and-commit`

**Checkpoint:**
- `README.md`, `data/README.md`, `docs/technical_summary.md`
- Dockerfile valid

---

## 20. Session 18 — GitHub Actions CI and Final Polish
**Goal:** CI pipeline, linting clean, all quality checks passing. | **Model:** Sonnet

1. `/model sonnet`
2. `/setup-ci`
3. Verify: `.github/workflows/ci.yml` exists with black, ruff, pytest steps
4. Verify: all tests pass, coverage ≥ 80%
5. Verify: CLAUDE.md "Current Phase" updated to complete
6. `/review-and-commit`

**Checkpoint:**
- `.github/workflows/ci.yml`
- All linting clean
- CLAUDE.md up to date

---

## 21. Session 19 — Stretch Goals

**Goal:** Implement whichever stretch goals you have time for.
**Duration:** Variable.
**Model:** Opus.

This session is optional. Choose based on time and interest:

### Option A: Streamlit Demo App (Stretch Goal S3)

> Build a Streamlit app in app/streamlit_app.py that: loads the trained model, lets the user paste a SMILES string, embeds it using the molecular encoder, finds the 5 nearest neighbors in the embedding space across all modalities, and displays the results with compound names and similarity scores. Include a UMAP plot showing where the query compound falls in the embedding space.

### Option B: AttentiveFP Alternative Encoder (Stretch Goal S3)

> Implement an AttentiveFP molecular encoder as an alternative to GIN in src/models/encoders.py. Add a config option to switch between GIN and AttentiveFP. Run a training comparison and add the results to the ablation table.

### Option C: Cross-Perturbation Retrieval (Stretch Goal S2)

> Implement cross-perturbation-type retrieval: given a CRISPR knockout gene, retrieve the most similar chemical compound. This requires downloading LINCS-ORF data in addition to Rosetta. Add this as an advanced evaluation in src/evaluation/cross_perturbation.py.

### Step 2: Commit stretch goal work

> /review-and-commit

---

## 22. Session 20 — Final Review and Ship
**Goal:** Repository is portfolio-ready. Push to GitHub. | **Model:** Opus

1. `/model opus`
2. Run full code review: check for TODOs, hardcoded paths, missing docstrings, stale config values
3. Fix all FIX items, triage WARN items
4. Commit: `chore: final polish for release`
5. Push to GitHub:
   ```
   git remote add origin https://github.com/[your-username]/capy.git
   git push -u origin main
   ```
6. Verify on GitHub: README renders, CI triggers, no large files committed

**Checkpoint — Project Complete:**
- Working tri-modal contrastive learning system
- Rigorous ablation study (tri-modal vs bi-modal)
- Interpretability analysis with biological insights
- 80%+ test coverage with CI pipeline
- Paper-quality visualizations
- Comprehensive documentation
- Portfolio-ready

---

## Appendix A — All Configuration Files

Copy-paste each file exactly as shown. Line breaks and indentation matter for YAML and markdown.

### A1. CLAUDE.md (project root)

```
# CaPy Project — Claude Code Instructions

## Project Overview
CaPy (Contrastive Alignment of Phenotypic Yields) is a tri-modal contrastive
learning framework aligning molecular structure, cell morphology, and gene
expression for drug discovery. Portfolio project for an insitro internship.

## Tech Stack
- Python 3.10+, PyTorch 2.1+, PyTorch Geometric 2.4+
- RDKit for molecular graphs, scanpy for expression data, gseapy for GSEA
- wandb for experiment tracking, omegaconf for config management
- black (formatter), ruff (linter), pytest (testing)

## Repository Structure
Source code in src/, scripts in scripts/, tests in tests/, configs in configs/.
Data in data/ (gitignored). Notebooks in notebooks/.

## Critical Conventions
- ALL public functions: Google-style docstrings with type hints
- NEVER use raw print() — use the logger from src/utils/logging.py
- Every module must have a corresponding test file
- Embedding dimension is 256 everywhere — constant, not magic number
- Use cfg (omegaconf DictConfig) for all hyperparameters, never hardcode
- InfoNCE loss is SYMMETRIC (average both directions) — verify every time
- Molecular graphs: always validate node features are [num_atoms, feat_dim]
- All random seeds (torch, numpy, random, PyG) must be set via utils/config.py

## What NOT To Do
- Do NOT use raw images — we use pre-extracted CellProfiler features only
- Do NOT commit data/ or checkpoints/ (gitignored)
- Do NOT use Transformers/attention for tabular encoders (MLPs are correct)
- Do NOT hardcode paths — use configs or Path objects

## Teaching Mode
When implementing new components, ALWAYS:
1. Write the code
2. Write the test
3. Explain the key design decision in a WHY THIS WORKS comment block
   at the top of the file, written for someone learning ML/PyTorch

## Build & Test Commands
- Format: black src/ tests/ scripts/
- Lint: ruff check src/ tests/ scripts/
- Test: pytest tests/ -v --tb=short
- Coverage: pytest tests/ --cov=src --cov-report=term-missing
- Train: python scripts/train.py --config configs/default.yaml
- Sanity: python -c "from src.models.capy import CaPy; print('OK')"

## Current Phase
Phase 1 — Foundation (Data Pipeline + Model Architecture)
```

### A2. .claude/agents/architect-explainer.md

```
---
name: architect-explainer
description: Explains ML architecture decisions, PyTorch patterns, and math
  intuition for a CS/math student learning deep learning. Invoked after
  implementation to teach WHY something was built the way it was.
tools:
  - Read
  - Grep
  - Glob
---

# Architect-Explainer Agent

You are a patient ML educator explaining CaPy's design to a CS/math student
who is new to PyTorch and deep learning but strong in programming and math.

## Your Communication Style
- Start with the INTUITION (what is this trying to do, in plain English?)
- Then the MATH (one or two key equations, connect them to the code)
- Then the PYTORCH PATTERN (why this API call, what are the alternatives?)
- End with WHAT COULD GO WRONG (common bugs for beginners)

## When Explaining Code
- Point to specific line numbers in the actual project files
- Compare to alternatives (we used X instead of Y because...)
- Use analogies from CS concepts the student already knows
- Name well-known patterns (CLIP-style training, SimCLR projection heads)

## Topics You Cover
- Why GIN over other GNNs (GAT, GCN, SchNet)
- Why InfoNCE and not triplet loss or NT-Xent variants
- Why learnable temperature, and what happens if it is fixed
- Why L2 normalization before computing similarities
- Why scaffold splitting prevents data leakage
- Why projection heads help (the SimCLR finding)
- Batch size tradeoffs in contrastive learning
- What gradient clipping does and when you need it

## Response Format
### What This Does (1-2 sentences)
### The Math Behind It
### The PyTorch Implementation
### What Could Go Wrong
```

### A3. .claude/agents/code-reviewer.md

```
---
name: code-reviewer
description: Reviews CaPy code for correctness, style, testing gaps, and
  common ML bugs. Run before committing.
tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# Code Reviewer Agent

You review code in the CaPy project. Your priorities:

## Review Checklist (in priority order)
1. Correctness: Does the math match the spec? Are tensor shapes right?
2. ML Bugs: Forgotten .detach(), wrong loss direction, missing .eval()
3. Type hints and docstrings: Every public function, Google style
4. Test coverage: Does a test exist? Does it test edge cases?
5. Style: black + ruff compliance, no magic numbers

## Common ML Bugs to Catch
- Forgetting model.eval() during validation
- Not detaching tensors that should not propagate gradients
- InfoNCE labels off by one or wrong device
- Temperature going to zero (should be clamped)
- Forgetting L2 normalization before cosine similarity
- Data leakage in train/val/test split
- Not setting all random seeds before data splitting

## Output Format
For each file, output:
- PASS: thing that is correct and well-done
- WARN: potential issue, not blocking
- FIX: must fix before merging
```

### A4. .claude/agents/data-detective.md

```
---
name: data-detective
description: Investigates data quality issues in the Rosetta dataset.
  Checks distributions, missing values, alignment between modalities.
tools:
  - Read
  - Bash
  - Grep
  - Glob
---

# Data Detective Agent

You specialize in data quality for the CaPy project's Rosetta dataset.

## Your Investigations
- Check for NaN/inf values in morphology and expression features
- Verify compound IDs match across all three modalities
- Check feature distributions (are any zero-variance? heavily skewed?)
- Validate scaffold split does not leak similar molecules
- Confirm SMILES parse correctly to molecular graphs via RDKit
- Verify normalization (RobustScaler for morphology, z-scores for expression)

## Reporting Style
- Give exact counts: "42 of 2084 compounds have missing expression data"
- Save diagnostic plots to data/diagnostics/
- Flag deviations from the PRD specification
- Suggest concrete fixes for every issue found
```

### A5. .claude/commands/phase1-scaffold.md

```
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
```

### A6. .claude/commands/phase1-data.md

```
Implement the complete data pipeline for CaPy. Create these files:

1. src/data/download.py
   Download Rosetta CDRP-bio profiles. Try S3 first (aws s3 cp with --no-sign-request), fall back to direct HTTP links from Cell Painting Gallery GitHub. Include a --dry-run flag. Use tqdm for progress. Save to data/raw/.

2. src/data/preprocess.py
   Match compounds across morphology and expression by compound ID. Remove compounds missing any modality. Remove zero-variance and >50% NaN morphology features. RobustScaler for morphology (clip to [-5,5]). Verify z-scores for expression. Bemis-Murcko scaffold split 70/15/15, stratified by MOA where available. Save to data/processed/.

3. src/data/featurize.py
   SMILES to PyG molecular graphs. 9-dim node features (atomic number one-hot, degree, formal charge, num H, is aromatic, is in ring). 4-dim edge features (bond type, conjugated, in ring, stereo). Both directions for edges. Use RDKit.

4. src/data/dataset.py
   CaPyDataset(torch.utils.data.Dataset) returning (mol_graph, morph_vector, expr_vector) triples. Custom collate_fn that uses PyG Batch for molecular graphs and torch.stack for vectors.

5. tests/test_data.py
   Test with synthetic data (no real download needed): correct shapes, no NaNs, split proportions, graph validity (edge_index is Long, both directions present), collate function produces correct batch.

After implementation, use the architect-explainer agent to explain: scaffold splitting and data leakage prevention, PyG batching of variable-size graphs, why RobustScaler over StandardScaler for morphology features.
```

### A7. .claude/commands/phase1-model.md

```
Implement all model components for CaPy. Create these files:

1. src/models/encoders.py
   MolecularEncoder: 5-layer GIN (hidden_dim=300), global mean pool, projection MLP (300->512->256), L2 normalize. Use torch_geometric.nn.GINConv with batch norms.
   TabularEncoder: configurable input_dim, 3 MLP layers (input->512, 512->512, 512->512) with LayerNorm, ReLU, Dropout(0.1). Residual connections on layers 2-3. Projection head (512->256), L2 normalize.
   MorphologyEncoder = TabularEncoder(input_dim=config)
   ExpressionEncoder = TabularEncoder(input_dim=config)

2. src/models/losses.py
   info_nce(z_a, z_b, temperature): symmetric InfoNCE loss. Compute similarity matrix z_a @ z_b.T / temperature. Subtract max for numerical stability. CrossEntropy both directions. Average.

3. src/models/capy.py
   CaPy(nn.Module) with mol_encoder, morph_encoder, expr_encoder, learnable log_temperature (init log(1/0.07)). Temperature clamped to [0.01, 10.0]. compute_loss returns weighted sum of 3 pairwise InfoNCE losses + per-pair loss dict.

4. tests/test_models.py
   Shape checks: all encoders output [batch_size, 256]. Normalization check: output norms approximately 1.0. Gradient flow: all parameters have non-None gradients after backward. Device placement: works on CPU.

5. tests/test_losses.py
   Perfect matching loss approximately log(batch_size). Random embeddings produce higher loss. Symmetry: info_nce(a,b,t) == info_nce(b,a,t). Batch size 1 does not crash.

After implementation, use the architect-explainer agent to explain: why GIN is provably powerful (WL test connection), why L2 normalize BEFORE similarity, why learnable temperature outperforms fixed, what projection heads do (SimCLR finding), why residual connections in the MLP encoders.
```

### A8. .claude/commands/phase2-training.md

```
Implement training infrastructure. Create these files:

1. src/training/trainer.py
   Full training loop. AdamW optimizer with separate param groups: GIN params at lr=1e-4, all other params at lr=1e-3. Weight decay 1e-4. Cosine annealing with 10-epoch linear warmup. Gradient clipping max_norm=1.0. wandb logging of: total loss, per-pair losses, temperature, learning rate, epoch time. Early stopping on validation mean Recall@10 with patience=20. Checkpoint best model to checkpoints/best_model.pt.

2. src/training/scheduler.py
   CosineAnnealingWithWarmup class wrapping torch.optim.lr_scheduler. Linear warmup from 0 to target LR over warmup_epochs, then cosine decay.

3. src/evaluation/retrieval.py
   Evaluate all 6 retrieval directions (mol->morph, morph->mol, mol->expr, expr->mol, morph->expr, expr->morph). For each: compute cosine similarity matrix, rank, compute Recall@1, Recall@5, Recall@10, MRR. Return dict of all metrics plus mean across directions.

4. scripts/train.py
   Main entrypoint. Argparse with --config flag. Load YAML via omegaconf. Seed all random sources. Init wandb. Build dataset, dataloaders, model, optimizer, scheduler. Call trainer. Save final results.

5. configs/default.yaml
   Update with any new parameters. All values must have inline comments.

After implementation, use the architect-explainer agent to explain: why different learning rates for GNN vs MLP, why cosine annealing with warmup, why batch size matters MORE in contrastive learning than supervised, what early stopping on retrieval means practically.
```

### A9. .claude/commands/phase2-ablations.md

```
Implement ablation study infrastructure. Create:

1. configs/ablation_mol_morph.yaml (lambda1=1.0, lambda2=0.0, lambda3=0.0)
2. configs/ablation_mol_expr.yaml (lambda1=0.0, lambda2=1.0, lambda3=0.0)
3. configs/ablation_morph_expr.yaml (lambda1=0.0, lambda2=0.0, lambda3=1.0)

All ablation configs inherit from default.yaml and only override lambda weights. Use the same random seed as default.

4. scripts/run_ablations.py
   Run tri-modal + 3 bi-modal variants sequentially. Same seeds across all variants. Save each variant's metrics to results/. Generate comparison table (markdown) at the end.

5. src/evaluation/clustering.py
   MOA evaluation: k-means clustering (k = number of unique MOAs), agglomerative clustering. Metrics: AMI, ARI, silhouette score. k-NN MOA classification accuracy for k=5, 10, 20. Handle compounds without MOA labels (exclude from evaluation, report count).

After implementation, use the architect-explainer agent to explain: what the ablation results will tell us biologically, how to interpret AMI and ARI for a clustering beginner, what it means if one bi-modal pair dominates.
```

### A10. .claude/commands/phase3-interpret.md

```
Implement interpretability analysis. Create:

1. src/evaluation/interpretability.py
   Functions for:
   - feature_importance: for each morphology feature, compute Pearson correlation between that feature's values and the morph->expr embedding alignment score. Return top 20 features.
   - gene_morphology_mapping: for each gene, correlate expression values with morph->expr retrieval rank. Top 100 genes are "morphology-visible". 
   - scaffold_analysis: group compounds by Bemis-Murcko scaffold, compute mean embedding per scaffold, find scaffolds closest to each MOA cluster center.
   - modality_disagreement: for each compound pair, compute rank in morph space and rank in expr space. Find pairs with largest rank discrepancy. Return top 10 in each direction (morph-similar/expr-different and expr-similar/morph-different).

2. notebooks/03_interpretability.ipynb
   Section 1: Top 20 morphology features bar chart, color-coded by compartment
   Section 2: Gene-morphology mapping results
   Section 3: GSEA on top 100 genes using gseapy with MSigDB Hallmark gene sets
   Section 4: Top 10 disagreement compound pairs table with known targets
   Section 5: Scaffold-to-MOA mapping visualization
   All figures paper-quality (300 DPI, clean fonts, colorblind-friendly).

After implementation, use the architect-explainer agent to explain: how to read GSEA results, what morphology-visible genes are biologically, why modality disagreement is the most novel analysis in the project.
```

### A11. .claude/commands/review-and-commit.md

```
Run a complete quality check before committing:

1. Run black src/ tests/ scripts/ and fix any formatting issues
2. Run ruff check src/ tests/ scripts/ --fix and fix any linting issues
3. Run pytest tests/ -v --tb=short and confirm all tests pass
4. Use the code-reviewer agent to review all files changed since last commit
5. Fix any FIX items from the review
6. Generate a conventional commit message (feat/fix/test/chore prefix)
7. Stage all changes and create the commit

Show me the proposed commit message before committing. Do not push.
```

### A12. .claude/commands/explain-this.md

```
Use the architect-explainer agent to explain the file or concept I am about to describe.

Structure your explanation as:
1. What it does (plain English, 1-2 sentences)
2. The key math or algorithm (with equations if relevant)
3. Why this approach was chosen over alternatives
4. Common pitfalls for beginners working with PyTorch

Wait for me to specify what I want explained. Do not start explaining until I give the topic.
```

### A13. .claude/commands/colab-export.md

```
Create or update notebooks/colab_training.ipynb to be fully self-contained for Google Colab with a T4 or V100 GPU.

The notebook must have these cells in order:
1. GPU verification: check torch.cuda.is_available(), print GPU name and memory
2. Install dependencies: pip install torch torch-geometric rdkit-pypi scanpy gseapy wandb omegaconf tqdm scikit-learn scipy matplotlib seaborn umap-learn
3. Clone the repo or upload src/ directory
4. Download Rosetta data (using src/data/download.py)
5. Preprocess data (using src/data/preprocess.py)
6. wandb login cell
7. Load config from configs/default.yaml (or whichever config is specified)
8. Training run with progress display
9. Evaluation on test set with results display
10. Save best checkpoint and download to local machine
11. Print peak GPU memory usage

The notebook should work top-to-bottom with no manual intervention after the wandb login.
```

### A14. .claude/skills/pytorch-geometric-patterns.md

```
---
name: pytorch-geometric-patterns
description: PyTorch Geometric patterns for molecular graphs. Load when working on molecular encoder or data featurization.
---

# PyG Patterns for CaPy

## SMILES to Graph
Use RDKit to parse SMILES, extract atom and bond features, create torch_geometric.data.Data objects. Node features shape: [num_atoms, feat_dim]. Edge index shape: [2, num_edges] in COO format. Must include BOTH directions (i->j AND j->i).

## Batching Variable-Size Graphs
PyG DataLoader handles this via Batch.from_data_list(). Creates single disconnected graph with batch vector mapping nodes to original graphs. global_mean_pool(x, batch) pools per-graph.

## GIN Layer
GINConv wraps an MLP applied to aggregated neighborhoods. Pattern:
mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
conv = GINConv(mlp)

## Common Pitfalls
- edge_index must be Long tensor, not Float
- Edge features need both directions
- batch vector is auto-created by DataLoader
- Forgetting to L2-normalize output before contrastive loss
```

### A15. .claude/skills/contrastive-learning-reference.md

```
---
name: contrastive-learning-reference
description: InfoNCE, temperature, CLIP-style training reference. Load when working on losses or training.
---

# Contrastive Learning Reference

## InfoNCE (NT-Xent)
S[i,j] = cos(z_a[i], z_b[j]) / temperature
Loss = CrossEntropy(S, [0,1,...,N-1]) averaged both directions.
Perfect matching: loss = log(N). This is the MINIMUM, not zero.

## Temperature
Below 0.05: unstable gradients. Above 0.5: too smooth, weak signal.
CLIP uses learnable temperature, initialized at 0.07.
Always clamp: temp.clamp(min=0.01, max=10.0).

## Numerical Stability
Subtract max logits before softmax: logits -= logits.max(dim=-1, keepdim=True).values

## Multi-Modal CLIP Pattern
K modalities produce K*(K-1)/2 pairwise losses.
CaPy: 3 modalities = 3 pairwise losses, weighted by lambda.

## Batch Size
More negatives = better contrastive signal. Minimum useful: 64. Sweet spot for ~2K dataset: 128-256.
```

### A16. SCRATCHPAD.md

```
# CaPy Development Scratchpad
> This file is gitignored. Claude uses it for cross-session notes.

## Data Pipeline Notes
(to be filled during Phase 1)

## Training Notes
(to be filled during Phase 2)

## Ablation Results
(to be filled during Phase 2)

## Debugging Log
(to be filled as issues arise)

## Decisions That Differ From PRD
(track any deviations here)
```

### A17. .gitignore

```
# Data
data/raw/
data/processed/
data/diagnostics/

# Checkpoints and results
checkpoints/
wandb/

# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp

# Misc
SCRATCHPAD.md
.DS_Store
*.log
```

---

## Appendix B — Troubleshooting Common Issues

### Claude Code runs out of context mid-session
Start a fresh session. Claude will re-read CLAUDE.md and regain project context. Use SCRATCHPAD.md to pass notes between sessions. For long tasks, use subagents (they have their own context window).

### Plugin installation fails
Run `/plugin` and check the Errors tab. Most common cause: the language server binary is not installed. For pyright: `npm install -g pyright`. If a marketplace plugin has been renamed, search the Discover tab by functionality.

### Training produces NaN loss
Check in this order: (1) Is temperature clamped? (2) Are there NaN/inf values in the input data? (3) Is gradient clipping enabled? (4) Is the learning rate too high for the GIN?

### PyG import errors on Colab
PyTorch Geometric requires matching versions with PyTorch. Use: `pip install torch-geometric -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html` where you replace the versions with your Colab's torch and CUDA versions.

### Scaffold splitting produces very imbalanced splits
Some scaffolds are very common (benzene ring appears in many drugs). If one split is much larger than expected, try increasing the minimum scaffold frequency threshold or fall back to random splitting with a note about the limitation.

### wandb won't log from Colab
Run `wandb.login()` in its own cell and enter your API key when prompted. If running headless, set the API key as an environment variable: `os.environ["WANDB_API_KEY"] = "your-key"`.

### Model evaluation is slow
Retrieval evaluation computes an NxN similarity matrix. For N=2000 this is fast. If somehow N is much larger, compute in chunks. Use `torch.no_grad()` during evaluation.

---

## Appendix C — Prompt Patterns and Techniques

### The Interrupt Pattern
At any point during Claude's implementation, you can interrupt and ask for an explanation. This is the fastest way to learn:

> Wait — before you continue, explain to me what you just did with [specific thing]. Why did you choose [approach A] instead of [approach B]?

### The Teach-Back Pattern
After the architect-explainer explains something, verify your understanding:

> Let me make sure I understand. [Your explanation in your own words]. Is that right? What am I missing?

### The What-If Pattern
Explore alternatives to deepen understanding:

> What would happen if we used [alternative approach]? Why would that be worse (or better)?

### The Debug-By-Narration Pattern
When something breaks, describe what you see rather than guessing at causes:

> The training ran for 12 epochs, then the loss jumped from 2.3 to NaN. The temperature at that point was 0.008. The per-pair losses were: mol-morph=2.1, mol-expr=NaN, morph-expr=2.4. What happened?

### The Commit-Message-As-Summary Pattern
The commit message forces Claude to summarize what was accomplished. If the message is vague, the session's output might be too:

> Before committing, give me a conventional commit message that specifically names the modules created and the key design decisions made. Not just "implement models" — something like "feat: implement GIN molecular encoder with 5-layer architecture and symmetric InfoNCE loss with learnable temperature".

### The Scratchpad-As-Memory Pattern
At the end of every session, update the scratchpad:

> Update SCRATCHPAD.md with: what we built today, any bugs we hit, any decisions that deviated from the PRD, and any open questions for next session.

This ensures continuity across sessions even when context is lost.
