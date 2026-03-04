# Agent Guide

- Windows environment.
- Activate the virtual environment with `venv\Scripts\activate`.

This repository simulates a 4-armed contextual bandit game ("Mining in Space") with linear contextual agents and an LLM-agent pipeline.

## Game setup

- Context has three binary signals: Mercury, Krypton, Nobelium (`-1` or `+1`).
- Arms are buttons `0..3` that are permuted to canonical planets `A..D` once per game (`arm_permutation`).
- Rewards are sampled with Gaussian noise and rounded to integers by default (`integer_rewards=True`).
- Canonical expected rewards in `scripts/game.py`:
  - `A: 50 + 15*Mercury - 15*Krypton`
  - `B: 50 + 15*Krypton - 15*Nobelium`
  - `C: 50 + 15*Nobelium - 15*Mercury`
  - `D: 50`

All agents follow this interface:

- `select_arm(context) -> (int, dict)`
- `update(context, arm, reward)`
- `get_recommendations(context, k=1.96)`

## Project structure

- `scripts/agents.py`: Linear/RLS agent implementations.
- `scripts/game.py`: Shared game environment.
- `scripts/agent/`: Simulation CLI, cleanup pipeline, and manual console game.
- `scripts/llm/`: LLM agent implementation and LLM simulation runner.
- `scripts/analysis/agents/`: Analysis scripts for linear-agent simulations.
- `scripts/analysis/llm/`: Analysis scripts for LLM simulations.
- `data/agents/`, `data/llm/`: Output CSV files.
- `graphics/agents/`, `graphics/llm/`: Generated analysis artifacts.

## Linear contextual agents (RLS)

`LinearRegressionAgent` (`scripts/agents.py`) adds an intercept to context (`[1, x1, x2, x3]`) and tracks per-arm:

- `A_inv` (inverse precision matrix)
- `b` (reward-weighted feature accumulator)

Updates use Sherman-Morrison. Prediction returns `(mean, sigma)` where:

- `mean = beta^T x`, with `beta = A_inv @ b`
- `sigma = sqrt(x^T A_inv x)`

Key parameters:

- `n_features=3` (before intercept)
- `regularization=100` by default (higher value -> larger initial uncertainty)

Concrete agent:

- `LinearUCBAgent`: chooses `argmax(mu + k*sigma)` with `exploration_multiplier` (`k`, default `1.96`).

### Decision transparency

`LinearUCBAgent.select_arm` returns rich metadata in `info`:

- `intent`: `explore` or `exploit`
- `intent_reason`: short rationale string
- `explanation`: human-readable sentence tied to current context
- `exploration`: `1` for explore, `0` for exploit
- `context_features`: ordered feature names
- `chosen_mean`, `chosen_sigma`, `chosen_ucb`

Utility methods:

- `get_feature_weights(feature_names)`
- `get_feature_uncertainties(context, feature_names)`

## Running simulations

### Linear-agent pipeline

- Batch simulation:
  - `python scripts/agent/simulation.py --n_simulations 20 --n_trials 50`
- Quick run (subset grid):
  - `python scripts/agent/simulation.py --quick`
- Key API entry points in `scripts/agent/simulation.py`:
  - `run_single_game(...)`
  - `run_grid_simulation(...)`

Outputs default to `data/agents/simulation_results.csv`.

### Cleanup + full linear pipeline

- `python scripts/agent/cleanup_and_run_all.py --simulations 20 --trials 50`
- This cleans `data/agents/` and `graphics/agents/`, runs simulation, then runs all linear analysis scripts.
- Script prefers `venv\Scripts\python.exe` when available.

### LLM pipeline

- Requires `GEMINI_API_KEY`.
- Run:
  - `python scripts/llm/run_simulation.py --simulations 20 --trials 50 --output data/llm/simulation_results.csv`
- Main classes:
  - `LLMAgent`
  - `LLMDecisionError`

## Analysis scripts

### Linear-agent analysis (`scripts/analysis/agents/`)

- `summarize_models.py` -> `graphics/agents/model_summary.csv`
- `analyze_performance.py` -> `graphics/agents/performance_analysis.pdf`
- `analyze_exploration.py` -> `graphics/agents/exploration_analysis.pdf`
- `analyze_uncertainty.py` -> `graphics/agents/uncertainty_analysis.pdf`
- `analyze_weights.py` -> `graphics/agents/weight_evolution_analysis.pdf`
- `analyze_planet_distributions.py` -> `graphics/agents/planet_distribution_analysis.pdf`
- `visualize_simulation_arms.py` -> `graphics/agents/simulations/*.pdf`

### LLM analysis (`scripts/analysis/llm/`)

- `summarize_models.py` -> `graphics/llm/model_summary.csv`
- `analyze_performance.py` -> `graphics/llm/performance_analysis.pdf`
- `analyze_exploration.py` -> `graphics/llm/exploration_analysis.pdf`

## Other utilities

- Manual play: `python scripts/agent/game_in_console.py`
- Prototyping notebook: `scripts/testing.ipynb`
- Simulation schema reference: `SIMULATION_RESULTS_SCHEMA.md`

## Extending agents

- Derive from `LinearRegressionAgent` when possible.
- Keep the interface contract (`select_arm`, `update`, `get_recommendations`).
- Surface hyperparameters in `__init__` for experiment control.
- If you change logged outputs, keep simulation scripts and `SIMULATION_RESULTS_SCHEMA.md` aligned.
