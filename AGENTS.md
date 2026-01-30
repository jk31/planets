# Agent Guide

- Windows Environment
- Use python venv\scripts\activate

This project simulates a 4-armed contextual bandit game ("Mining in Space") with several learning agents. Context features are three binary signals: Mercury, Krypton, Nobelium. Arms correspond to planets A-D (permuted each game). All agents share a minimal interface:

- `select_arm(context) -> int` picks an arm index 0-3.
- `update(context, arm, reward)` incorporates the observed reward.
- `get_recommendations(context, k=1.96)` returns per-arm means/uncertainties for logging or UI.

## Context-blind agents
- **RandomAgent** (`agents.py`): Chooses uniformly at random; keeps no state. Recommendations are fixed placeholders (mean=50, no interval).
- **MeanTrackingAgent** (`agents.py`): Tracks per-arm sample means with an incremental update; action probabilities use softmax over estimates with temperature `gamma` (default 0.1). Recommendations currently return a placeholder mean (50) rather than the tracked estimate.

## Linear contextual agents (Recursive Least Squares)
Defined in `LinearRegressionAgent` (`agents.py`), extended by UCB/Thompson variants. Adds an intercept to the 3-feature context (shape 4). Maintains `A_inv` (precision) and `b` per arm; updates via Sherman-Morrison. Prediction returns `(mean, sigma)` where `sigma` comes from `x^T A_inv x`.

Key parameters:
- `n_features=3` (context dims), `regularization=100`.
- `pseudo_observations` (default `False`): if `True`, each arm pretrains on 10 random binary contexts with rewards ~N(50,10).

Concrete agents:
- **LinearUCBAgent**: Picks `argmax(mu + 1.96*sigma)` (UCB with 95% multiplier).
- **LinearThompsonAgent**: Samples `y* ~ N(mu, sigma)` per arm and picks the argmax (Thompson sampling).
- Utilities: `get_feature_weights(feature_names)` returns readable betas per arm.

## Gaussian Process contextual agents
Base **GaussianProcessAgent** (`agents.py`) models each arm with an RBF kernel:
- Hyperparameters: `lengthscale=2.0`, `noise_std=5.0`, `signal_std=20.0`.
- Data storage: `X[arm]`, `y[arm]` lists. The pseudo-observation loop currently short-circuits, so GPs start empty; when no data are present, predictions fall back to mean 50 with variance `signal_std^2`.
- Prediction centers rewards around 50 before the GP solve, then adds 50 back.

Concrete agents:
- **GPUCBAgent**: Chooses `argmax(mu + 1.96*sigma)`.
- **GPThompsonAgent**: Samples `y* ~ N(mu, sigma)` per arm and picks the argmax.

## How agents are used in the repo
- Simulation entrypoints: `run_single_game` / `run_batch_simulation` in `simulation.py` wire any `agent_class` with `MiningInSpaceGame` (`game.py`). Logs include choices, rewards, arm permutation, and (for linear agents) current weights plus agent uncertainties.
- Experiment script: `learning_curves.py` defines an `agents_to_test` dict and generates `learning_curves.pdf` and `individual_agents_ci.pdf`.
- Manual play: `game_in_console.py` lets a human choose arms in the console.

## Adding or tweaking agents
- Derive from `LinearRegressionAgent` or `GaussianProcessAgent` to reuse prediction/uncertainty helpers.
- Expose tunable hyperparameters via `__init__` and add instances to the `agents_to_test` mapping in `learning_curves.py`.
- Remember to update `get_recommendations` if you want UI/logs to reflect new belief calculations.
