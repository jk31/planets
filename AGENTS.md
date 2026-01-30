# Agent Guide

- Windows Environment
- Use python venv\scripts\activate

This project simulates a 4-armed contextual bandit game ("Mining in Space") with linear contextual learning agents. Context features are three binary signals: Mercury, Krypton, Nobelium. Arms correspond to planets A-D (permuted each game). All agents share a minimal interface:

- `select_arm(context) -> int` picks an arm index 0-3.
- `update(context, arm, reward)` incorporates the observed reward.
- `get_recommendations(context, k=1.96)` returns per-arm means/uncertainties for logging or UI.

## Linear contextual agents (Recursive Least Squares)
Defined in `LinearRegressionAgent` (`agents.py`), extended by UCB/Thompson variants. Adds an intercept to the 3-feature context (shape 4). Maintains `A_inv` (precision) and `b` per arm; updates via Sherman-Morrison. Prediction returns `(mean, sigma)` where `sigma` comes from `x^T A_inv x`.

Key parameters:
- `n_features=3` (context dims).
- `regularization` (default 100): Controls initial uncertainty. Higher values mean higher initial variance.

Concrete agents:
- **LinearUCBAgent**: Picks `argmax(mu + 1.96*sigma)` (UCB with 95% multiplier).
- **LinearThompsonAgent**: Samples `y* ~ N(mu, sigma)` per arm and picks the argmax (Thompson sampling).
- Utilities: `get_feature_weights(feature_names)` returns readable betas per arm.

## How agents are used in the repo
- **Simulation CLI**: Run `python simulation.py --n_simulations 20 --n_trials 150` to execute a batch of simulations for UCB and Thompson agents with varying regularization. Results are saved to `simulation_results.csv`.
- **Simulation API**: `run_single_game` and `run_batch_simulation` in `simulation.py` wire any `agent_class` with `MiningInSpaceGame` (`game.py`). Logs include choices, rewards, arm permutation, and current weights plus agent uncertainties (see [SIMULATION_RESULTS_SCHEMA.md](SIMULATION_RESULTS_SCHEMA.md) for full column details).
- **Experiment script**: `learning_curves.py` defines an `agents_to_test` dict, loads/saves to `simulation_results.csv`, and generates `regularization_sweep.pdf`.
- **Development**: `testing.ipynb` is used for prototyping and iterative testing of agent behaviors.
- **Manual play**: `game_in_console.py` lets a human choose arms in the console.

## Adding or tweaking agents
- Derive from `LinearRegressionAgent` to reuse prediction/uncertainty helpers.
- Expose tunable hyperparameters via `__init__` and add instances to the `agents_to_test` mapping in `learning_curves.py`.
- Remember to update `get_recommendations` if you want UI/logs to reflect new belief calculations.
