# Agent Guide

- Windows Environment
- Use python venv\scripts\activate

This project simulates a 4-armed contextual bandit game ("Mining in Space") with linear contextual learning agents. Context features are three binary signals: Mercury, Krypton, Nobelium. Arms correspond to planets A-D (permuted each game). All agents share a minimal interface:

- `select_arm(context) -> int` picks an arm index 0-3.
- `update(context, arm, reward)` incorporates the observed reward.
- `get_recommendations(context, k=1.96)` returns per-arm means/uncertainties for logging or UI.

## Project Structure
- `scripts/`: Python scripts for simulation, analysis, and visualization.
- `data/`: CSV files containing simulation results.
- `graphics/`: PDF files of generated plots and visualizations.

## Linear contextual agents (Recursive Least Squares)
Defined in `LinearRegressionAgent` (`scripts/agents.py`), extended by UCB variant. Adds an intercept to the 3-feature context (shape 4). Maintains `A_inv` (precision) and `b` per arm; updates via Sherman-Morrison. Prediction returns `(mean, sigma)` where `sigma` comes from `x^T A_inv x`.

Key parameters:
- `n_features=3` (context dims).
- `regularization` (default 100): Controls initial uncertainty. Higher values mean higher initial variance.

Concrete agents:
- **LinearUCBAgent**: Picks `argmax(mu + k*sigma)`. The `exploration_multiplier` (k) defaults to 1.96 but is configurable via `__init__`.
- Utilities: `get_feature_weights(feature_names)` returns readable betas per arm.

## How agents are used in the repo
- **Simulation CLI**: Run `python scripts/simulation.py --n_simulations 20 --n_trials 150` to execute a batch of simulations for UCB agents with varying regularization. Results are saved to `data/simulation_results.csv`.
- **Simulation API**: `run_single_game` and `run_batch_simulation` in `scripts/simulation.py` wire any `agent_class` with `MiningInSpaceGame` (`scripts/game.py`). Logs include choices, rewards, arm permutation, exploration status, and current weights plus agent uncertainties (see [SIMULATION_RESULTS_SCHEMA.md](SIMULATION_RESULTS_SCHEMA.md) for full column details).
- **Experiment script**: `scripts/learning_curves.py` defines an `agents_to_test` dict, loads/saves to `data/simulation_results.csv`, and generates `graphics/regularization_sweep.pdf`.
- **Exploration Analysis**: 
    - `scripts/exploration_analysis.py`: Visualizes exploration rates over time for UCB across different regularization values. Generates `graphics/exploration_rates.pdf`.
    - `scripts/ucb_exploration_rate.py`: Specifically analyzes the impact of the exploration multiplier (k) on the UCB exploration rate. Generates `graphics/ucb_exploration_rate.pdf`.
- **UCB Parameter Sweeps**: 
    - `scripts/ucb_sweep.py`: Tests the impact of varying the `exploration_multiplier` (k) for a fixed regularization. Generates `graphics/ucb_multiplier_sweep.pdf`.
    - `scripts/ucb_grid_search.py`: Performs a full grid search across `regularization` and `exploration_multiplier` (k) values. Generates a heatmap and faceted learning curves in `graphics/ucb_grid_search.pdf`.
- **Weight Visualization**: `scripts/visualize_weights.py` processes simulation results to show how agent weights converge to ground truth. Generates `graphics/weight_evolution_ucb.pdf`.
- **Development**: `scripts/testing.ipynb` is used for prototyping and iterative testing of agent behaviors.
- **Manual play**: `scripts/game_in_console.py` lets a human choose arms in the console.

## Adding or tweaking agents
- Derive from `LinearRegressionAgent` to reuse prediction/uncertainty helpers.
- Expose tunable hyperparameters via `__init__` and add instances to the `agents_to_test` mapping in `scripts/learning_curves.py`.
- Remember to update `get_recommendations` if you want UI/logs to reflect new belief calculations.
