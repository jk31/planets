# Agent Guide

- Windows Environment
- Use python venv\scripts\activate

This project simulates a 4-armed contextual bandit game ("Mining in Space") with linear contextual learning agents. Context features are three binary signals: Mercury, Krypton, Nobelium. Arms correspond to planets A-D (permuted each game). Rewards are rounded to the nearest integer to simplify gameplay and analysis. All agents share a minimal interface:

- `select_arm(context) -> (int, dict)` picks an arm index 0-3 and returns an info dictionary containing decision metadata (intent, explanation, etc.).
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

### Concrete agents
- **LinearUCBAgent**: Picks `argmax(mu + k*sigma)`. The `exploration_multiplier` (k) defaults to 1.96 but is configurable via `__init__`.

### Self-Explaining Decisions
The `LinearUCBAgent` provides transparency into its decision-making process by returning an info dictionary with:
- `intent`: Classified as `explore` when uncertainty influences the choice (the UCB-best arm is not the unique mean-best arm, or mean estimates are tied), otherwise `exploit`.
- `explanation`: A human-readable narrative.
    - **Exploration**: Lists all three context signals to show what the agent is trying to learn in this situation.
    - **Exploitation**: Lists all three context signals with per-feature value contributions (`beta * x`) for the chosen arm.
- `context_features`: The three context signals.

- Utilities: 
    - `get_feature_weights(feature_names)`: Returns readable betas per arm.
    - `get_feature_uncertainties(context, feature_names)`: Returns per-arm, per-feature uncertainty contributions `(x^2 * A_inv[i,i])` for the given context.

## How agents are used in the repo
- **Simulation CLI**: Run `python scripts/simulation.py --n_simulations 20 --n_trials 50` to execute a batch of simulations for UCB agents with varying regularization. Results are saved to `data/simulation_results.csv`.
- **Simulation API**: `run_single_game` and `run_grid_simulation` in `scripts/simulation.py` wire any `agent_class` with `MiningInSpaceGame` (`scripts/game.py`). The game defaults to 50 trials and uses integer rewards. Logs include choices, rewards, arm permutation, exploration status, current weights, per-arm total uncertainties, and per-feature uncertainty contributions (see [SIMULATION_RESULTS_SCHEMA.md](SIMULATION_RESULTS_SCHEMA.md) for full column details).
- **Cleanup and Full Run**: `python scripts/cleanup_and_run_all.py` is the main entry point to clean `/data` and `/graphics`, execute a fresh batch of simulations (default 50 trials), and run all analytical scripts.
- **Consolidated Analysis**:
    - `scripts/analyze_performance.py`: Performs overall reward analysis via heatmaps and learning curves. Generates `graphics/performance_analysis.pdf`.
    - `scripts/analyze_exploration.py`: Visualizes exploration rates over time across different regularization and $k$ values. Generates `graphics/exploration_analysis.pdf`.
    - `scripts/analyze_uncertainty.py`: Analyzes the evolution of agent uncertainty (sigma) and breaks down feature-level uncertainty contributions. Generates `graphics/uncertainty_analysis.pdf`.
    - `scripts/analyze_weights.py`: Processes simulation results to show how agent weights converge to ground truth for each planet. Generates `graphics/weight_evolution_analysis.pdf`.
    - `scripts/analyze_planet_distributions.py`: Shows distributions of selected planets overall and over time. Generates `graphics/planet_distribution_analysis.pdf`.
    - `scripts/visualize_simulation_arms.py`: Generates detailed timelines of arm selections for every individual simulation. Generates PDFs in `graphics/simulations/`.
- **Development**: `scripts/testing.ipynb` is used for prototyping and iterative testing of agent behaviors.
- **Manual play**: `scripts/game_in_console.py` lets a human choose arms in the console.

## Adding or tweaking agents
- Derive from `LinearRegressionAgent` to reuse prediction/uncertainty helpers.
- Expose tunable hyperparameters via `__init__`.
- Remember to update `get_recommendations` if you want UI/logs to reflect new belief calculations.
