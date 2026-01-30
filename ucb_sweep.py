import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import run_batch_simulation
from game import MiningInSpaceGame
from agents import LinearUCBAgent

# 1. Setup and Run Simulation
DATA_FILE = "ucb_multiplier_results.csv"
multipliers = [0.0, 0.5, 1.96, 5.0, 10.0]

if os.path.exists(DATA_FILE):
    print(f"Loading existing simulation data from {DATA_FILE}...")
    results = pd.read_csv(DATA_FILE)
else:
    agents_to_test = {}
    for k in multipliers:
        agents_to_test[f"UCB (k={k})"] = lambda val=k: LinearUCBAgent(exploration_multiplier=val)

    print(f"Running simulation for {len(agents_to_test)} agent configurations...")
    # Using 50 simulations and 150 trials for better convergence visualization
    results = run_batch_simulation(
        agents_to_test, 
        MiningInSpaceGame, 
        n_simulations=50, 
        n_trials=150, 
        output_path=DATA_FILE
    )

# 2. Process Data
# Calculate means across simulations for each (agent_name, trial)
plot_data = results.groupby(['agent_name', 'trial'])['reward_received'].mean().reset_index()

# Extract Multiplier for plotting
plot_data['k_val'] = plot_data['agent_name'].str.extract(r'k=([\d.]+)').astype(float)

# 3. Create Plots
plt.figure(figsize=(10, 6))

# Colors for different multipliers
palette = sns.color_palette("viridis", n_colors=len(multipliers))

sns.lineplot(
    data=plot_data,
    x='trial',
    y='reward_received',
    hue='k_val',
    palette=palette,
    linewidth=2,
    errorbar=None
)

plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline (Random)')
plt.title('Impact of Exploration Multiplier (k) on Linear UCB Performance', fontsize=14)
plt.ylabel('Average Reward')
plt.xlabel('Trial Number')
plt.legend(title='Exploration Multiplier (k)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ucb_multiplier_sweep.pdf')

print("Plot saved to ucb_multiplier_sweep.pdf")
