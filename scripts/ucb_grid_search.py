import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import run_batch_simulation
from game import MiningInSpaceGame
from agents import LinearUCBAgent

# 1. Setup Experiment Grid
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', "ucb_grid_results.csv")
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')
reg_values = [1, 10, 100, 1000]
k_values = [0.0, 0.5, 1.96, 5.0, 10.0]

if os.path.exists(DATA_FILE):
    print(f"Loading existing simulation data from {DATA_FILE}...")
    results = pd.read_csv(DATA_FILE)
else:
    agents_to_test = {}
    for reg in reg_values:
        for k in k_values:
            name = f"UCB(reg={reg}, k={k})"
            # Use closures to capture current reg and k
            agents_to_test[name] = lambda r=reg, val=k: LinearUCBAgent(regularization=r, exploration_multiplier=val)

    print(f"Running grid search for {len(agents_to_test)} configurations...")
    # Using 30 simulations and 100 trials to balance runtime and significance
    results = run_batch_simulation(
        agents_to_test, 
        MiningInSpaceGame, 
        n_simulations=30, 
        n_trials=100, 
        output_path=DATA_FILE
    )

# 2. Process Data for Visualization
# Extract parameters from agent_name
results['reg'] = results['agent_name'].str.extract(r'reg=(\d+)').astype(int)
results['k'] = results['agent_name'].str.extract(r'k=([\d.]+)').astype(float)

# Calculate average reward per configuration
grid_summary = results.groupby(['reg', 'k'])['reward_received'].mean().reset_index()

# 3. Visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.3)

# --- Plot A: Heatmap of Overall Average Reward ---
ax0 = fig.add_subplot(gs[0])
pivot_data = grid_summary.pivot(index='reg', columns='k', values='reward_received')
sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax0)
ax0.set_title('Overall Average Reward: Regularization vs Exploration Multiplier (k)', fontsize=14)
ax0.set_ylabel('Regularization')
ax0.set_xlabel('Exploration Multiplier (k)')

# --- Plot B: Faceted Learning Curves ---
# We'll use a standard lineplot with faceting by regularization
ax1 = fig.add_subplot(gs[1])
learning_data = results.groupby(['reg', 'k', 'trial'])['reward_received'].mean().reset_index()

# For a cleaner plot, we'll use relplot-like faceting manually or just a grouped lineplot
# Since we are in a single figure, let's use hue=k and style=reg or separate subplots
sns.lineplot(
    data=learning_data,
    x='trial',
    y='reward_received',
    hue='k',
    style='reg',
    palette='viridis',
    ax=ax1
)
ax1.set_title('Learning Curves: Grouped by k and Regularization', fontsize=14)
ax1.set_ylabel('Average Reward')
ax1.set_xlabel('Trial Number')
ax1.legend(title='Parameters', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(GRAPHICS_DIR, 'ucb_grid_search.pdf')
plt.savefig(output_file)
print(f"Plots saved to {output_file}")
