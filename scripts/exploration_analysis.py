import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import run_batch_simulation
from game import MiningInSpaceGame
from agents import LinearUCBAgent

# 1. Setup and Load Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', "exploration_grid_results.csv")
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')
reg_values = [1, 10, 100, 1000]
k_values = [0.0, 0.5, 1.96, 5.0, 10.0]

if not os.path.exists(DATA_FILE):
    print(f"{DATA_FILE} not found. Running new simulations...")
    agents_to_test = {}
    
    # Grid for UCB
    for reg in reg_values:
        for k in k_values:
            name = f"UCB (reg={reg}, k={k})"
            agents_to_test[name] = lambda r=reg, val=k: LinearUCBAgent(regularization=r, exploration_multiplier=val)

    results = run_batch_simulation(
        agents_to_test, 
        MiningInSpaceGame, 
        n_simulations=30, 
        n_trials=100, 
        output_path=DATA_FILE
    )
else:
    results = pd.read_csv(DATA_FILE)

# 2. Process Data
# Extract Regularization and K for plotting
results['reg_val'] = results['agent_name'].str.extract(r'reg=(\d+)').astype(int)
results['k_val'] = results['agent_name'].str.extract(r'k=([\d.]+)').astype(float)

# Calculate exploration rate (mean of exploration binary) per agent and trial
expl_data = results.groupby(['agent_name', 'reg_val', 'k_val', 'trial'], dropna=False)['exploration'].mean().reset_index()

# 3. Create Plots
# We'll use a grid of UCB exploration rates (Faceted by Reg, lines are K)

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
palette_k = sns.color_palette("rocket_r", n_colors=len(k_values))

for i, reg in enumerate(reg_values):
    ax = axes[i]
    subset = expl_data[expl_data['reg_val'] == reg]
    sns.lineplot(
        data=subset,
        x='trial',
        y='exploration',
        hue='k_val',
        palette=palette_k,
        ax=ax,
        legend=(i == len(reg_values) - 1)
    )
    ax.set_title(f'UCB (reg={reg})', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('P(Explore)' if i == 0 else '')
    ax.grid(True, alpha=0.3)
    if i == len(reg_values) - 1:
        ax.legend(title='k value', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('UCB Exploration Behavior across Regularization and K-Multiplier', fontsize=16)
plt.tight_layout(rect=(0, 0, 0.9, 0.95))
output_file = os.path.join(GRAPHICS_DIR, 'exploration_rates.pdf')
plt.savefig(output_file, bbox_inches='tight')

print(f"Analysis complete. Plot saved to {output_file}")
