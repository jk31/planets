import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import run_batch_simulation
from game import MiningInSpaceGame
from agents import LinearUCBAgent, LinearThompsonAgent

# 1. Setup and Run Simulation
DATA_FILE = "simulation_results.csv"
reg_values = [1, 10, 100, 1000]

if os.path.exists(DATA_FILE):
    print(f"Loading existing simulation data from {DATA_FILE}...")
    results = pd.read_csv(DATA_FILE)
else:
    agents_to_test = {}
    for reg in reg_values:
        agents_to_test[f"UCB (reg={reg})"] = lambda r=reg: LinearUCBAgent(regularization=r)
        agents_to_test[f"TS (reg={reg})"] = lambda r=reg: LinearThompsonAgent(regularization=r)

    print(f"Running simulation for {len(agents_to_test)} agent configurations...")
    results = run_batch_simulation(
        agents_to_test, 
        MiningInSpaceGame, 
        n_simulations=50, 
        n_trials=100, 
        output_path=DATA_FILE
    )

# 2. Process Data
# Calculate means across simulations for each (agent_name, trial)
plot_data = results.groupby(['agent_name', 'trial'])['reward_received'].mean().reset_index()

# Extract Algorithm and Regularization for plotting
plot_data['algorithm'] = plot_data['agent_name'].apply(lambda x: 'UCB' if 'UCB' in x else 'Thompson')
plot_data['reg_val'] = plot_data['agent_name'].str.extract(r'reg=(\d+)').astype(int)

# 3. Create Plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Colors for different regularization terms
palette = sns.color_palette("viridis", n_colors=len(reg_values))

# Plot UCB
sns.lineplot(
    data=plot_data[plot_data['algorithm'] == 'UCB'],
    x='trial',
    y='reward_received',
    hue='reg_val',
    ax=axes[0],
    palette=palette,
    linewidth=2,
    errorbar=None
)
axes[0].set_title('Linear UCB Performance', fontsize=14)
axes[0].set_ylabel('Average Reward')
axes[0].legend(title='Regularization')

# Plot Thompson Sampling
sns.lineplot(
    data=plot_data[plot_data['algorithm'] == 'Thompson'],
    x='trial',
    y='reward_received',
    hue='reg_val',
    ax=axes[1],
    palette=palette,
    linewidth=2,
    errorbar=None
)
axes[1].set_title('Linear Thompson Sampling Performance', fontsize=14)
axes[1].legend(title='Regularization')

# Common styling
for ax in axes:
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trial Number')
    ax.grid(True, alpha=0.3)

plt.suptitle('Comparison of Regularization Terms (No CI)', fontsize=16)
plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.savefig('regularization_sweep.pdf')

print("Plot saved to regularization_sweep.pdf")
