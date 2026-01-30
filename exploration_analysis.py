import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import run_batch_simulation
from game import MiningInSpaceGame
from agents import LinearUCBAgent, LinearThompsonAgent

# 1. Setup and Load Data
DATA_FILE = "simulation_results.csv"
reg_values = [1, 10, 100, 1000]

if not os.path.exists(DATA_FILE):
    print(f"{DATA_FILE} not found. Running new simulations...")
    agents_to_test = {}
    for reg in reg_values:
        agents_to_test[f"UCB (reg={reg})"] = lambda r=reg: LinearUCBAgent(regularization=r)
        agents_to_test[f"TS (reg={reg})"] = lambda r=reg: LinearThompsonAgent(regularization=r)

    results = run_batch_simulation(
        agents_to_test, 
        MiningInSpaceGame, 
        n_simulations=50, 
        n_trials=100, 
        output_path=DATA_FILE
    )
else:
    results = pd.read_csv(DATA_FILE)
    if 'exploration' not in results.columns:
        print("Existing data missing 'exploration' column. Re-running simulations...")
        agents_to_test = {}
        for reg in reg_values:
            agents_to_test[f"UCB (reg={reg})"] = lambda r=reg: LinearUCBAgent(regularization=r)
            agents_to_test[f"TS (reg={reg})"] = lambda r=reg: LinearThompsonAgent(regularization=r)

        results = run_batch_simulation(
            agents_to_test, 
            MiningInSpaceGame, 
            n_simulations=50, 
            n_trials=100, 
            output_path=DATA_FILE
        )

# 2. Process Data
# Calculate exploration rate (mean of exploration binary) per agent and trial
expl_data = results.groupby(['agent_name', 'trial'])['exploration'].mean().reset_index()

# Extract Algorithm and Regularization for plotting
expl_data['algorithm'] = expl_data['agent_name'].apply(lambda x: 'UCB' if 'UCB' in x else 'Thompson')
expl_data['reg_val'] = expl_data['agent_name'].str.extract(r'reg=(\d+)').astype(int)

# 3. Create Plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Colors for different regularization terms
palette = sns.color_palette("viridis", n_colors=len(reg_values))

# Plot UCB
sns.lineplot(
    data=expl_data[expl_data['algorithm'] == 'UCB'],
    x='trial',
    y='exploration',
    hue='reg_val',
    ax=axes[0],
    palette=palette,
    linewidth=2
)
axes[0].set_title('Linear UCB Exploration Rate', fontsize=14)
axes[0].set_ylabel('Exploration Rate (P(Explore))')
axes[0].legend(title='Regularization')

# Plot Thompson Sampling
sns.lineplot(
    data=expl_data[expl_data['algorithm'] == 'Thompson'],
    x='trial',
    y='exploration',
    hue='reg_val',
    ax=axes[1],
    palette=palette,
    linewidth=2
)
axes[1].set_title('Linear Thompson Sampling Exploration Rate', fontsize=14)
axes[1].legend(title='Regularization')

# Common styling
for ax in axes:
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Trial Number')
    ax.grid(True, alpha=0.3)

plt.suptitle('Agent Exploration Behavior Over Time', fontsize=16)
plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.savefig('exploration_rates.pdf')

print("Analysis complete. Plot saved to exploration_rates.pdf")
