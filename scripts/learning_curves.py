import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import run_batch_simulation
from game import MiningInSpaceGame
from agents import LinearUCBAgent

# 1. Setup and Run Simulation
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', "simulation_results.csv")
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')
reg_values = [1, 10, 100, 1000]

if os.path.exists(DATA_FILE):
    print(f"Loading existing simulation data from {DATA_FILE}...")
    results = pd.read_csv(DATA_FILE)
else:
    agents_to_test = {}
    for reg in reg_values:
        agents_to_test[f"UCB (reg={reg})"] = lambda r=reg: LinearUCBAgent(regularization=r)

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

# Extract Regularization for plotting
plot_data['reg_val'] = plot_data['agent_name'].str.extract(r'reg=(\d+)').astype(int)

# 3. Create Plots
fig, ax = plt.subplots(figsize=(10, 6))

# Colors for different regularization terms
palette = sns.color_palette("viridis", n_colors=len(reg_values))

# Plot UCB
sns.lineplot(
    data=plot_data,
    x='trial',
    y='reward_received',
    hue='reg_val',
    ax=ax,
    palette=palette,
    linewidth=2,
    errorbar=None
)
ax.set_title('Linear UCB Performance by Regularization', fontsize=14)
ax.set_ylabel('Average Reward')
ax.set_xlabel('Trial Number')
ax.legend(title='Regularization')
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(GRAPHICS_DIR, 'regularization_sweep.pdf')
plt.savefig(output_file)

print(f"Plot saved to {output_file}")
