import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup and Load Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', "simulation_results.csv")
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')
reg_values = [1, 10, 100, 1000]

if not os.path.exists(DATA_FILE):
    print(f"Data file {DATA_FILE} not found. Please run scripts/simulation.py first.")
    exit(1)

results = pd.read_csv(DATA_FILE)

# 2. Process Data
# We want to compare regularization values for a fixed k (default k=1.96)
results = results[results['k'] == 1.96]

# Calculate means across simulations for each (agent_name, trial)
plot_data = results.groupby(['agent_name', 'trial', 'reg'])['reward_received'].mean().reset_index()

# 3. Create Plots
fig, ax = plt.subplots(figsize=(10, 6))

# Colors for different regularization terms
palette = sns.color_palette("viridis", n_colors=len(reg_values))

# Plot UCB
sns.lineplot(
    data=plot_data,
    x='trial',
    y='reward_received',
    hue='reg',
    ax=ax,
    palette=palette,
    linewidth=2,
    errorbar=None
)
ax.set_title('Linear UCB Performance by Regularization (k=1.96)', fontsize=14)
ax.set_ylabel('Average Reward')
ax.set_xlabel('Trial Number')
ax.legend(title='Regularization')
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(GRAPHICS_DIR, 'regularization_sweep.pdf')
plt.savefig(output_file)

print(f"Plot saved to {output_file}")
