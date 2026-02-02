import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup and Load Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', "simulation_results.csv")
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')
multipliers = [0.0, 0.5, 1.96, 5.0, 10.0]

if not os.path.exists(DATA_FILE):
    print(f"Data file {DATA_FILE} not found. Please run scripts/simulation.py first.")
    exit(1)

results = pd.read_csv(DATA_FILE)

# 2. Process Data
# Compare multipliers for a fixed regularization (e.g. 100)
results = results[results['reg'] == 100]

# Calculate means across simulations for each (agent_name, trial)
plot_data = results.groupby(['agent_name', 'trial', 'k'])['reward_received'].mean().reset_index()

# 3. Create Plots
plt.figure(figsize=(10, 6))

# Colors for different multipliers
palette = sns.color_palette("viridis", n_colors=len(multipliers))

sns.lineplot(
    data=plot_data,
    x='trial',
    y='reward_received',
    hue='k',
    palette=palette,
    linewidth=2,
    errorbar=None
)

plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline (Random)')
plt.title('Impact of Exploration Multiplier (k) on Linear UCB Performance (reg=100)', fontsize=14)
plt.ylabel('Average Reward')
plt.xlabel('Trial Number')
plt.legend(title='Exploration Multiplier (k)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(GRAPHICS_DIR, 'ucb_multiplier_sweep.pdf')
plt.savefig(output_file)

print(f"Plot saved to {output_file}")
