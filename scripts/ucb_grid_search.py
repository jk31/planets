import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup and Load Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', "simulation_results.csv")
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')

if not os.path.exists(DATA_FILE):
    print(f"Data file {DATA_FILE} not found. Please run scripts/simulation.py first.")
    exit(1)

results = pd.read_csv(DATA_FILE)

# 2. Process Data for Visualization
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
ax1 = fig.add_subplot(gs[1])
learning_data = results.groupby(['reg', 'k', 'trial'])['reward_received'].mean().reset_index()

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
