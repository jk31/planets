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
# Compare exploration rates for a fixed regularization (e.g. 100)
results = results[results['reg'] == 100]

# Calculate exploration rate per configuration and trial
expl_data = results.groupby(['agent_name', 'trial', 'k'])['exploration'].mean().reset_index()

# 3. Create Plot
plt.figure(figsize=(10, 6))

# Colors for different multipliers
palette = sns.color_palette("viridis", n_colors=len(multipliers))

sns.lineplot(
    data=expl_data,
    x='trial',
    y='exploration',
    hue='k',
    palette=palette,
    linewidth=2
)

plt.title('Impact of Exploration Multiplier (k) on UCB Exploration Rate (reg=100)', fontsize=14)
plt.ylabel('Exploration Rate (P(Explore))')
plt.xlabel('Trial Number')
plt.legend(title='Exploration Multiplier (k)')
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)

plt.tight_layout()
output_file = os.path.join(GRAPHICS_DIR, 'ucb_exploration_rate.pdf')
plt.savefig(output_file)

print(f"Analysis complete. Plot saved to {output_file}")
