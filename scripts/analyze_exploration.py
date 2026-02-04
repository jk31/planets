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

# 2. Process Data
expl_data = results.groupby(['agent_name', 'reg', 'k', 'trial'])['exploration'].mean().reset_index()
reg_values = sorted(expl_data['reg'].unique())
k_values = sorted(expl_data['k'].unique())

# 3. Create Plots
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
palette_k = sns.color_palette("rocket_r", n_colors=len(k_values))

for i, reg in enumerate(reg_values):
    ax = axes[i]
    subset = expl_data[expl_data['reg'] == reg]
    sns.lineplot(
        data=subset,
        x='trial',
        y='exploration',
        hue='k',
        palette=palette_k,
        ax=ax,
        legend=(i == len(reg_values) - 1)
    )
    ax.set_title(f'Reg={reg}', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('P(Explore)' if i == 0 else '')
    ax.grid(True, alpha=0.3)
    if i == len(reg_values) - 1:
        ax.legend(title='k value', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('Consolidated Exploration Behavior Analysis', fontsize=16)
output_file = os.path.join(GRAPHICS_DIR, 'exploration_analysis.pdf')
plt.savefig(output_file, bbox_inches='tight')
print(f"Analysis complete. Plot saved to {output_file}")
