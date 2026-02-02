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

# 2. Visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# --- Plot A: Overall Distribution of Selected Planets ---
ax0 = fig.add_subplot(gs[0, 0])
planet_counts = results['choice_planet_label'].value_counts().sort_index().reset_index()
planet_counts.columns = ['Planet', 'Count']
sns.barplot(data=planet_counts, x='Planet', y='Count', hue='Planet', palette="viridis", ax=ax0, legend=False)
ax0.set_title('Overall Distribution of Selected Planets', fontsize=14)
ax0.set_ylabel('Count')
ax0.set_xlabel('Planet')

# --- Plot B: Planet Selection over Trials (All Agents) ---
ax1 = fig.add_subplot(gs[0, 1])
# We'll bin trials to make the plot cleaner if there are many trials
results['trial_bin'] = (results['trial'] - 1) // 10 * 10
bin_dist = results.groupby(['trial_bin', 'choice_planet_label']).size().unstack(fill_value=0)
bin_dist_pct = bin_dist.div(bin_dist.sum(axis=1), axis=0)
bin_dist_pct.plot(kind='area', stacked=True, ax=ax1, colormap='viridis', alpha=0.7)
ax1.set_title('Planet Selection Frequency over Trials (Binned)', fontsize=14)
ax1.set_ylabel('Proportion')
ax1.set_xlabel('Trial Bin')
ax1.legend(title='Planet', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- Plot C: Planet Distribution by k (Aggregated over reg) ---
ax2 = fig.add_subplot(gs[1, 0])
k_dist = results.groupby(['k', 'choice_planet_label']).size().unstack(fill_value=0)
k_dist_pct = k_dist.div(k_dist.sum(axis=1), axis=0)
k_dist_pct.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis', alpha=0.8)
ax2.set_title('Planet Selection by Exploration Multiplier (k)', fontsize=14)
ax2.set_ylabel('Proportion')
ax2.set_xlabel('k value')
ax2.legend(title='Planet', bbox_to_anchor=(1.05, 1), loc='upper left')

# --- Plot D: Planet Distribution by reg (Aggregated over k) ---
ax3 = fig.add_subplot(gs[1, 1])
reg_dist = results.groupby(['reg', 'choice_planet_label']).size().unstack(fill_value=0)
reg_dist_pct = reg_dist.div(reg_dist.sum(axis=1), axis=0)
reg_dist_pct.plot(kind='bar', stacked=True, ax=ax3, colormap='viridis', alpha=0.8)
ax3.set_title('Planet Selection by Regularization', fontsize=14)
ax3.set_ylabel('Proportion')
ax3.set_xlabel('Regularization')
ax3.legend(title='Planet', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.suptitle('Planet Selection Distributions', fontsize=16)
output_file = os.path.join(GRAPHICS_DIR, 'planet_distribution_analysis.pdf')
plt.savefig(output_file)
print(f"Plots saved to {output_file}")
