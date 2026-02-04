import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Setup and Load Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', "simulation_results.csv")
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')

if not os.path.exists(DATA_FILE):
    print(f"Data file {DATA_FILE} not found. Please run scripts/simulation.py first.")
    exit(1)

results = pd.read_csv(DATA_FILE)

# 2. Process Data for Total Uncertainty
# We want to see how the agent's uncertainty (sigma) for the chosen arm evolves
# Since choice_arm_index tells us which arm was picked, we extract that sigma.
def get_chosen_sigma(row):
    idx = int(row['choice_arm_index'])
    return row[f'agent_sigma_arm_{idx}']

results['chosen_sigma'] = results.apply(get_chosen_sigma, axis=1)

# Average across simulations
sigma_data = results.groupby(['agent_name', 'reg', 'k', 'trial'])['chosen_sigma'].mean().reset_index()

# 3. Process Data for Feature Uncertainty Breakdown
# We focus on the chosen arm's feature contributions
feature_names = ['mercury', 'krypton', 'nobelium']
for feat in feature_names:
    results[f'chosen_unc_{feat}'] = results.apply(
        lambda r: r[f'feature_uncertainty_{feat}_arm_{int(r["choice_arm_index"])}'], 
        axis=1
    )

# Average contributions across simulations
feat_unc_columns = [f'chosen_unc_{f}' for f in feature_names]
feat_data = results.groupby(['trial'])[feat_unc_columns].mean().reset_index()

# 4. Create Plots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot 1: Total Chosen Sigma over Time
reg_values = sorted(sigma_data['reg'].unique())
palette_k = sns.color_palette("viridis", n_colors=len(sigma_data['k'].unique()))

sns.lineplot(
    data=sigma_data,
    x='trial',
    y='chosen_sigma',
    hue='reg',
    style='k',
    ax=axes[0]
)
axes[0].set_title('Evolution of Uncertainty (Chosen Arm)', fontsize=14)
axes[0].set_xlabel('Trial Number')
axes[0].set_ylabel('Mean Sigma (Prediction Uncertainty)')
axes[0].grid(True, alpha=0.3)

# Plot 2: Feature Uncertainty Breakdown (Diagonal components)
# We'll use a stacked area plot for one specific (default or average) configuration
# Or just show them as lines. Let's do a stacked area for the average across all configs 
# to show relative importance, or pick a specific one.
# For simplicity, let's show them as lines for the average behavior.

feat_data_melted = feat_data.melt(id_vars='trial', var_name='Feature', value_name='Uncertainty')
feat_data_melted['Feature'] = feat_data_melted['Feature'].str.replace('chosen_unc_', '').str.capitalize()

sns.lineplot(
    data=feat_data_melted,
    x='trial',
    y='Uncertainty',
    hue='Feature',
    ax=axes[1]
)
axes[1].set_title('Diagonal Uncertainty Contributions (Chosen Arm)', fontsize=14)
axes[1].set_xlabel('Trial Number')
axes[1].set_ylabel('Variance Contribution (x_i^2 * A_inv[i,i])')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(GRAPHICS_DIR, 'uncertainty_analysis.pdf')
plt.savefig(output_file, bbox_inches='tight')
print(f"Analysis complete. Plot saved to {output_file}")
