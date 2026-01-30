import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import run_batch_simulation
from game import MiningInSpaceGame
from agents import LinearUCBAgent

# 1. Setup and Load Data
DATA_FILE = "ucb_multiplier_results.csv"
multipliers = [0.0, 0.5, 1.96, 5.0, 10.0]

# Check if we need to run simulations
run_sim = False
if not os.path.exists(DATA_FILE):
    run_sim = True
else:
    results = pd.read_csv(DATA_FILE)
    if 'exploration' not in results.columns:
        print("Existing data missing 'exploration' column. Re-running simulations...")
        run_sim = True

if run_sim:
    agents_to_test = {}
    for k in multipliers:
        agents_to_test[f"UCB (k={k})"] = lambda val=k: LinearUCBAgent(exploration_multiplier=val)

    print(f"Running simulation for {len(agents_to_test)} agent configurations...")
    results = run_batch_simulation(
        agents_to_test, 
        MiningInSpaceGame, 
        n_simulations=50, 
        n_trials=150, 
        output_path=DATA_FILE
    )

# 2. Process Data
# Calculate exploration rate per agent and trial
expl_data = results.groupby(['agent_name', 'trial'])['exploration'].mean().reset_index()

# Extract Multiplier for plotting
expl_data['k_val'] = expl_data['agent_name'].str.extract(r'k=([\d.]+)').astype(float)

# 3. Create Plot
plt.figure(figsize=(10, 6))

# Colors for different multipliers
palette = sns.color_palette("viridis", n_colors=len(multipliers))

sns.lineplot(
    data=expl_data,
    x='trial',
    y='exploration',
    hue='k_val',
    palette=palette,
    linewidth=2
)

plt.title('Impact of Exploration Multiplier (k) on UCB Exploration Rate', fontsize=14)
plt.ylabel('Exploration Rate (P(Explore))')
plt.xlabel('Trial Number')
plt.legend(title='Exploration Multiplier (k)')
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('ucb_exploration_rate.pdf')

print("Analysis complete. Plot saved to ucb_exploration_rate.pdf")
