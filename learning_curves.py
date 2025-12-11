import matplotlib.pyplot as plt
import seaborn as sns
from simulation import run_batch_simulation
from game import MiningInSpaceGame
from agents import (RandomAgent, MeanTrackingAgent, LinearUCBAgent, 
                    LinearThompsonAgent, GPUCBAgent, GPThompsonAgent)

# 1. Setup and Run Simulation
agents_to_test = {
    "Random": RandomAgent,
    "MeanTracker": MeanTrackingAgent,
    # "Linear-UCB": LinearUCBAgent,
    # "Linear-Thompson": LinearThompsonAgent,
    "Linear-UCB-No-Pseudo": lambda: LinearUCBAgent(pseudo_observations=False),
    "Linear-Thompson-No-Pseudo": lambda: LinearThompsonAgent(pseudo_observations=False),
    "Gaussian-UCB": GPUCBAgent,
    "Gaussian-Thompson": GPThompsonAgent
}

print("Running simulation... this may take a minute.")
results = run_batch_simulation(agents_to_test, MiningInSpaceGame, n_simulations=10, n_trials=150)

# 2. Process Data for Plotting (First Graph)
# Average reward at each trial number for the combined plot
learning_curves = results.groupby(['agent_name', 'trial'])['reward'].mean().reset_index()

# ---------------------------------------------------------
# Graph 1: Combined Learning Curves (Averages Only)
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=learning_curves, 
    x='trial', 
    y='reward', 
    hue='agent_name',
    linewidth=2
)

plt.axhline(y=50, color='gray', linestyle='--', label='Chance Level (50)')
plt.title('Agent Learning Curves: Average Score per Trial', fontsize=16)
plt.xlabel('Trial Number', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.legend(title='Agent Strategy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curves.pdf')
plt.show()

# ---------------------------------------------------------
# Graph 2: Individual Agents with Confidence Intervals
# ---------------------------------------------------------
# We use the raw 'results' dataframe here. 
# Seaborn automatically aggregates the 'n_simulations' to show the 
# mean (solid line) and the 95% confidence interval (shaded area).

g = sns.relplot(
    data=results,
    x='trial', 
    y='reward', 
    col='agent_name', 
    col_wrap=3,           # Arranges plots in a grid of 3 columns
    kind='line',          # Line plot type
    height=4, 
    aspect=1.5,
    linewidth=2,
    errorbar=('ci', 95)                 # 95% Confidence Interval (default)
)

# Add reference lines and styling to each subplot
for ax in g.axes.flat:
    ax.axhline(y=50, color='gray', linestyle='--', label='Chance Level')
    ax.grid(True, alpha=0.3)

# Adjust overall title and layout
g.fig.suptitle('Individual Agent Performance (with 95% Confidence Intervals)', fontsize=16, y=1.02)
g.set_axis_labels('Trial Number', 'Reward')

plt.savefig('individual_agents_ci.pdf')
plt.show()