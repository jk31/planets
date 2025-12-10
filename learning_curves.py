import matplotlib.pyplot as plt
import seaborn as sns
from simulation import run_batch_simulation
from game import MiningInSpaceGame
from agents import (RandomAgent, MeanTrackingAgent, LinearUCBAgent, 
                    LinearThompsonAgent, GPUCBAgent, GPThompsonAgent)

# 1. Setup and Run Simulation
# We use 50 simulations per agent to get a smooth average (similar to the 47 participants in the paper [cite: 96])
agents_to_test = {
    "Random": RandomAgent,
    "MeanTracker": MeanTrackingAgent,
    "Linear-UCB": LinearUCBAgent,
    "Linear-Thompson": LinearThompsonAgent,
    "Gaussian-UCB": GPUCBAgent,
    "Gaussian-Thompson": GPThompsonAgent
}

print("Running simulation... this may take a minute.")
results = run_batch_simulation(agents_to_test, MiningInSpaceGame, n_simulations=10, n_trials=150)

# 2. Process Data for Plotting
# We calculate the average reward at each trial number (1-150) for each agent
learning_curves = results.groupby(['agent_name', 'trial'])['reward'].mean().reset_index()

# 3. Plotting
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=learning_curves, 
    x='trial', 
    y='reward', 
    hue='agent_name',
    linewidth=2
)

# Add reference lines
plt.axhline(y=50, color='gray', linestyle='--', label='Chance Level (50)')

plt.title('Agent Learning Curves: Average Score per Trial', fontsize=16)
plt.xlabel('Trial Number', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.legend(title='Agent Strategy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.savefig('learning_curves.pdf')