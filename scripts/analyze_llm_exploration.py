import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', "llm_simulation_results.csv")
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')

def analyze_llm_exploration():
    if not os.path.exists(DATA_FILE):
        print(f"Data file {DATA_FILE} not found. Please run scripts/run_llm_simulation.py first.")
        return

    # Load data
    results = pd.read_csv(DATA_FILE)

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Plot exploration probability over time
    # lineplot automatically calculates mean and confidence interval (default 95% CI)
    # y='exploration' represents the probability of exploring at each trial
    ax = sns.lineplot(
        data=results,
        x='trial',
        y='exploration',
        color='orange',
        linewidth=2,
        label='LLM Agent (Gemini 3 Flash)'
    )

    # Labels and Titles
    plt.title('LLM Agent Exploration Behavior: Probability of Exploration over Time', fontsize=16)
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('P(Explore)', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # Add insight annotation
    initial_expl = results[results['trial'] <= 5]['exploration'].mean()
    final_expl = results[results['trial'] >= 45]['exploration'].mean()
    
    plt.text(5, 0.9, f'Initial Explore Rate: {initial_expl:.1%}', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(35, 0.1, f'Final Explore Rate: {final_expl:.1%}', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    output_file = os.path.join(GRAPHICS_DIR, 'llm_exploration_analysis.pdf')
    plt.savefig(output_file)
    print(f"LLM exploration plot saved to {output_file}")

if __name__ == "__main__":
    analyze_llm_exploration()
