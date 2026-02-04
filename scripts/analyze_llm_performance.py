import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', "llm_simulation_results.csv")
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')

def analyze_llm_performance():
    if not os.path.exists(DATA_FILE):
        print(f"Data file {DATA_FILE} not found. Please run scripts/run_llm_simulation.py first.")
        return

    # Load data
    results = pd.read_csv(DATA_FILE)

    # Visualization
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    # Plot learning curve with confidence intervals
    # lineplot automatically calculates mean and confidence interval (default 95% CI)
    ax = sns.lineplot(
        data=results,
        x='trial',
        y='reward_received',
        color='blue',
        linewidth=2,
        label='LLM Agent (Gemini 3 Flash)'
    )

    # Add a horizontal line for the baseline (Safe Planet D reward is approx 50)
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.6, label='Baseline (Safe Planet D)')

    # Labels and Titles
    plt.title('LLM Agent Learning Curve: Reward per Trial (with 95% CI)', fontsize=16)
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('Average Reward Received', fontsize=12)
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')
    
    # Text annotation for final average reward
    avg_final = results[results['trial'] == results['trial'].max()]['reward_received'].mean()
    plt.annotate(f'Final Avg: {avg_final:.1f}', 
                 xy=(results['trial'].max(), avg_final),
                 xytext=(results['trial'].max()-10, avg_final+10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()

    # Save the plot
    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    output_file = os.path.join(GRAPHICS_DIR, 'llm_performance_analysis.pdf')
    plt.savefig(output_file)
    print(f"LLM performance plot saved to {output_file}")

if __name__ == "__main__":
    analyze_llm_performance()
