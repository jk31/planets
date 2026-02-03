import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_all_simulations(file_path, output_dir=os.path.join("graphics", "simulations")):
    # Load data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get unique simulations
    simulations = df[['agent_name', 'simulation_id', 'reg', 'k']].drop_duplicates()
    
    total_sims = len(simulations)
    print(f"Found {total_sims} simulations. Generating plots...")

    for idx, row in simulations.iterrows():
        agent_name = row['agent_name']
        sim_id = row['simulation_id']
        reg = row['reg']
        k = row['k']

        # Filter data for this specific simulation
        sim_data = df[(df['agent_name'] == agent_name) & (df['simulation_id'] == sim_id)].copy()
        sim_data = sim_data.sort_values(by=['trial'])

        # Extract arm mapping
        mapping = {}
        for i in range(4):
            col = f'mapping_arm_{i}'
            if col in sim_data.columns:
                planet = sim_data[col].values[0]
                mapping[i] = planet
            else:
                mapping[i] = "?"

        # Setup Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Step function for arm selection
        ax.step(sim_data['trial'], sim_data['choice_arm_index'], where='post', color='gray', alpha=0.5, label='Selected Arm')

        # Scatter points for optimality
        optimal = sim_data[sim_data['is_optimal'] == 1]
        suboptimal = sim_data[sim_data['is_optimal'] == 0]

        ax.scatter(optimal['trial'], optimal['choice_arm_index'], c='green', marker='o', label='Optimal Choice', s=20, zorder=3)
        ax.scatter(suboptimal['trial'], suboptimal['choice_arm_index'], c='red', marker='x', label='Suboptimal Choice', s=20, zorder=3)

        # Configure Axes
        ax.set_yticks(range(4))
        yticklabels = [f"Arm {i} ({mapping[i]})" for i in range(4)]
        ax.set_yticklabels(yticklabels)
        
        ax.set_xlabel("Trial")
        ax.set_ylabel("Selected Arm (Planet)")
        ax.set_title(f"Arm Selection over Time\nAgent: {agent_name} | Sim ID: {sim_id}")
        
        # Add Grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Text box for parameters
        mean_reward = sim_data["reward_received"].mean()
        textstr = '\n'.join((
            f'Regularization: {reg}',
            f'Exploration Mul (k): {k}',
            f'Total Trials: {len(sim_data)}',
            f'Mean Reward: {mean_reward:.2f}'
        ))
        
        # Place text box in upper right
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        ax.legend(loc='upper right')

        # Adjust layout
        plt.tight_layout(rect=(0, 0, 0.85, 1))

        # Save file
        # Filename includes reg, k, and sim_id
        filename = f"reg_{reg}_k_{k}_sim_{sim_id}.pdf"
        output_file = os.path.join(output_dir, filename)
        plt.savefig(output_file)
        plt.close(fig) # Close to save memory

    print(f"All {total_sims} plots saved to {output_dir}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_FILE = os.path.join(BASE_DIR, "data", "simulation_results.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "graphics", "simulations")
    visualize_all_simulations(RESULTS_FILE, output_dir=OUTPUT_DIR)
