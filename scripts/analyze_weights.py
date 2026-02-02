import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

def visualize_weight_evolution(csv_path, output_file="weight_evolution_all.pdf"):
    df = pd.read_csv(csv_path)
    
    reg_values = sorted(df['reg'].unique())
    k_values = sorted(df['k'].unique())
    
    ground_truth = {
        'A': {'Intercept': 50, 'Mercury': 20, 'Krypton': -10, 'Nobelium': 0},
        'B': {'Intercept': 50, 'Mercury': 0, 'Krypton': 20, 'Nobelium': -10},
        'C': {'Intercept': 50, 'Mercury': -10, 'Krypton': 0, 'Nobelium': 20},
        'D': {'Intercept': 50, 'Mercury': 0, 'Krypton': 0, 'Nobelium': 0}
    }

    features = ['Intercept', 'Mercury', 'Krypton', 'Nobelium']
    planets = ['A', 'B', 'C', 'D']
    colors = {'Intercept': 'black', 'Mercury': 'red', 'Krypton': 'green', 'Nobelium': 'blue'}

    with PdfPages(output_file) as pdf:
        for reg in reg_values:
            for k in k_values:
                agent_df = df[(df['reg'] == reg) & (df['k'] == k)].copy()
                if agent_df.empty: continue

                print(f"Processing weights for reg={reg}, k={k}...")
                
                planet_data = []
                for _, row in agent_df.iterrows():
                    trial = row['trial']
                    sim_id = row['simulation_id']
                    for arm_i in range(4):
                        planet_label = row[f'mapping_arm_{arm_i}']
                        record = {'trial': trial, 'sim_id': sim_id, 'planet': planet_label}
                        for feat in features:
                            col_name = f"w_{feat.lower()}_arm_{arm_i}"
                            record[feat] = row[col_name]
                        planet_data.append(record)
                        
                remapped_df = pd.DataFrame(planet_data)
                avg_weights = remapped_df.groupby(['planet', 'trial'])[features].mean().reset_index()
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
                axes = axes.flatten()
                
                for i, planet in enumerate(planets):
                    ax = axes[i]
                    p_df = avg_weights[avg_weights['planet'] == planet]
                    for feat in features:
                        ax.plot(p_df['trial'], p_df[feat], label=feat, color=colors[feat], linewidth=2)
                        ax.axhline(y=ground_truth[planet][feat], color=colors[feat], linestyle='--', alpha=0.5)
                    ax.set_title(f"Planet {planet}")
                    ax.grid(True, alpha=0.3)
                    if i >= 2: ax.set_xlabel("Trial")
                    if i % 2 == 0: ax.set_ylabel("Weight Value")
                        
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 0.95))
                plt.suptitle(f"Weight Evolution: reg={reg}, k={k}\n(Dashed lines = Ground Truth)", fontsize=16)
                plt.tight_layout(rect=(0, 0, 0.92, 0.95))
                
                pdf.savefig(fig)
                plt.close(fig)

    print(f"All weight visualizations saved to {output_file}")
 
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FILE = os.path.join(BASE_DIR, 'data', "simulation_results.csv")
    GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')
    
    visualize_weight_evolution(DATA_FILE, output_file=os.path.join(GRAPHICS_DIR, "weight_evolution_analysis.pdf"))
