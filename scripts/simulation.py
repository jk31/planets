import pandas as pd
import numpy as np
import os
import argparse
from game import MiningInSpaceGame
from agents import LinearUCBAgent

def run_single_game(agent_class, game_class, n_trials=50, agent_kwargs=None):
    if agent_kwargs is None:
        agent_kwargs = {}
    
    game = game_class(n_trials=n_trials)
    agent = agent_class(**agent_kwargs) 
    
    full_simulation_log = []
    
    for t in range(n_trials):
        context = game.current_context.copy()
        
        # 1. Agent Decision & Beliefs
        recs = agent.get_recommendations(context, k=1.96)
        feature_uncertainties = agent.get_feature_uncertainties(
            context,
            feature_names=["Mercury", "Krypton", "Nobelium"]
        )
        arm_idx, info = agent.select_arm(context)
        
        # 2. Game Step
        reward, done, info_game = game.step(arm_idx)
        agent.update(context, arm_idx, reward)
        
        # 3. Logging Standard Data
        game_log = game.history[-1]
        
        record = {
            "trial": t + 1,
            
            # Decisions
            "choice_arm_index": arm_idx,
            "choice_planet_label": game_log["canonical_planet_label"], 
            "is_optimal": 1 if arm_idx == game_log["optimal_choice"] else 0,
            "exploration": info.get("exploration", 1),
            "intent": info.get("intent", "explore"),
            "intent_reason": info.get("intent_reason", ""),
            "explanation": info.get("explanation", ""),
            
            # Context
            "context_mercury": context[0],
            "context_krypton": context[1],
            "context_nobelium": context[2],
            "reward_received": reward,
        }

        # 4. Save Evolving Weights
        current_weights = agent.get_feature_weights(feature_names=["Mercury", "Krypton", "Nobelium"])
        
        for arm_i in range(len(game.planet_labels)): # Loop over 4 arms
            w_data = current_weights[f"Arm_{arm_i}"]
            record[f"w_intercept_arm_{arm_i}"] = w_data["Intercept"]
            record[f"w_mercury_arm_{arm_i}"]   = w_data["Mercury"]
            record[f"w_krypton_arm_{arm_i}"]   = w_data["Krypton"]
            record[f"w_nobelium_arm_{arm_i}"]  = w_data["Nobelium"]

        # 5. Save Feature Uncertainty Contributions
        for arm_i in range(len(game.planet_labels)):
            u_data = feature_uncertainties[f"Arm_{arm_i}"]
            record[f"feature_uncertainty_mercury_arm_{arm_i}"] = u_data["Mercury"]
            record[f"feature_uncertainty_krypton_arm_{arm_i}"] = u_data["Krypton"]
            record[f"feature_uncertainty_nobelium_arm_{arm_i}"] = u_data["Nobelium"]

        # 6. Save Mapping (e.g. Arm 0 -> 'D')
        for i, planet_id in enumerate(game_log["arm_permutation"]):
            record[f"mapping_arm_{i}"] = game.planet_labels[planet_id]

        # 7. Save Agent Uncertainty stats
        for i, data in enumerate(recs):
            record[f"agent_mu_arm_{i}"]     = data['mean']
            record[f"agent_sigma_arm_{i}"]  = data['sigma']
            
        full_simulation_log.append(record)
        if done: break
            
    return pd.DataFrame(full_simulation_log)


def run_grid_simulation(agent_class, game_class, reg_values, k_values, n_simulations=50, n_trials=50, output_path=None):
    """
    Runs a grid search over regularization and exploration multiplier.
    """
    all_results = []
    
    total_configs = len(reg_values) * len(k_values)
    config_idx = 1
    
    for reg in reg_values:
        for k in k_values:
            agent_name = f"UCB(reg={reg}, k={k})"
            print(f"[{config_idx}/{total_configs}] Simulating {agent_name} ({n_simulations} runs)...")
            
            for sim_id in range(n_simulations):
                df = run_single_game(agent_class, game_class, n_trials, agent_kwargs={'regularization': reg, 'exploration_multiplier': k})
                
                df['simulation_id'] = sim_id
                df['agent_name'] = agent_name
                df['reg'] = reg
                df['k'] = k
                
                all_results.append(df)
            config_idx += 1
            
    final_df = pd.concat(all_results, ignore_index=True)
    
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
    return final_df

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_OUTPUT = os.path.join(BASE_DIR, 'data', "simulation_results.csv")

    parser = argparse.ArgumentParser(description="Run consolidated Mining in Space simulations.")
    parser.add_argument("--n_simulations", type=int, default=50, help="Number of simulations per configuration")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials per simulation")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save results")
    parser.add_argument("--quick", action="store_true", help="Run a subset of configurations for speed")
    args = parser.parse_args()

    if args.quick:
        reg_values = [10, 100]
        k_values = [0.0, 1.96]
        n_sims = 5
        n_trials = 50
    else:
        reg_values = [1, 10, 100, 1000]
        k_values = [0.0, 0.5, 1.96, 5.0, 10.0]
        n_sims = args.n_simulations
        n_trials = args.n_trials

    run_grid_simulation(
        LinearUCBAgent, 
        MiningInSpaceGame, 
        reg_values=reg_values,
        k_values=k_values,
        n_simulations=n_sims, 
        n_trials=n_trials, 
        output_path=args.output
    )
