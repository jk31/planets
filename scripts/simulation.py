import pandas as pd

def run_single_game(agent_class, game_class, n_trials=100):
    game = game_class(n_trials=n_trials)
    agent = agent_class() 
    
    full_simulation_log = []
    
    for t in range(n_trials):
        context = game.current_context.copy()
        
        # 1. Agent Decision & Beliefs
        recs = agent.get_recommendations(context, k=1.96)
        arm_idx, info = agent.select_arm(context)
        
        # 2. Game Step
        reward, done, info_game = game.step(arm_idx)
        agent.update(context, arm_idx, reward)
        
        # 3. Logging Standard Data
        game_log = game.history[-1]
        
        record = {
            "agent": agent_class.__name__,
            "trial": t + 1,
            
            # Decisions
            "choice_arm_index": arm_idx,
            "choice_planet_label": game_log["canonical_planet_label"], 
            "is_optimal": 1 if arm_idx == game_log["optimal_choice"] else 0,
            "exploration": info.get("exploration", 1),
            
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
            
            # Flatten the dictionary for the CSV/DataFrame
            record[f"w_intercept_arm_{arm_i}"] = w_data["Intercept"]
            record[f"w_mercury_arm_{arm_i}"]   = w_data["Mercury"]
            record[f"w_krypton_arm_{arm_i}"]   = w_data["Krypton"]
            record[f"w_nobelium_arm_{arm_i}"]  = w_data["Nobelium"]

        # 5. Save Mapping (e.g. Arm 0 -> 'D')
        for i, planet_id in enumerate(game_log["arm_permutation"]):
            record[f"mapping_arm_{i}"] = game.planet_labels[planet_id]

        # 6. Save Agent Uncertainty stats
        for i, data in enumerate(recs):
            record[f"agent_mu_arm_{i}"]     = data['mean']
            record[f"agent_sigma_arm_{i}"]  = data['sigma']
            
        full_simulation_log.append(record)
        if done: break
            
    return pd.DataFrame(full_simulation_log)


def run_batch_simulation(agent_classes, game_class, n_simulations=50, n_trials=100, output_path=None):
    """
    Runs the simulation for multiple agents and multiple repetitions.
    """
    all_results = []
    
    for agent_name, agent_cls in agent_classes.items():
        print(f"Simulating Agent: {agent_name} ({n_simulations} runs)...")
        
        for sim_id in range(n_simulations):
            df = run_single_game(agent_cls, game_class, n_trials)
            
            df['simulation_id'] = sim_id
            df['agent_name'] = agent_name
            
            all_results.append(df)
            
    final_df = pd.concat(all_results, ignore_index=True)
    
    if output_path:
        final_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
    return final_df

if __name__ == "__main__":
    import argparse
    import os
    from game import MiningInSpaceGame
    from agents import LinearUCBAgent

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_OUTPUT = os.path.join(BASE_DIR, 'data', "simulation_results.csv")

    parser = argparse.ArgumentParser(description="Run Mining in Space simulations.")
    parser.add_argument("--n_simulations", type=int, default=50, help="Number of simulations per agent")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials per simulation")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save results")
    args = parser.parse_args()

    # Define a standard suite of agents for CLI usage
    reg_values = [1, 10, 100, 1000]
    agents_to_test = {}
    for reg in reg_values:
        agents_to_test[f"UCB (reg={reg})"] = lambda r=reg: LinearUCBAgent(regularization=r)

    run_batch_simulation(
        agents_to_test, 
        MiningInSpaceGame, 
        n_simulations=args.n_simulations, 
        n_trials=args.n_trials, 
        output_path=args.output
    )
