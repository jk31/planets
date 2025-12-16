import pandas as pd

def run_single_game(agent_class, game_class, n_trials=150):
    game = game_class(n_trials=n_trials)
    agent = agent_class() 
    
    full_simulation_log = []
    
    for t in range(n_trials):
        context = game.current_context.copy()
        
        # 1. Agent Decision & Beliefs
        recs = agent.get_recommendations(context, k=1.96)
        arm_idx = agent.select_arm(context)
        
        # 2. Game Step
        reward, done, info = game.step(arm_idx)
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
            
            # Context
            "context_mercury": context[0],
            "context_krypton": context[1],
            "context_nobelium": context[2],
            "reward_received": reward,
        }

        # --- NEW: SAVE EVOLVING WEIGHTS ---
        # We use the helper function from the previous step
        # Note: We pass the specific context names from your Game class
        current_weights = agent.get_feature_weights(feature_names=["Mercury", "Krypton", "Nobelium"])
        
        for arm_i in range(game.planet_labels.__len__()): # Loop over 4 arms
            w_data = current_weights[f"Arm_{arm_i}"]
            
            # Flatten the dictionary for the CSV/DataFrame
            record[f"w_intercept_arm_{arm_i}"] = w_data["Intercept"]
            record[f"w_mercury_arm_{arm_i}"]   = w_data["Mercury"]
            record[f"w_krypton_arm_{arm_i}"]   = w_data["Krypton"]
            record[f"w_nobelium_arm_{arm_i}"]  = w_data["Nobelium"]
        # ----------------------------------

        # Save Mapping (e.g. Arm 0 -> 'D')
        for i, planet_id in enumerate(game_log["arm_permutation"]):
            record[f"mapping_arm_{i}"] = game.planet_labels[planet_id]

        # Save Agent Uncertainty stats
        for i, data in enumerate(recs):
            record[f"agent_mu_{i}"]     = data['mean']
            record[f"agent_sigma_{i}"]  = data['sigma']
            
        full_simulation_log.append(record)
        if done: break
            
    return pd.DataFrame(full_simulation_log)


def run_batch_simulation(agent_classes, game_class, n_simulations=10, n_trials=150):
    """
    Runs the simulation for multiple agents and multiple repetitions.
    """
    all_results = []
    
    for agent_name, agent_cls in agent_classes.items():
        print(f"Simulating Agent: {agent_name} ({n_simulations} runs)...")
        
        for sim_id in range(n_simulations):
            # Run one game
            df = run_single_game(agent_cls, game_class, n_trials)
            
            # Add simulation metadata
            df['simulation_id'] = sim_id
            df['agent_name'] = agent_name
            
            all_results.append(df)
            
    # Combine all into one big DataFrame
    full_data = pd.concat(all_results, ignore_index=True)
    return full_data