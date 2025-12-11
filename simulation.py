import numpy as np
import pandas as pd

def run_single_game(agent_class, game_class, n_trials=150):
    game = game_class(n_trials=n_trials)
    agent = agent_class() 
    history = []
    
    for t in range(n_trials):
        context = game.current_context.copy()
        
        # 1. ASK AGENT FOR DATA
        # The agent handles the math (mu +/- 1.96sigma) internally
        recs = agent.get_recommendations(context, k=1.96)
        
        # 2. DECISION
        arm_idx = agent.select_arm(context)
        
        # 3. STEP
        reward, done, info = game.step(arm_idx)
        agent.update(context, arm_idx, reward)
        
        # 4. LOGGING
        record = {
            "agent": agent_class.__name__,
            "trial": t + 1,
            "choice": arm_idx,
            "reward": reward,
        }
        
        # Unpack the agent's calculations into columns
        for i, data in enumerate(recs):
            record[f"mu_{i}"]    = data['mean']
            record[f"sigma_{i}"] = data['sigma']
            record[f"lower_{i}"] = data['lower']
            record[f"upper_{i}"] = data['upper']
            
        history.append(record)
        if done: break
            
    return pd.DataFrame(history)


def run_batch_simulation(agent_classes, game_class, n_simulations=10, n_trials=150):
    """
    Runs the simulation for multiple agents and multiple repetitions (participants).
    
    Args:
        agent_classes: Dictionary { "Name": AgentClass }
        game_class: The game class.
        n_simulations: How many 'participants' per agent (Paper used 47).
        n_trials: Trials per game.
        
    Returns:
        pd.DataFrame: Aggregated results for all agents and simulations.
    """
    all_results = []
    
    for agent_name, agent_cls in agent_classes.items():
        print(f"Simulating Agent: {agent_name} ({n_simulations} runs)...")
        
        for sim_id in range(n_simulations):
            # Run one game
            df = run_single_game(agent_cls, game_class, n_trials)
            
            # Add metadata
            df['simulation_id'] = sim_id
            df['agent_name'] = agent_name
            
            all_results.append(df)
            
    # Combine all into one big DataFrame
    full_data = pd.concat(all_results, ignore_index=True)
    return full_data

if __name__ == "__main__":
    # Example Usage
    from game import MiningInSpaceGame
    from agents import RandomAgent, MeanTrackingAgent, LinearUCBAgent, LinearThompsonAgent, GPUCBAgent, GPThompsonAgent
    
    agents_to_test = {
        "Random": RandomAgent,
        "MeanTracker": MeanTrackingAgent,
        "Linear-UCB": LinearUCBAgent,
        "Linear-Thompson": LinearThompsonAgent,
        'Gaussian-UCB': GPUCBAgent,
        'Gaussian-Thompson': GPThompsonAgent
    }
    
    # Run a small batch to test
    results = run_batch_simulation(agents_to_test, MiningInSpaceGame, n_simulations=100, n_trials=150)
    
    # Print summary
    print("\n--- Average Total Score per Agent ---")
    print(results.groupby('agent_name')['reward'].sum() / 100)