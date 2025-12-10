import numpy as np
import pandas as pd

def run_single_game(agent_class, game_class, n_trials=150):
    """
    Simulates a single participant playing the game.
    
    Args:
        agent_class: The class of the agent to instantiate (e.g., LinearUCBAgent).
        game_class: The class of the game (MiningInSpaceGame).
        n_trials: Number of trials to run (default 150).
        
    Returns:
        pd.DataFrame: A record of every trial (trial, choice, reward, etc.).
    """
    # Instantiate specific game and agent for this run
    game = game_class(n_trials=n_trials)
    agent = agent_class() 
    
    history = []
    
    for t in range(n_trials):
        # 1. Observe current context
        # We copy it because the game updates the context after the step
        context = game.current_context.copy()
        
        # 2. Agent chooses an arm
        # Note: Even context-blind agents accept the context argument (and ignore it)
        arm_idx = agent.select_arm(context)
        
        # 3. Game executes the step
        reward, done, info = game.step(arm_idx)
        
        # 4. Agent learns from the outcome
        agent.update(context, arm_idx, reward)
        
        # 5. Log Data
        history.append({
            "agent": agent_class.__name__,
            "trial": t + 1,
            "context": str(context), # Storing as string for CSV simplicity
            "choice": arm_idx,
            "reward": reward,
            "optimal_choice": np.argmax(info['means']), # From game debug info
            "regret": np.max(info['means']) - info['means'][arm_idx]
        })
        
        if done:
            break
            
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