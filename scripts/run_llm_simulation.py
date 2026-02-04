import os
import sys
import argparse
from game import MiningInSpaceGame
from llm_agents import LLMAgent

def run_llm_simulation(n_trials=20):
    print(f"Starting LLM Agent Simulation ({n_trials} trials)...")
    
    # Initialize game and agent
    game = MiningInSpaceGame(n_trials=n_trials)
    agent = LLMAgent(n_arms=4, n_trials=n_trials)
    
    # Print the hidden mapping for our reference (agent doesn't see this)
    mapping = {i: game.planet_labels[p] for i, p in enumerate(game.arm_permutation)}
    print(f"Hidden Mapping (Button -> Planet): {mapping}")
    print("-" * 50)
    
    total_reward = 0
    
    for t in range(n_trials):
        context = game.current_context.copy()
        mercury, krypton, nobelium = context
        
        print(f"Trial {t+1}/{n_trials}")
        print(f"Context: Mercury={mercury}, Krypton={krypton}, Nobelium={nobelium}")
        
        # 1. Agent Decision
        arm_idx, info = agent.select_arm(context)
        explanation = info.get("explanation", "No explanation provided.")
        
        # 2. Game Step
        reward, done, _ = game.step(arm_idx)
        agent.update(context, arm_idx, reward)
        
        total_reward += reward
        
        # 3. Print Progress
        planet_label = mapping[arm_idx]
        print(f"Agent chose Button {arm_idx} (Planet {planet_label})")
        print(f"Reward received: {reward}")
        print(f"Reasoning: {explanation}")
        print(f"Total Score: {total_reward}")
        print("-" * 30)
        
        if done:
            break
            
    print(f"Simulation Complete. Final Score: {total_reward}")

if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is available
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Run LLM Agent simulation.")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    args = parser.parse_args()
    
    run_llm_simulation(n_trials=args.trials)
