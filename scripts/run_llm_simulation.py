import os
import sys
import argparse
import pandas as pd
from game import MiningInSpaceGame
from llm_agents import LLMAgent

def run_llm_simulation(n_trials=20, output_path=None):
    print(f"Starting LLM Agent Simulation ({n_trials} trials)...")
    
    # Initialize game and agent
    game = MiningInSpaceGame(n_trials=n_trials)
    agent = LLMAgent(n_arms=4, n_trials=n_trials)
    
    # Hidden mapping (Button -> Planet Label)
    mapping = {i: game.planet_labels[p] for i, p in enumerate(game.arm_permutation)}
    print(f"Hidden Mapping (Button -> Planet): {mapping}")
    print("-" * 50)
    
    total_reward = 0
    results = []
    
    for t in range(n_trials):
        context = game.current_context.copy()
        mercury, krypton, nobelium = context
        
        print(f"Trial {t+1}/{n_trials}")
        
        # 1. Agent Decision
        arm_idx, info = agent.select_arm(context)
        explanation = info.get("explanation", "No explanation provided.")
        intent = info.get("intent", "unknown")
        exploration = info.get("exploration", 0)
        
        # 2. Game Step
        reward, done, _ = game.step(arm_idx)
        agent.update(context, arm_idx, reward)
        
        # 3. Get game log for verification
        game_log = game.history[-1]
        planet_label = mapping[arm_idx]
        is_optimal = 1 if arm_idx == game_log["optimal_choice"] else 0
        
        # 4. Record trial data
        record = {
            "simulation_id": 0,
            "agent_name": "LLM_Gemini_3_Flash",
            "trial": t + 1,
            "context_mercury": mercury,
            "context_krypton": krypton,
            "context_nobelium": nobelium,
            "choice_arm_index": arm_idx,
            "choice_planet_label": planet_label,
            "is_optimal": is_optimal,
            "intent": intent,
            "exploration": exploration,
            "explanation": explanation,
            "reward_received": reward,
        }
        
        # Add mapping to record
        for i, planet_lbl in mapping.items():
            record[f"mapping_arm_{i}"] = planet_lbl
            
        results.append(record)
        total_reward += reward
        
        # 5. Print Progress
        print(f"Context: M={mercury}, K={krypton}, N={nobelium} | Chose: Button {arm_idx} ({planet_label}) | Reward: {reward}")
        # print explanation
        print(f"Intent: {intent} | Exploration: {exploration}")
        print(f"Explanation: {explanation}")
        print("-" * 30)
        
        if done:
            break
            
    print(f"Simulation Complete. Final Score: {total_reward}")
    
    if output_path:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
        
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_OUTPUT = os.path.join(BASE_DIR, 'data', "llm_simulation_results.csv")

    parser = argparse.ArgumentParser(description="Run LLM Agent simulation and save results.")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save results")
    args = parser.parse_args()
    
    run_llm_simulation(n_trials=args.trials, output_path=args.output)
