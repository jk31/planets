import os
import sys
import argparse
import pandas as pd
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from game import MiningInSpaceGame
from llm_agents import LLMAgent

def run_llm_simulation(n_simulations=1, n_trials=50, output_path=None):
    all_results = []
    
    for sim_id in range(n_simulations):
        print(f"\n--- Starting Simulation {sim_id + 1}/{n_simulations} ({n_trials} trials) ---")
        
        # Initialize game and agent for each simulation
        game = MiningInSpaceGame(n_trials=n_trials)
        agent = LLMAgent(n_arms=4, n_trials=n_trials)
        
        # Hidden mapping (Button -> Planet Label)
        mapping = {i: game.planet_labels[p] for i, p in enumerate(game.arm_permutation)}
        print(f"Hidden Mapping (Button -> Planet): {mapping}")
        
        total_reward = 0
        
        for t in range(n_trials):
            context = game.current_context.copy()
            mercury, krypton, nobelium = context
            
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
                "simulation_id": sim_id,
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
                
            all_results.append(record)
            total_reward += reward
            
            # 5. Print Progress
            print(f"Trial {t+1}: M={mercury}, K={krypton}, N={nobelium} | {intent.upper()} Button {arm_idx} ({planet_label}) | Reward: {reward}")
            # print explanation
            print(f"Explanation: {explanation}")
            
            if done:
                break
                
        print(f"Simulation {sim_id + 1} Complete. Final Score: {total_reward}")

        # Cleanup history.json after each simulation run to ensure 
        # a fresh start for the next agent
        if os.path.exists("history.json"):
            try:
                os.remove("history.json")
                print(f"Cleaned up history.json for simulation {sim_id + 1}")
            except Exception as e:
                print(f"Warning: Could not delete history.json: {e}")

    if output_path and all_results:
        df = pd.DataFrame(all_results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nAll results saved to {output_path}")

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
        
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DEFAULT_OUTPUT = os.path.join(BASE_DIR, 'data', 'llm', "simulation_results.csv")

    parser = argparse.ArgumentParser(description="Run multiple LLM Agent simulations and save results.")
    parser.add_argument("--simulations", type=int, default=20, help="Number of simulations to run")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials per simulation")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save results")
    args = parser.parse_args()
    
    run_llm_simulation(n_simulations=args.simulations, n_trials=args.trials, output_path=args.output)
