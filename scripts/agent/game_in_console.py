import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from game import MiningInSpaceGame
from agents import LinearUCBAgent

def _prompt_float(label, default, min_value=None):
    prompt = f"{label} [{default}]: "
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Invalid number. Try again.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}. Try again.")
            continue
        return value


def _prompt_int(label, default, min_value=None):
    prompt = f"{label} [{default}]: "
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Invalid integer. Try again.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}. Try again.")
            continue
        return value


def play_console_game():
    """
    A simple text loop to play the game manually.
    """
    print("--- WELCOME TO MINING IN SPACE ---")
    print("Goal: Maximize emeralds mined over 50 trials.")
    print("Contexts: Mercury, Krypton, Nobelium can be ON (+) or OFF (-).")
    print("Planets: 1, 2, 3, 4\n")

    print("Advisor setup:")
    print("1) Default parameters")
    print("2) Customize parameters")

    choice = input("Select an option (1-2): ").strip()
    if choice == "2":
        regularization = _prompt_int("Regularization", 100, min_value=0)
        exploration_multiplier = _prompt_float("Exploration multiplier (k)", 1.96, min_value=0.0)
        agent = LinearUCBAgent(
            regularization=regularization,
            exploration_multiplier=exploration_multiplier,
        )
    else:
        agent = LinearUCBAgent()  # Advisor agent

    game = MiningInSpaceGame(integer_rewards=True)
    
    while True:
        # Display Status
        print(f"\nTrial: {game.current_trial + 1}/{game.n_trials}")
        print(f"Current Score: {game.total_score}")
        
        # Display Context (The key to the puzzle)
        ctx_display = [
            f"{name}: {'+' if val > 0 else '-'}" 
            for name, val in zip(game.context_names, game.current_context)
        ]
        print(f"CURRENT GALAXY STATE: { ' | '.join(ctx_display) }")
        
        # Advisor section
        context = game.current_context.copy()
        _, advisor_info = agent.select_arm(context)
        print(f"STRATEGIC ADVICE: {advisor_info['explanation']}")
        
        # Get User Input
        try:
            choice = int(input("Choose a planet to mine (1-4): "))
            if choice < 1 or choice > 4:
                raise ValueError
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")
            continue
            
        # Execute Step (Convert 1-based input to 0-based index)
        reward, done, info = game.step(choice - 1)
        
        # Update advisor agent with what actually happened
        agent.update(context, choice - 1, reward)
        
        print(f"Result: You mined {reward} emeralds!")
        
        if done:
            print("\n--- GAME OVER ---")
            print(f"Final Score: {game.total_score}")
            print(f"Average Score: {int(round(game.total_score / game.n_trials))}")
            break

if __name__ == "__main__":
    play_console_game()
