from game import MiningInSpaceGame
from agents import LinearUCBAgent

def play_console_game():
    """
    A simple text loop to play the game manually.
    """
    game = MiningInSpaceGame()
    agent = LinearUCBAgent() # Advisor agent
    
    print("--- WELCOME TO MINING IN SPACE ---")
    print("Goal: Maximize emeralds mined over 150 trials.")
    print("Contexts: Mercury, Krypton, Nobelium can be ON (+) or OFF (-).")
    print("Planets: 1, 2, 3, 4\n")

    while True:
        # Display Status
        print(f"\nTrial: {game.current_trial + 1}/{game.n_trials}")
        print(f"Current Score: {game.total_score:.2f}")
        
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
        
        print(f"Result: You mined {reward:.2f} emeralds!")
        
        if done:
            print("\n--- GAME OVER ---")
            print(f"Final Score: {game.total_score:.2f}")
            print(f"Average Score: {game.total_score/game.n_trials:.2f}")
            break

if __name__ == "__main__":
    play_console_game()