import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel

class AgentResponse(BaseModel):
    arm_choice: int
    intent: str
    explanation: str

class LLMAgent:
    """
    LLM-powered agent for the Mining in Space game.
    Uses google-genai and Gemini models to make decisions.
    """
    def __init__(self, n_arms=4, n_trials=50, **kwargs):
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.history = []
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3-flash-preview"
        
        # Single prompt template with placeholders
        self.prompt_template = """
You are an expert agent playing a contextual bandit game called "Mining in Space".
There are 4 mining locations (Buttons 0, 1, 2, 3).
Each buttons's reward depends on three binary context signals: Mercury, Krypton, and Nobelium.
Signals are either 1 (on) or -1 (off).
The game lasts for {n_trials} trials. Your goal is to maximize total profit.

Game Rules:
- Each button (0, 1, 2, 3) has a unique, hidden reward function based on the three context signals.
- One button might have a constant reward, while others vary significantly based on specific signals.
- Your reward is the expected button reward plus some random noise.

Current Context:
- Mercury: {mercury}
- Krypton: {krypton}
- Nobelium: {nobelium}

Previous History (JSON format):
{history_json}

Task:
Analyze your history to deduce how each button's reward function reacts to the Mercury, Krypton, and Nobelium signals.
Then, choose the best button (0, 1, 2, or 3) for the current context to maximize your expected reward.

Strategic Intent:
- Classify your choice as either "explore" or "exploit".
- Use "explore" if you are trying a button to learn more about its reward function or mapping.
- Use "exploit" if you are confident in your knowledge and are choosing the button you believe has the highest expected reward for the current context.

Return the result as a JSON object with keys:
- "arm_choice": (int) The button index (0, 1, 2, or 3).
- "intent": (string) Either "explore" or "exploit".
- "explanation": (string) A brief explanation of your reasoning.
"""

    def select_arm(self, context):
        """
        Selects an arm (button) based on the current context and history.
        """
        # Prepare context
        mercury, krypton, nobelium = context
        
        # Prepare history as JSON string
        history_json = json.dumps(self.history, indent=2)
        
        # Fill prompt
        prompt = self.prompt_template.format(
            n_trials=self.n_trials,
            mercury=mercury,
            krypton=krypton,
            nobelium=nobelium,
            history_json=history_json
        )
        
        try:
            # Call Gemini
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AgentResponse,
                )
            )
            
            # Parse response
            result = response.parsed
            arm_choice = result.arm_choice
            intent = result.intent
            explanation = result.explanation
            
            # Ensure arm_choice is valid
            if not (0 <= arm_choice < self.n_arms):
                print(f"Warning: LLM chose out-of-bounds arm {arm_choice}. Defaulting to 0.")
                arm_choice = 0
                
        except Exception as e:
            print(f"Error calling LLM or parsing response: {e}. Defaulting to Arm 0.")
            arm_choice = 0
            intent = "error"
            explanation = f"Error: {str(e)}"
            
        return int(arm_choice), {
            "explanation": explanation,
            "exploration": 1 if intent == "explore" else 0,
            "intent": intent
        }

    def update(self, context, arm, reward):
        """
        Updates the agent's history with the result of the last action.
        """
        entry = {
            "trial": len(self.history) + 1,
            "context": {
                "Mercury": int(context[0]),
                "Krypton": int(context[1]),
                "Nobelium": int(context[2])
            },
            "button_chosen": int(arm),
            "reward_received": int(reward)
        }
        self.history.append(entry)
        
        # Save history to a JSON file as requested
        with open("history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    # Dummy methods to support scripts/agent/simulation.py logging
    def get_recommendations(self, context, k=1.96):
        return [{"mean": 0.0, "sigma": 0.0, "lower": 0.0, "upper": 0.0} for _ in range(self.n_arms)]

    def get_feature_uncertainties(self, context, feature_names=None):
        return {f"Arm_{i}": {name: 0.0 for name in (feature_names or [])} for i in range(self.n_arms)}

    def get_feature_weights(self, feature_names=None):
        return {f"Arm_{i}": {"Intercept": 0.0, **{name: 0.0 for name in (feature_names or [])}} for i in range(self.n_arms)}
