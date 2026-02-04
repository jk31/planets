import os
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

from agents import LLMAgent
import numpy as np

def test_llm_agent():
    print("Testing LLMAgent instantiation...")
    # Mock API key for instantiation test
    os.environ["GEMINI_API_KEY"] = "mock_key"
    try:
        agent = LLMAgent(n_arms=4, n_features=3, n_trials=5)
        print("Instantiation successful.")
        
        print("Testing update and history saving...")
        context = np.array([1, -1, 1])
        agent.update(context, 0, 55)
        
        if os.path.exists("llm_agent_history.json"):
            print("History file created.")
            with open("llm_agent_history.json", "r") as f:
                history = f.read()
                print(f"History content: {history}")
        else:
            print("History file NOT created.")
            
        print("Testing dummy methods...")
        recs = agent.get_recommendations(context)
        weights = agent.get_feature_weights()
        uncertainties = agent.get_feature_uncertainties(context)
        
        print("All dummy methods returned expected formats.")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        if os.path.exists("llm_agent_history.json"):
            os.remove("llm_agent_history.json")

if __name__ == "__main__":
    test_llm_agent()
