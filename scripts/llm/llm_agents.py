import os
import json
import time
from collections import defaultdict
from statistics import mean, pstdev
from google import genai
from google.genai import types
from pydantic import BaseModel


class LLMDecisionError(RuntimeError):
    """Raised when the LLM cannot produce a valid decision."""

class AgentResponse(BaseModel):
    arm_choice: int
    intent: str
    explanation: str

class LLMAgent:
    """
    LLM-powered agent for the Mining in Space game.
    Uses google-genai and Gemini models to make decisions.
    """
    def __init__(self, n_arms=4, n_trials=50, max_attempts=3, retry_base_delay=0.5, **kwargs):
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.history = []
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_delay = float(retry_base_delay)
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3.1-flash-lite-preview"
        
        # Single prompt template with placeholders
        self.prompt_template = """
You are an expert agent playing a contextual bandit game called "Mining in Space".
There are 4 mining locations (Buttons 0, 1, 2, 3).
Each button's reward depends on three binary context signals: Mercury, Krypton, and Nobelium.
Signals are either 1 (on) or -1 (off).
The game lasts for {n_trials} trials. Your goal is to maximize total profit.

Important environment facts:
- Each button has a unique hidden reward function.
- Some buttons are context-sensitive and one may be near-constant.
- Rewards include random noise, so single outcomes are not reliable evidence.

Current Context:
- Trial: {current_trial} of {n_trials}
- Mercury: {mercury}
- Krypton: {krypton}
- Nobelium: {nobelium}

Button-Level Summary (JSON):
{arm_stats_json}

Context-Level Summary (JSON):
{context_stats_json}

Previous History (JSON):
{history_json}

Task:
Infer a compact reward model for each button from all available data.
For the current context, estimate expected reward and uncertainty for each button.
Choose the button with the best long-run value by balancing immediate reward and information value.

Reasoning principles:
- Use all history, not only recent outcomes.
- Prefer repeated patterns over isolated highs/lows.
- Compare top alternatives before deciding.
- Favor optimistic value: expected_reward + uncertainty_bonus.
- If options are close, resolving uncertainty has value.
- Avoid locking into one hypothesis too early when plausible alternatives remain under-tested.

Strategic Intent:
- Use "explore" when uncertainty materially drives your choice.
- Use "exploit" when one option is clearly best with strong evidence.
- Include confidence (high/medium/low) and key uncertainty in your explanation.

Return the result as a JSON object with keys:
- "arm_choice": (int) The button index (0, 1, 2, or 3).
- "intent": (string) Either "explore" or "exploit".
- "explanation": (string) One or two sentences explaining the decision.
"""

    def _context_key(self, context):
        return f"M={int(context[0])},K={int(context[1])},N={int(context[2])}"

    def _build_arm_stats(self):
        arm_rewards = {arm: [] for arm in range(self.n_arms)}
        arm_context_counts = {arm: defaultdict(int) for arm in range(self.n_arms)}

        for entry in self.history:
            arm = int(entry["button_chosen"])
            reward = int(entry["reward_received"])
            context = entry["context"]
            context_key = self._context_key(
                [context["Mercury"], context["Krypton"], context["Nobelium"]]
            )
            arm_rewards[arm].append(reward)
            arm_context_counts[arm][context_key] += 1

        stats = {}
        for arm in range(self.n_arms):
            rewards = arm_rewards[arm]
            top_contexts = sorted(
                arm_context_counts[arm].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:4]
            stats[f"button_{arm}"] = {
                "n_samples": len(rewards),
                "mean_reward": round(mean(rewards), 2) if rewards else None,
                "reward_std": (
                    round(pstdev(rewards), 2)
                    if len(rewards) > 1
                    else (0.0 if rewards else None)
                ),
                "n_contexts_seen": len(arm_context_counts[arm]),
                "most_seen_contexts": [
                    {"context": context_key, "count": count}
                    for context_key, count in top_contexts
                ],
            }

        return stats

    def _build_context_stats(self):
        context_arm_rewards = defaultdict(lambda: defaultdict(list))

        for entry in self.history:
            context = entry["context"]
            context_key = self._context_key(
                [context["Mercury"], context["Krypton"], context["Nobelium"]]
            )
            arm = int(entry["button_chosen"])
            reward = int(entry["reward_received"])
            context_arm_rewards[context_key][arm].append(reward)

        stats = {}
        for context_key, arm_data in sorted(context_arm_rewards.items()):
            arm_means = {
                f"button_{arm}": round(mean(rewards), 2)
                for arm, rewards in sorted(arm_data.items())
            }
            best_arm = max(arm_data.keys(), key=lambda arm: mean(arm_data[arm]))
            stats[context_key] = {
                "samples": int(sum(len(rewards) for rewards in arm_data.values())),
                "best_observed_button": int(best_arm),
                "best_observed_mean": round(mean(arm_data[best_arm]), 2),
                "arm_means": arm_means,
            }

        return stats

    def select_arm(self, context):
        """
        Selects an arm (button) based on the current context and history.
        """
        # Prepare context
        mercury, krypton, nobelium = context
        current_trial = len(self.history) + 1
        
        # Prepare summaries and history as JSON strings
        arm_stats_json = json.dumps(self._build_arm_stats(), indent=2)
        context_stats_json = json.dumps(self._build_context_stats(), indent=2)
        history_json = json.dumps(self.history, separators=(",", ":"))
        
        # Fill prompt
        prompt = self.prompt_template.format(
            n_trials=self.n_trials,
            current_trial=current_trial,
            mercury=mercury,
            krypton=krypton,
            nobelium=nobelium,
            arm_stats_json=arm_stats_json,
            context_stats_json=context_stats_json,
            history_json=history_json
        )
        
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AgentResponse,
                    )
                )

                result = response.parsed
                if result is None:
                    raise ValueError("response.parsed is None")

                if isinstance(result, dict):
                    arm_raw = result.get("arm_choice")
                    intent_raw = result.get("intent")
                    explanation_raw = result.get("explanation")
                else:
                    arm_raw = getattr(result, "arm_choice", None)
                    intent_raw = getattr(result, "intent", None)
                    explanation_raw = getattr(result, "explanation", None)

                if arm_raw is None or intent_raw is None:
                    raise ValueError("missing required fields in parsed response")

                arm_choice = int(arm_raw)
                intent = str(intent_raw).strip().lower()
                explanation = str(explanation_raw or "").strip()

                if not (0 <= arm_choice < self.n_arms):
                    raise ValueError(f"arm_choice out of bounds: {arm_choice}")
                if intent not in {"explore", "exploit"}:
                    raise ValueError(f"invalid intent: {intent}")
                if not explanation:
                    explanation = "No explanation provided."

                return arm_choice, {
                    "explanation": explanation,
                    "exploration": 1 if intent == "explore" else 0,
                    "intent": intent,
                    "llm_attempts": attempt,
                }

            except Exception as e:
                last_error = e
                if attempt < self.max_attempts:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    print(
                        f"LLM decision attempt {attempt}/{self.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

        raise LLMDecisionError(
            f"LLM decision failed after {self.max_attempts} attempts: {last_error}"
        )

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
