import numpy as np

class MiningInSpaceGame:
    def __init__(self, n_trials=150):
        """
        Initializes the CMAB task based on the paper parameters.
        
        Args:
            n_trials: Fixed to 150 in the paper[cite: 139].
        """
        self.n_trials = n_trials
        self.current_trial = 0
        self.total_score = 0
        self.history = []
        
        # Context names based on the screenshot [cite: 143-145]
        self.context_names = ["Mercury", "Krypton", "Nobelium"]
        
        # Initialize first state
        self.current_context = self._generate_context()

    def _generate_context(self):
        """
        Generates the binary context (s) for the current trial.
        Contexts are binary (+ or -).
        Mapped to 1 (On) and -1 (Off) to match the paper's score range.
        """
        while True:
            # Generate random -1 or 1
            ctx = np.random.choice([-1, 1], size=3)
            
            # Constraint: Situations only containing - or + were not used [cite: 160]
            # i.e., Exclude [-1, -1, -1] and [1, 1, 1]
            if np.all(ctx == -1) or np.all(ctx == 1):
                continue
            return ctx

    def _calculate_expected_rewards(self, s):
        """
        Calculates expected reward (mean) for each arm based on Equations 15-18 .
        s represents the context vector [s1, s2, s3].
        """
        # s[0]=Mercury, s[1]=Krypton, s[2]=Nobelium
        # TODO randomize planet order
        # Planet 1: 50 + 15*s1 - 15*s2 [cite: 131]
        mu_1 = 50 + 15 * s[0] - 15 * s[1]
        
        # Planet 2: 50 + 15*s2 - 15*s3 [cite: 132]
        mu_2 = 50 + 15 * s[1] - 15 * s[2]
        
        # Planet 3: 50 + 15*s3 - 15*s1 [cite: 133]
        mu_3 = 50 + 15 * s[2] - 15 * s[0]
        
        # Planet 4: 50 (Constant mean) 
        mu_4 = 50
        
        return [mu_1, mu_2, mu_3, mu_4]

    def step(self, arm_choice):
        """
        Executes a turn.
        
        Args:
            arm_choice: int (0-3) representing the chosen planet.
        """
        if self.current_trial >= self.n_trials:
            return 0, True, {"msg": "Game Over"}

        # 1. Calculate Expected Means for current context
        means = self._calculate_expected_rewards(self.current_context)
        
        # 2. Generate Rewards: Mean + Independent Noise for EACH arm
        # The paper defines epsilon_{k,t} ~ N(0, 5) individually 
        noises = np.random.normal(loc=0, scale=5.0, size=4)
        
        # Calculate potential rewards for ALL arms (latent state)
        potential_rewards = [m + n for m, n in zip(means, noises)]
        
        # Pick the reward for the chosen arm
        reward = potential_rewards[arm_choice]
        
        # 3. Log history
        self.history.append({
            "trial": self.current_trial + 1,
            "context": self.current_context.copy(),
            "choice": arm_choice + 1,
            "reward": reward,
            "optimal_choice": np.argmax(means) + 1,
            "latent_rewards": potential_rewards # Useful for debugging/analysis
        })

        # 4. Update State
        self.total_score += reward
        self.current_trial += 1
        done = self.current_trial >= self.n_trials
        
        # 5. Generate new context for next turn (if not done)
        if not done:
            self.current_context = self._generate_context()
            
        return reward, done, {
            "means": means,
            "context_names": [
                f"{name}: {'+' if val == 1 else '-'}" 
                for name, val in zip(self.context_names, self.current_context)
            ]
        }