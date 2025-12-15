import numpy as np

class MiningInSpaceGame:
    def __init__(self, n_trials=150):
        self.n_trials = n_trials
        self.current_trial = 0
        self.total_score = 0
        self.history = []
        
        self.context_names = ["Mercury", "Krypton", "Nobelium"]
        
        # DEFINITION OF LOGIC LABELS
        # 0 -> A: Mercury/Krypton
        # 1 -> B: Krypton/Nobelium
        # 2 -> C: Nobelium/Mercury
        # 3 -> D: Safe (Constant)
        self.logic_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        
        # Randomize the buttons ONCE
        self.arm_permutation = np.random.permutation(4)
        
        self.current_context = self._generate_context()

    def _generate_context(self):
        while True:
            ctx = np.random.choice([-1, 1], size=3)
            if np.all(ctx == -1) or np.all(ctx == 1):
                continue
            return ctx

    def _calculate_expected_rewards(self, s):
        # Calculate canonical means (0-3)
        mu_0 = 50 + 15 * s[0] - 15 * s[1] # Logic A
        mu_1 = 50 + 15 * s[1] - 15 * s[2] # Logic B
        mu_2 = 50 + 15 * s[2] - 15 * s[0] # Logic C
        mu_3 = 50                         # Logic D
        
        canonical_means = [mu_0, mu_1, mu_2, mu_3]
        
        # Shuffle according to the game instance's random layout
        return [canonical_means[i] for i in self.arm_permutation]

    def step(self, arm_choice):
        if self.current_trial >= self.n_trials:
            return 0, True, {"msg": "Game Over"}

        means = self._calculate_expected_rewards(self.current_context)
        noises = np.random.normal(loc=0, scale=5.0, size=4)
        potential_rewards = [m + n for m, n in zip(means, noises)]
        
        reward = potential_rewards[arm_choice]
        
        # --- LOGGING THE LABELS INTERNALLY ---
        canonical_idx = self.arm_permutation[arm_choice] # 0-3
        canonical_lbl = self.logic_labels[canonical_idx] # A-D
        
        self.history.append({
            "trial": self.current_trial + 1,
            "context": self.current_context.copy(),
            "arm_choice_index": arm_choice,           # Physical button (0-3)
            "canonical_logic_index": canonical_idx,   # Math index (0-3)
            "canonical_logic_label": canonical_lbl,   # Label (A-D) <-- SAVED HERE
            "reward": reward,
            "optimal_choice": np.argmax(means),
            "latent_rewards": potential_rewards,
            "arm_permutation": self.arm_permutation.tolist()
        })
        # -------------------------------------

        self.total_score += reward
        self.current_trial += 1
        done = self.current_trial >= self.n_trials
        
        if not done:
            self.current_context = self._generate_context()
            
        return reward, done, {}