import numpy as np

class MiningInSpaceGame:
    def __init__(self, n_trials=50, integer_rewards=True):
        self.n_trials = n_trials
        self.current_trial = 0
        self.total_score = 0
        self.history = []
        self.integer_rewards = integer_rewards
        
        self.context_names = ["Mercury", "Krypton", "Nobelium"]
        
        # DEFINITION OF Planet LABELS
        # 0 -> A: Mercury/Krypton
        # 1 -> B: Krypton/Nobelium
        # 2 -> C: Nobelium/Mercury
        # 3 -> D: Safe (Constant)
        self.planet_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        
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
        """
        Calculates expected reward (mean) for each arm based on Equations 15-18 .
        s represents the context vector [s1, s2, s3].
        """
        # s[0]=Mercury, s[1]=Krypton, s[2]=Nobelium
        
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
        if self.current_trial >= self.n_trials:
            return 0, True, {"msg": "Game Over"}

        canonical_means = self._calculate_expected_rewards(self.current_context)
        physical_means = [canonical_means[int(self.arm_permutation[i])] for i in range(4)]
        noises = np.random.normal(loc=0, scale=5.0, size=4)
        potential_rewards = [m + n for m, n in zip(physical_means, noises)]

        if self.integer_rewards:
            potential_rewards = [int(np.round(value)) for value in potential_rewards]

        reward = potential_rewards[arm_choice]
        
        # --- LOGGING THE LABELS INTERNALLY ---
        canonical_idx = int(self.arm_permutation[arm_choice]) # 0-3
        canonical_lbl = self.planet_labels[canonical_idx] # A-D
        
        self.history.append({
            "trial": self.current_trial + 1,
            "context": self.current_context.copy(),
            "arm_choice_index": arm_choice,           # Physical button (0-3)
            "canonical_planet_index": canonical_idx,   # Math index (0-3)
            "canonical_planet_label": canonical_lbl,   # Label (A-D) <-- SAVED HERE
            "reward": reward,
            "optimal_choice": np.argmax(physical_means),
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
