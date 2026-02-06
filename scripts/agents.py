import numpy as np

class LinearRegressionAgent:
    """
    Base Contextual Model using Recursive Least Squares.
    """
    def __init__(self, n_arms=4, n_features=3, regularization=100):
        self.n_arms = n_arms
        self.n_features = n_features + 1 # +1 for intercept
        self.regularization = regularization
        
        # Storage for RLS matrices
        self.A_inv = [] 
        self.b = []      
        self.counts = np.zeros(self.n_arms, dtype=int)

        for _ in range(n_arms):
            # Standard RLS initialization:
            # A_inv = (1/delta) * I.  delta is small ridge factor.
            current_A_inv = np.eye(self.n_features) * self.regularization
            current_b = np.zeros(self.n_features)
            
            self.A_inv.append(current_A_inv)
            self.b.append(current_b)

    def _get_features(self, context):
        """Adds intercept: [1, s1, s2, s3]"""
        return np.concatenate(([1.0], context))

    def get_arm_params(self, arm_idx):
        """Returns beta weights: beta = A^-1 * b"""
        return self.A_inv[arm_idx] @ self.b[arm_idx]

    def predict_with_uncertainty(self, context, arm_idx):
        """
        Returns predictive mean and standard deviation.
        """
        x = self._get_features(context)
        beta = self.get_arm_params(arm_idx)
        
        # Equation 4: mean = beta * x
        mean = np.dot(beta, x)
        
        # Variance = x.T * A_inv * x
        variance = x @ self.A_inv[arm_idx] @ x
        
        # Ensure non-negative variance
        variance = max(variance, 1e-6)
        
        return mean, np.sqrt(variance)

    def select_arm(self, context):
        """Returns choice and info dict"""
        return int(0), {} 

    def update(self, context, arm, reward):
        """Recursive Least Squares Update"""
        self.counts[arm] += 1
        x = self._get_features(context)
        A_inv = self.A_inv[arm]
        
        # Sherman-Morrison update
        num = np.outer(A_inv @ x, x @ A_inv)
        den = 1 + x @ A_inv @ x
        self.A_inv[arm] = A_inv - (num / den)
        
        self.b[arm] += reward * x

    def get_recommendations(self, context, k=1.96):
        """
        Returns the full package of data needed for the UI or Analysis.
        
        Args:
            context: The current state/context.
            k (float): Confidence multiplier (1.96 = 95% CI).
            
        Returns:
            List of dicts, one for each arm:
            [{'mean': 50.0, 'sigma': 10.0, 'lower': 30.4, 'upper': 69.6}, ...]
        """
        results = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            
            # Ensure sigma is not NaN (for safety)
            if np.isnan(sigma): sigma = 0.0
                
            results.append({
                "mean": mu,
                "sigma": sigma,
                "lower": mu - (k * sigma),
                "upper": mu + (k * sigma)
            })
        return results
    
    def get_feature_weights(self, feature_names=None):
        """
        Returns a human-readable dictionary of the current weights (betas) for every arm.
        
        Args:
            feature_names (list, optional): List of strings for feature names (e.g., ['Price', 'Color', 'Size']). 
                                            If None, defaults to ['Feat_1', 'Feat_2', ...].
        """
        # Default naming if none provided
        if feature_names is None:
            # -1 because the first weight is always the Intercept
            feature_names = [f"Feat_{i+1}" for i in range(self.n_features - 1)]
            
        weight_report = {}
        
        for arm_idx in range(self.n_arms):
            # Calculate beta = A_inv @ b
            weights = self.get_arm_params(arm_idx)
            
            # Map weights to names
            # The first weight is always the Intercept (bias)
            arm_data = {"Intercept": weights[0]}
            
            # Map the rest to the provided feature names
            for name, w in zip(feature_names, weights[1:]):
                arm_data[name] = w
                
            weight_report[f"Arm_{arm_idx}"] = arm_data
            
        return weight_report

    def get_feature_uncertainties(self, context, feature_names=None):
        """
        Returns per-arm feature uncertainty contributions for the given context.
        contrib_i = (x_i ** 2) * A_inv[i, i]
        """
        if feature_names is None:
            feature_names = [f"Feat_{i+1}" for i in range(self.n_features - 1)]

        x = self._get_features(context)
        uncertainty_report = {}

        for arm_idx in range(self.n_arms):
            A_inv = self.A_inv[arm_idx]
            arm_data = {}
            for idx, name in enumerate(feature_names, start=1):
                contrib = (x[idx] ** 2) * A_inv[idx, idx]
                arm_data[name] = contrib

            uncertainty_report[f"Arm_{arm_idx}"] = arm_data

        return uncertainty_report


class LinearUCBAgent(LinearRegressionAgent):
    """
    Linear Regression with UCB.
    """
    def __init__(self, n_arms=4, n_features=3, regularization=100, exploration_multiplier=1.96):
        super().__init__(n_arms, n_features, regularization)
        self.exploration_multiplier = exploration_multiplier

    def _get_feature_names(self):
        default_names = ["Mercury", "Krypton", "Nobelium"]
        if self.n_features - 1 <= len(default_names):
            return default_names[:self.n_features - 1]
        return [f"Feature {i + 1}" for i in range(self.n_features - 1)]

    def _select_uncertain_feature_indices(self, context, arm_idx, top_k=1):
        x = self._get_features(context)
        A_inv = self.A_inv[arm_idx]
        contributions = []

        for i in range(1, self.n_features):
            contrib = (x[i] ** 2) * A_inv[i, i]
            contributions.append((i - 1, contrib))

        contributions.sort(key=lambda item: item[1], reverse=True)

        if not contributions or top_k <= 0:
            return []

        if top_k >= len(contributions):
            return [idx for idx, _ in contributions]

        cutoff = contributions[top_k - 1][1]
        return [
            idx
            for idx, value in contributions
            if value > cutoff or np.isclose(value, cutoff, rtol=1e-9, atol=1e-12)
        ]

    def _select_value_feature_indices(self, context, arm_idx, top_k=2):
        x = self._get_features(context)
        beta = self.get_arm_params(arm_idx)
        contributions = []

        for i in range(1, self.n_features):
            contrib = beta[i] * x[i]
            contributions.append((i - 1, contrib))

        contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        return [idx for idx, _ in contributions[:top_k]]

    def _format_signed(self, value, precision=1):
        if abs(value) < 1e-6:
            value = 0.0
        return f"{value:+.{precision}f}"

    def _format_value_contributions(self, context, arm_idx, precision=1):
        x = self._get_features(context)
        beta = self.get_arm_params(arm_idx)
        feature_names = self._get_feature_names()
        parts = []

        for i, name in enumerate(feature_names, start=1):
            state = "on" if context[i - 1] > 0 else "off"
            contrib = beta[i] * x[i]
            parts.append(f"{name} {state} ({self._format_signed(contrib, precision)})")

        return ", ".join(parts)

    def _format_learning_context(self, context):
        feature_names = self._get_feature_names()
        parts = []

        for idx, name in enumerate(feature_names):
            state = "on" if context[idx] > 0 else "off"
            parts.append(f"{name} {state}")

        return ", ".join(parts)

    def select_arm(self, context):
        ucb_values = []
        means = []
        sigmas = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            # Paper uses 1.96 for 95% CI [cite: 106]
            ucb = mu + self.exploration_multiplier * sigma
            ucb_values.append(ucb)
            means.append(mu)
            sigmas.append(sigma)
            
        # Algorithm 1: Choose argmax
        choice = np.argmax(ucb_values)

        max_mean = np.max(means)
        mean_ties = np.isclose(means, max_mean)
        if np.sum(mean_ties) > 1:
            exploration = 1
        else:
            mean_best_idx = int(np.argmax(means))
            exploration = 0 if choice == mean_best_idx else 1

        if exploration == 1:
            reason = "learn how each signal affects this planet"
            context_phrase = self._format_learning_context(context)
            explanation = f"Exploring Planet {choice + 1} to {reason}: {context_phrase}."
        else:
            reason = "estimated effects favor this planet"
            context_phrase = self._format_value_contributions(context, choice)
            explanation = f"Exploit Planet {choice + 1} because {reason}: {context_phrase}."

        feature_names = self._get_feature_names()
        context_features = list(feature_names)

        return int(choice), {
            "exploration": exploration,
            "intent": "explore" if exploration == 1 else "exploit",
            "intent_reason": reason,
            "context_features": context_features,
            "explanation": explanation,
            "chosen_mean": means[choice],
            "chosen_sigma": sigmas[choice],
            "chosen_ucb": ucb_values[choice],
        }
