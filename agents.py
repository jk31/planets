import numpy as np

class LinearRegressionAgent:
    """
    Base Contextual Model using Recursive Least Squares.
    Reference: [cite: 64-71]
    """
    def __init__(self, n_arms=4, n_features=3, regularization=100):
        self.n_arms = n_arms
        self.n_features = n_features + 1 # +1 for intercept
        self.regularization = regularization
        
        # Storage for RLS matrices
        self.A_inv = [] 
        self.b = []      

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
        """Placeholder for subclasses"""
        return 0 

    def update(self, context, arm, reward):
        """Recursive Least Squares Update"""
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


class LinearUCBAgent(LinearRegressionAgent):
    """
    Linear Regression with UCB.
    Reference: Algorithm 1 [cite: 105]
    """
    def __init__(self, n_arms=4, n_features=3, regularization=100, exploration_multiplier=1.96):
        super().__init__(n_arms, n_features, regularization)
        self.exploration_multiplier = exploration_multiplier

    def select_arm(self, context):
        ucb_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            # Paper uses 1.96 for 95% CI [cite: 106]
            ucb = mu + self.exploration_multiplier * sigma
            ucb_values.append(ucb)
            
        # Algorithm 1: Choose argmax
        return np.argmax(ucb_values)


class LinearThompsonAgent(LinearRegressionAgent):
    """
    Linear Regression with Thompson Sampling.
    Reference: Algorithm 2 [cite: 115]
    """
    def select_arm(self, context):
        sampled_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            # Sample y* ~ N(mu, sigma)
            sample = np.random.normal(mu, sigma)
            sampled_values.append(sample)
            
        # Algorithm 2: Choose argmax
        return np.argmax(sampled_values)
