import numpy as np
from softmax import softmax  # Assuming you put the helper in softmax.py

class RandomAgent:
    """
    Context-blind: Picks every arm with equal probability.
    Reference: [cite: 56]
    """
    def __init__(self, n_arms=4):
        self.n_arms = n_arms

    def select_arm(self, context=None):
        return np.random.choice(self.n_arms)

    def update(self, context, arm, reward):
        pass


class MeanTrackingAgent:
    """
    Context-blind: Tracks mean rewards and uses Softmax to choose.
    Reference: [cite: 58-61]
    """
    def __init__(self, n_arms=4, gamma=0.1):
        self.n_arms = n_arms
        self.gamma = gamma
        self.counts = np.zeros(n_arms)
        self.estimates = np.full(n_arms, 50.0) # Init to grand mean

    def select_arm(self, context=None):
        probs = softmax(self.estimates, self.gamma)
        return np.random.choice(self.n_arms, p=probs)

    def update(self, context, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental mean update [cite: 59]
        self.estimates[arm] += (1/n) * (reward - self.estimates[arm])


class LinearRegressionAgent:
    """
    Base Contextual Model.
    Learns linear functions f_k(s) = beta * s.
    Reference: [cite: 64-71]
    """
    def __init__(self, n_arms=4, n_features=3):
        self.n_arms = n_arms
        self.n_features = n_features + 1 # +1 for intercept
        
        # Lists to store matrices for each arm
        self.A_inv = [] 
        self.b = []      

        for _ in range(n_arms):
            # Initialization using pseudo-observations logic [cite: 71]
            # Identity matrix for A_inv implies a ridge-like prior
            self.A_inv.append(np.eye(self.n_features)) 
            
            # Initialize b to target the starting mean of 50
            init_b = np.zeros(self.n_features)
            init_b[0] = 50.0 
            self.b.append(init_b)

    def _get_features(self, context):
        """Adds intercept: [1, s1, s2, s3]"""
        return np.concatenate(([1.0], context))

    def get_arm_params(self, arm_idx):
        """Returns beta weights: beta = A^-1 * b"""
        return self.A_inv[arm_idx] @ self.b[arm_idx]

    def predict_with_uncertainty(self, context, arm_idx):
        """
        Returns predictive mean and standard deviation for a specific arm.
        variance = x.T * A_inv * x
        """
        x = self._get_features(context)
        beta = self.get_arm_params(arm_idx)
        
        mean = np.dot(beta, x)
        
        # Variance of the prediction (epistemic uncertainty)
        # var = x.T @ A_inv @ x
        variance = x @ self.A_inv[arm_idx] @ x
        
        # Ensure non-negative variance (numerical stability)
        variance = max(variance, 1e-6)
        
        return mean, np.sqrt(variance)

    def select_arm(self, context):
        """Default to Greedy selection on means"""
        means = []
        for arm in range(self.n_arms):
            mu, _ = self.predict_with_uncertainty(context, arm)
            means.append(mu)
        return np.argmax(means)

    def update(self, context, arm, reward):
        """Recursive Least Squares Update"""
        x = self._get_features(context)
        A_inv = self.A_inv[arm]
        
        # Sherman-Morrison update
        num = np.outer(A_inv @ x, x @ A_inv)
        den = 1 + x @ A_inv @ x
        self.A_inv[arm] = A_inv - (num / den)
        
        self.b[arm] += reward * x


class LinearUCBAgent(LinearRegressionAgent):
    """
    Linear Regression with Upper Confidence Bound sampling.
    Reference: Algorithm 1 
    """
    def select_arm(self, context):
        ucb_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            
            # UCB = Mean + 1.96 * Std 
            # 1.96 corresponds to 95% confidence interval
            ucb = mu + 1.96 * sigma
            ucb_values.append(ucb)
            
        return np.argmax(ucb_values)


class LinearThompsonAgent(LinearRegressionAgent):
    """
    Linear Regression with Thompson Sampling (Probability Matching).
    Reference: Algorithm 2 
    """
    def select_arm(self, context):
        sampled_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            
            # Sample y* ~ N(mu, sigma^2) 
            # Note: The paper samples from the posterior of the function output.
            sample = np.random.normal(mu, sigma)
            sampled_values.append(sample)
            
        return np.argmax(sampled_values)


import numpy as np

class GaussianProcessAgent:
    """
    Base Contextual Model using Gaussian Processes.
    Learns non-parametric functions f_k(s).
    Reference: 
    """
    def __init__(self, n_arms=4, n_dims=3, lengthscale=1.0, noise_std=5.0):
        self.n_arms = n_arms
        self.lengthscale = lengthscale
        self.noise_variance = noise_std**2
        
        # Storage for training data (Contexts X and Rewards y)
        self.X = [[] for _ in range(n_arms)]
        self.y = [[] for _ in range(n_arms)]

        # Initialization with 10 pseudo-observations
        # "The Gaussian Process was initialized by the use of 10 pseudo-observations...
        # ...created from a Normal distribution with N(50, 10)" [cite: 71, 98]
        for arm in range(n_arms):
            for _ in range(10):
                # We generate random binary contexts for these priors
                ctx = np.random.randint(0, 2, size=n_dims)
                # Pseudo reward centered at 50 with std 10
                reward = np.random.normal(50, 10)
                self.X[arm].append(ctx)
                self.y[arm].append(reward)

    def kernel(self, x1, x2):
        """
        Squared Exponential Kernel.
        k(x, x') = exp( - ||x - x'||^2 / lambda )
        Reference: Equation 8 
        """
        sq_dist = np.sum((x1 - x2)**2)
        return np.exp(-sq_dist / self.lengthscale)

    def predict_with_uncertainty(self, context, arm_idx):
        """
        Calculates the posterior mean and variance for a specific arm.
        Reference: Equations 10, 11, 12 
        """
        X_train = np.array(self.X[arm_idx])
        y_train = np.array(self.y[arm_idx])
        x_new = np.array(context)
        N = len(X_train)
        
        # 1. Build Covariance Matrix K (N x N)
        # In a production system, we would update this incrementally (Cholesky update),
        # but for N=150, rebuilding is fast enough.
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                val = self.kernel(X_train[i], X_train[j])
                K[i, j] = val
                K[j, i] = val # Symmetric
        
        # Add noise variance to diagonal: K + sigma^2 * I
        K_noise = K + self.noise_variance * np.eye(N)
        
        # 2. Build K_star (Covariance between training data and new point)
        K_star = np.array([self.kernel(xi, x_new) for xi in X_train])
        
        # 3. K_star_star (Variance of new point)
        K_star_star = self.kernel(x_new, x_new) # Usually 1.0 for RBF
        
        # 4. Calculate Posterior Mean and Variance
        # We use np.linalg.solve for stability instead of direct inversion
        try:
            # Mean = K_star.T * (K + sigma^2 I)^-1 * y
            K_inv_y = np.linalg.solve(K_noise, y_train)
            mean = K_star.dot(K_inv_y)
            
            # Variance = k(x,x) - k(x, X) * (K + sigma^2 I)^-1 * k(X, x)
            K_inv_k_star = np.linalg.solve(K_noise, K_star)
            variance = K_star_star - K_star.dot(K_inv_k_star)
            
            # Numerical stability clip
            variance = max(variance, 1e-9)
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular (rare with noise added)
            mean = 50.0
            variance = 100.0

        return mean, np.sqrt(variance)

    def update(self, context, arm, reward):
        """Stores the new observation."""
        self.X[arm].append(context)
        self.y[arm].append(reward)
        # Note: The paper optimizes hyperparameters (lengthscale) here using gradient descent.
        # We skip the optimization step in this simulation for performance.

    def select_arm(self, context):
        """Default greedy selection."""
        means = []
        for arm in range(self.n_arms):
            mu, _ = self.predict_with_uncertainty(context, arm)
            means.append(mu)
        return np.argmax(means)


class GPUCBAgent(GaussianProcessAgent):
    """
    Gaussian Process with Upper Confidence Bound sampling.
    Reference: Algorithm 1 [cite: 105]
    """
    def select_arm(self, context):
        ucb_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            
            # UCB = Mean + 1.96 * Std (95% CI)
            ucb = mu + 1.96 * sigma
            ucb_values.append(ucb)
            
        return np.argmax(ucb_values)


class GPThompsonAgent(GaussianProcessAgent):
    """
    Gaussian Process with Thompson Sampling.
    Reference: Algorithm 2 [cite: 115]
    """
    def select_arm(self, context):
        sampled_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            
            # Sample y* ~ N(mu, sigma^2) (Marginal posterior)
            sample = np.random.normal(mu, sigma)
            sampled_values.append(sample)
            
        return np.argmax(sampled_values)