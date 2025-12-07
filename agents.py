import numpy as np
from softmax import softmax 

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
        self.estimates = np.full(n_arms, 50.0)

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
    Base Contextual Model using Recursive Least Squares.
    Reference: [cite: 64-71]
    """
    def __init__(self, n_arms=4, n_features=3):
        self.n_arms = n_arms
        self.n_features = n_features + 1 # +1 for intercept
        
        # Storage for RLS matrices
        self.A_inv = [] 
        self.b = []      

        # Initialization with 10 pseudo-observations 
        # We process these strictly as updates to ensure A_inv matches the data
        for _ in range(n_arms):
            # Start with "empty" state for RLS
            # A_inv = Large Identity (High variance/Low precision prior)
            # b = Zeros
            # We then "train" it on the 10 pseudo points.
            
            # Standard RLS initialization:
            # A_inv = (1/delta) * I.  delta is small ridge factor.
            current_A_inv = np.eye(self.n_features) * 100.0 
            current_b = np.zeros(self.n_features)
            
            # Generate 10 pseudo-observations
            for _ in range(10):
                # Random binary context (guess based on GP section which implies similar setup)
                fake_ctx = np.random.randint(0, 2, size=n_features)
                fake_x = np.concatenate(([1.0], fake_ctx))
                
                # Sample fake outcome from N(50, 10) 
                fake_y = np.random.normal(50, 10)
                
                # Perform RLS Update for this fake point
                # Sherman-Morrison
                num = np.outer(current_A_inv @ fake_x, fake_x @ current_A_inv)
                den = 1 + fake_x @ current_A_inv @ fake_x
                current_A_inv = current_A_inv - (num / den)
                
                current_b += fake_y * fake_x
            
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


class LinearUCBAgent(LinearRegressionAgent):
    """
    Linear Regression with UCB.
    Reference: Algorithm 1 [cite: 105]
    """
    def select_arm(self, context):
        ucb_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            # Paper uses 1.96 for 95% CI [cite: 106]
            ucb = mu + 1.96 * sigma
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

class GaussianProcessAgent:
    """
    Base Contextual Model using Gaussian Process Regression.
    Learns non-parametric functions f_k(s) ~ GP(m, k).
    Reference: 
    """
    def __init__(self, n_arms=4, n_dims=3, lengthscale=1.0, noise_std=5.0):
        self.n_arms = n_arms
        self.lengthscale = lengthscale
        # The paper uses a noise variance sigma_n^2 in the covariance (Eq 9)
        self.noise_variance = noise_std**2
        
        # Storage for training data: X (Contexts) and y (Rewards)
        self.X = [[] for _ in range(n_arms)]
        self.y = [[] for _ in range(n_arms)]

        # --- INITIALIZATION WITH PSEUDO-OBSERVATIONS ---
        # "The Gaussian Process was initialized by the use of 10 pseudo-observations...
        # ...created from a Normal distribution with N(50, 10)" 
        for arm in range(n_arms):
            for _ in range(10):
                # 1. Create random binary context for the pseudo-observation
                # (The paper doesn't specify context distribution for init, 
                # but random binary covers the space best)
                fake_ctx = np.random.randint(0, 2, size=n_dims)
                
                # 2. Sample fake outcome from N(50, 10)
                fake_reward = np.random.normal(50, 10)
                
                # 3. Store
                self.X[arm].append(fake_ctx)
                self.y[arm].append(fake_reward)

    def kernel(self, x1, x2):
        """
        Squared Exponential Kernel.
        k(x, x') = exp( - ||x - x'||^2 / lambda )
        Reference: Equation 8 [cite: 83]
        """
        # Squared Euclidean distance
        sq_dist = np.sum((x1 - x2)**2)
        return np.exp(-sq_dist / self.lengthscale)

    def predict_with_uncertainty(self, context, arm_idx):
        """
        Calculates the posterior mean and variance for a specific arm.
        Uses the standard GP regression formulas.
        Reference: Equations 10-12 [cite: 91-92]
        """
        X_train = np.array(self.X[arm_idx])
        y_train = np.array(self.y[arm_idx])
        x_new = np.array(context)
        N = len(X_train)
        
        # 1. Construct Covariance Matrix K (N x N)
        # k(x_p, x_q) for all training points
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                val = self.kernel(X_train[i], X_train[j])
                K[i, j] = val
                K[j, i] = val
        
        # Add noise variance to diagonal: K + sigma_n^2 * I (Eq 9)
        K_noise = K + self.noise_variance * np.eye(N)
        
        # 2. Construct K_star vector (Covariance between train and test)
        # k(x_i, x_new)
        K_star = np.array([self.kernel(xi, x_new) for xi in X_train])
        
        # 3. K_star_star (Prior variance of test point)
        # k(x_new, x_new) -> exp(0) = 1.0
        K_star_star = 1.0 
        
        # 4. Calculate Posterior Mean and Variance
        # We use strict matrix operations as defined in Eq 10 & 11
        try:
            # Invert (K + sigma^2 I)
            # Using solve is numerically more stable than inv()
            K_inv_y = np.linalg.solve(K_noise, y_train)
            
            # Mean = K_star^T * (K + sigma^2 I)^-1 * y (Eq 10)
            mean = K_star.dot(K_inv_y)
            
            # Variance = k(x,x) - K_star^T * (K + sigma^2 I)^-1 * K_star (Eq 11)
            K_inv_k_star = np.linalg.solve(K_noise, K_star)
            variance = K_star_star - K_star.dot(K_inv_k_star)
            
            # Clip negative variance due to numerical precision issues
            variance = max(variance, 1e-9)
            
        except np.linalg.LinAlgError:
            # Fallback for singular matrix (unlikely with ridge/noise)
            mean = 50.0
            variance = 1.0

        return mean, np.sqrt(variance)

    def select_arm(self, context):
        """Placeholder for subclasses"""
        return 0

    def update(self, context, arm, reward):
        """Stores the new observation."""
        self.X[arm].append(context)
        self.y[arm].append(reward)


class GPUCBAgent(GaussianProcessAgent):
    """
    Gaussian Process with Upper Confidence Bound sampling.
    Reference: Algorithm 1 [cite: 105]
    """
    def select_arm(self, context):
        ucb_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            
            # UCB = Mean + 1.96 * Std
            # "The trade-off parameter is set to 1.96, marking the 95% confidence interval." [cite: 106]
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
            
            # Sample y* ~ N(mu, sigma^2)
            # "Sample y* ~ M(s)" [cite: 115] - sampling from the posterior predictive.
            sample = np.random.normal(mu, sigma)
            sampled_values.append(sample)
            
        return np.argmax(sampled_values)