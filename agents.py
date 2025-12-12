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

    def get_recommendations(self, context, k=1.96):
        results = []
        for arm in range(self.n_arms):
            # For MeanTracking, use self.estimates[arm] instead of 50.0
            # For Random, just use 50.0 or 0.0
            est = 50.0 
            
            results.append({
                "mean": est,
                "sigma": np.nan,
                "lower": np.nan,  # NaN tells the UI "No interval here"
                "upper": np.nan
            })
        return results


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

    def get_recommendations(self, context, k=1.96):
        results = []
        for arm in range(self.n_arms):
            # For MeanTracking, use self.estimates[arm] instead of 50.0
            # For Random, just use 50.0 or 0.0
            est = 50.0 
            
            results.append({
                "mean": est,
                "sigma": np.nan,
                "lower": np.nan,  # NaN tells the UI "No interval here"
                "upper": np.nan
            })
        return results


class LinearRegressionAgent:
    """
    Base Contextual Model using Recursive Least Squares.
    Reference: [cite: 64-71]
    """
    def __init__(self, n_arms=4, n_features=3, pseudo_observations=True):
        self.n_arms = n_arms
        self.n_features = n_features + 1 # +1 for intercept
        self.pseudo_observations = pseudo_observations
        
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
            # TODO: try out different delta values.
            current_A_inv = np.eye(self.n_features) * 1000
            current_b = np.zeros(self.n_features)
            
            # Generate 10 pseudo-observations
            for _ in range(10):
                if not self.pseudo_observations:
                    break
                
                # Random binary context (guess based on GP section which implies similar setup)
                fake_ctx = np.random.choice([-1, 1], size=n_features)
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
    Contextual Model using Gaussian Process Regression.
    Learns non-parametric functions f_k(s) ~ GP(m, k).
    Reference: [cite: 72-98]
    """
    def __init__(self, n_arms=4, n_dims=3, lengthscale=2.0, noise_std=5.0, signal_std=20.0):
        self.n_arms = n_arms
        self.n_dims = n_dims
        
        # Hyperparameters
        # The paper estimates lengthscale via gradient descent.
        # For binary inputs (-1, 1), sq_dist is 0, 4, 8, or 12. 
        # A lengthscale of ~2.0 to 10.0 is reasonable if not optimizing.
        self.lengthscale = lengthscale 
        
        # Variance of the noise epsilon [cite: 86]
        self.noise_variance = noise_std**2
        
        # Variance of the signal (amplitude). 
        # CRITICAL FIX: The payoffs vary by ~15-30.
        # A kernel variance of 1.0 (default in previous code) is too small to model this.
        self.signal_variance = signal_std**2 

        self.X = [[] for _ in range(n_arms)]
        self.y = [[] for _ in range(n_arms)]

        # --- INITIALIZATION WITH PSEUDO-OBSERVATIONS ---
        # The paper initializes with 10 pseudo-observations N(50, 10) [cite: 71, 98]
        for arm in range(n_arms):
            for _ in range(10):
                break
                # Contexts are binary elements on (+) or off (-) [cite: 128]
                # Represented here as -1 and 1 to match regression logic.
                fake_ctx = np.random.choice([-1, 1], size=n_dims)
                
                # Sample fake outcome from N(50, 10)
                fake_reward = np.random.normal(50, 10)
                
                self.X[arm].append(fake_ctx)
                self.y[arm].append(fake_reward)

    def kernel(self, x1, x2):
        """
        Squared Exponential Kernel with Signal Amplitude.
        Reference: Equation 8 and 9 context [cite: 83-86]
        """
        sq_dist = np.sum((x1 - x2)**2)
        # Added signal_variance scaling to allow learning of large deviations (e.g. +/- 15)
        return self.signal_variance * np.exp(-sq_dist / self.lengthscale)

    def predict_with_uncertainty(self, context, arm_idx):
        """
        Calculates posterior mean and variance.
        Reference: Equations 10-12 [cite: 90-93]
        """
        X_train = np.array(self.X[arm_idx])
        y_train = np.array(self.y[arm_idx])
        x_new = np.array(context)
        N = len(X_train)
        
        # CRITICAL FIX: Center the data.
        # GPs assume a zero-mean prior. Data is around 50[cite: 148].
        # We subtract the mean (50) to work in residual space.
        y_centered = y_train - 50.0

        # 1. Construct Covariance Matrix K
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                val = self.kernel(X_train[i], X_train[j])
                K[i, j] = val
                K[j, i] = val
        
        # Add noise variance (Eq 9) [cite: 86]
        K_noise = K + self.noise_variance * np.eye(N)
        
        # 2. K_star (covariance between new point and training points)
        K_star = np.array([self.kernel(xi, x_new) for xi in X_train])
        
        # 3. K_star_star (prior variance of the new point)
        # MUST equal the signal variance (k(x,x)), not 1.0
        K_star_star = self.signal_variance 
        
        # 4. Posterior Mean and Variance
        try:
            # Solve (K + sigma^2 I)^-1 * y
            # We use Cholesky decomposition for stability, or standard solve
            K_inv_y = np.linalg.solve(K_noise, y_centered)
            
            # Mean of residuals
            mean_residual = K_star.dot(K_inv_y)
            
            # Add baseline (50) back to prediction
            mean = mean_residual + 50.0
            
            # Variance calculation: k(x,x) - k(x,X) * (K_noise)^-1 * k(X,x)
            K_inv_k_star = np.linalg.solve(K_noise, K_star)
            variance = K_star_star - K_star.dot(K_inv_k_star)
            
            # Numerical stability
            variance = max(variance, 1e-9)
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            mean = 50.0
            variance = self.signal_variance

        return mean, np.sqrt(variance)

    def select_arm(self, context):
        return 0

    def update(self, context, arm, reward):
        self.X[arm].append(context)
        self.y[arm].append(reward)


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


class GPUCBAgent(GaussianProcessAgent):
    """
    Gaussian Process with UCB (Algorithm 1)
    Reference: Algorithm 1 [cite: 105]
    """
    def select_arm(self, context):
        ucb_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            # UCB with 95% confidence interval (1.96) [cite: 106]
            ucb = mu + 1.96 * sigma
            ucb_values.append(ucb)
        return np.argmax(ucb_values)


class GPThompsonAgent(GaussianProcessAgent):
    """
    Gaussian Process with Thompson Sampling (Algorithm 2)
    Reference: Algorithm 2 [cite: 115]
    """
    def select_arm(self, context):
        sampled_values = []
        for arm in range(self.n_arms):
            mu, sigma = self.predict_with_uncertainty(context, arm)
            # Sample from the posterior distribution [cite: 115]
            sample = np.random.normal(mu, sigma)
            sampled_values.append(sample)
        return np.argmax(sampled_values)