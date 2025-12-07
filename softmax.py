import numpy as np

def softmax(values, gamma=1.0):
    """
    Transforms values to probabilities using softmax with inverse temperature gamma.
    Equation 2[cite: 52].
    """
    # Subtract max for numerical stability
    exp_values = np.exp(gamma * (values - np.max(values)))
    return exp_values / np.sum(exp_values)