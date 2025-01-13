import numpy as np


def positional_encoding(t: int, # Current time step (scalar or array) 
                        d: int # Embedding dimension
                        ) -> np.ndarray:
    """
    Compute sinusoidal positional encoding for time step t. Returns
    positional encoding vector of shape (d,)
    """
    positions = np.arange(d // 2)
    angle_rates = 1 / np.power(10000, (2 * positions) / d)
    pe = np.zeros(d)
    pe[0::2] = np.sin(t * angle_rates)  # Even indices
    pe[1::2] = np.cos(t * angle_rates)  # Odd indices
    return pe

# Example usage
time_step = 1
embedding_dim = 128

# Compute time embedding
time_embedding = positional_encoding(time_step, embedding_dim)
print(time_embedding)
print(time_embedding.shape)