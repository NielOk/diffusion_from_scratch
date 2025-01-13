import numpy as np
import os
from typing import Dict
import json

PROJECT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_BASE_DIR)
DATA_DIR = os.path.join(REPO_DIR, "training_data")

### Universal functions###
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

def normalize_pe(pe: np.ndarray # Positional embedding
                 ) -> np.ndarray:
    """
    Normalize positional embeddings to zero mean and unit variance.
    Returns Normalized positional embedding.
    """
    mean = np.mean(pe)
    std = np.std(pe)
    return (pe - mean) / std