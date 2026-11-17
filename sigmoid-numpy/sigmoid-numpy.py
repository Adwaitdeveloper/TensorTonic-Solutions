import numpy as np


def sigmoid(x):
    """Compute the sigmoid activation function:

    sigma(x) = 1 / (1 + exp(-x))

    Returns:
        float if x is scalar, or np.ndarray with the same shape as x.
    """
    arr = np.asarray(x, dtype=float)
    result = 1.0 / (1.0 + np.exp(-arr))

    # Return float if input was a scalar (0-D array), otherwise ndarray
    return float(result) if arr.ndim == 0 else result