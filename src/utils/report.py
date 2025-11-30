import numpy as np


def calc_running_mean(window_size, data):
    """Calculate the running mean of a 1D numpy array.

    Args:
        window_size (int): The size of the moving window.
        data (np.ndarray): The input 1D array.

    Returns:
        np.ndarray: The running mean array.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    if len(data) < window_size:
        raise ValueError("Data length must be at least as large as the window size.")

    cumsum = np.cumsum(np.insert(data, 0, 0))
    running_mean = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return running_mean


def plot_metrics():
    pass
