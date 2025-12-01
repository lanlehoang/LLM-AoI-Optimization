import numpy as np


def calc_running_mean_adaptive(nums, window_size):
    nums = np.asarray(nums, dtype=float)
    running_mean = np.empty_like(nums)

    for i in range(len(nums)):
        start = max(0, i - window_size + 1)
        running_mean[i] = nums[start : i + 1].mean()

    return running_mean


def plot_metrics():
    pass
