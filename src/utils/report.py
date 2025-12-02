"""
Utility functions for generating reports and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_running_mean_adaptive(nums, window_size):
    nums = np.asarray(nums, dtype=float)
    running_mean = np.empty_like(nums)

    for i in range(len(nums)):
        start = max(0, i - window_size + 1)
        running_mean[i] = nums[start : i + 1].mean()

    return running_mean


def draw_line_plot(data_array, plot_title, x_label, y_label):
    plt.figure(figsize=(10, 5))
    plt.plot(data_array, marker="o", linestyle="-", color="b", label=plot_title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.xticks(np.arange(0, len(data_array), step=5))
    plt.yticks(np.round(np.linspace(data_array.min(), data_array.max(), 10), 3))
    plt.legend()
    plt.tight_layout()
    plt.show()
