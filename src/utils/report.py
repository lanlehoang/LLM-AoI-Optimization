"""
Utility functions for generating reports and visualizations.
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt


def calc_running_mean_adaptive(nums, window_size=4):
    nums = np.asarray(nums, dtype=float)
    running_mean = np.empty_like(nums)

    for i in range(len(nums)):
        start = max(0, i - window_size + 1)
        running_mean[i] = nums[start : i + 1].mean()

    return running_mean


def plot_single_line(
    data_array: np.ndarray | List[float],
    height: int,
    width: int,
    plot_title: str,
    x_label: str,
    y_label: str,
    y_ticks: np.ndarray | List[float],
    save_path: str,
    grid_step: int = 5,
    is_log_scale: bool = False,
    markersize: int = 4,
):
    plt.figure(figsize=(width, height))
    plt.plot(data_array, marker="o", linestyle="-", color="b", label=plot_title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, markersize=markersize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.xticks(np.arange(0, len(data_array) + 1, step=grid_step))
    plt.yticks(y_ticks)
    # Set y-axis to logarithmic scale if specified
    if is_log_scale:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_multiple_lines(
    data_arrays: List[np.ndarray] | List[List[float]],
    height: int,
    width: int,
    plot_title: str,
    x_label: str,
    y_label: str,
    y_ticks: List[float],
    labels: List[str],
    save_path: str,
    grid_step: int = 1,
    is_log_scale: bool = False,
    markersize: int = 4,
):
    plt.figure(figsize=(width, height))
    for i, data_array in enumerate(data_arrays):
        plt.plot(data_array, marker="o", linestyle="-", label=labels[i], markersize=markersize)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.xticks(np.arange(0, len(data_arrays[0]), step=grid_step))
    plt.yticks(y_ticks)
    # Set y-axis to logarithmic scale if specified
    if is_log_scale:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
