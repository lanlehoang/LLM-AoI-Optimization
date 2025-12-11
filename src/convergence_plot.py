import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set IEEE-style plotting parameters
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["lines.linewidth"] = 2.5
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.linewidth"] = 0.8

# Set seaborn style
sns.set_style("whitegrid")

# Read the data
df = pd.read_csv("data/dqn_results_20251201.csv")

# Calculate objective
df["objective"] = df.apply(lambda row: row["average_aoi"] / (1 - row["dropped_ratio"]), axis=1)

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))

# Plot without markers for smooth convergence line
ax.plot(df.index, df["objective"], color=sns.color_palette("deep")[0], linewidth=2.5)

ax.set_xlabel("Episode")
ax.set_ylabel("Objective Function (s)")
# ax.set_title("DQN Convergence Analysis")
ax.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig("figures/base_dqn_objective.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("Plot saved to figures/base_dqn_objective.pdf")

# Create dual-axis plot for components
fig, ax1 = plt.subplots(figsize=(6, 4))

# Plot average AoI on left axis
color1 = sns.color_palette("deep")[0]
line1 = ax1.plot(df.index, df["average_aoi"], color=color1, linewidth=2.5, label="Average AoI")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Average AoI (s)")
ax1.grid(True, alpha=0.5)

# Create second y-axis for drop rate
ax2 = ax1.twinx()
color2 = sns.color_palette("deep")[1]
line2 = ax2.plot(df.index, df["dropped_ratio"] * 100, color=color2, linewidth=2.5, label="Drop Rate")
ax2.set_ylabel("Drop Rate (%)")

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best", frameon=True)

# ax1.set_title("Objective Components Convergence")

plt.tight_layout()
plt.savefig("figures/base_dqn_components.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("Plot saved to figures/base_dqn_components.pdf")
