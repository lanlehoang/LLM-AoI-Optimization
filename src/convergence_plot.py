import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =======================
# Global configuration
# =======================
FIGURE_SIZE = (7.2, 4.5)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.0)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        # Reduce internal padding
        "axes.xmargin": 0.0,
        "axes.ymargin": 0.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "text.antialiased": True,
        "pdf.fonttype": 42,  # IEEE-safe embedding
        "figure.figsize": FIGURE_SIZE,
        "savefig.pad_inches": 0.02,  # reduce outer whitespace
    }
)

# Palette (defined ONCE)
colors = sns.color_palette("deep", 3)
sns.set_palette(colors)

# =======================
# Data loading
# =======================
df = pd.read_csv("data/dqn_results_20251201.csv")

df["objective"] = df.apply(
    lambda row: row["average_aoi"] / (1 - row["dropped_ratio"]),
    axis=1,
)

# =======================
# Plot 1: Objective
# =======================
fig, ax = plt.subplots()

ax.plot(df.index, df["objective"], color=colors[0])

ax.set_xlabel("Episode")
ax.set_ylabel("Objective Function (s)")

# Kill automatic margins explicitly (important)
ax.margins(x=0.02, y=0.05)

ax.grid(True, alpha=0.5)

plt.tight_layout(pad=0.3)
plt.savefig(
    "figures/base_dqn_objective.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("Plot saved to figures/base_dqn_objective.pdf")

# =======================
# Plot 2: Components (dual axis)
# =======================
fig, ax1 = plt.subplots()

line1 = ax1.plot(
    df.index,
    df["average_aoi"],
    color=colors[0],
    label="Average AoI",
)

ax1.set_xlabel("Episode")
ax1.set_ylabel("Average AoI (s)")
ax1.margins(x=0.02, y=0.05)
ax1.grid(True, alpha=0.5)

ax2 = ax1.twinx()
line2 = ax2.plot(
    df.index,
    df["dropped_ratio"] * 100,
    color=colors[1],
    label="Drop Rate",
)
ax2.margins(x=0.02, y=0.05)
ax2.set_ylabel("Drop Rate (%)")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best", frameon=True)

plt.tight_layout(pad=0.3)
plt.savefig(
    "figures/base_dqn_components.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("Plot saved to figures/base_dqn_components.pdf")
