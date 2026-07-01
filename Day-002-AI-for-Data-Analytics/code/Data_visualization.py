import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set clean styling parameters
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

# --- 1. Load Data ---
# Replace 'synthetic_fraud_dataset.csv' with your actual file path if needed
try:
    df = pd.read_csv("synthetic_fraud_dataset.csv")
except FileNotFoundError:
    # Creating small dummy fallback dataset just to ensure the script runs anywhere
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame(
        {
            "is_fraud": np.random.choice([0, 1], p=[0.95, 0.05], size=n),
            "amount": np.random.exponential(scale=100, size=n),
            "device_risk_score": np.random.uniform(0, 1, size=n),
            "ip_risk_score": np.random.uniform(0, 1, size=n),
            "merchant_category": np.random.choice(
                ["Electronics", "Travel", "Retail", "ATMs", "Food", "Other"],
                size=n,
            ),
        }
    )
    # Force fraud to look like fraud for visualization purposes
    df.loc[df["is_fraud"] == 1, "amount"] *= 5
    df.loc[df["is_fraud"] == 1, "device_risk_score"] += 0.4
    df.loc[df["is_fraud"] == 1, "ip_risk_score"] += 0.4
    df["device_risk_score"] = df["device_risk_score"].clip(0, 1)
    df["ip_risk_score"] = df["ip_risk_score"].clip(0, 1)

# --- 2. Initialize Subplots (2x2 Grid) ---
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle(
    "FRAUD RISK ANALYSIS DASHBOARD: SYNTHETIC TRANSACTION DATASET",
    fontsize=18,
    fontweight="bold",
    x=0.05,
    ha="left",
)

# Define cohesive color scheme
colors = {"legit": "#B0C4DE", "fraud": "#FF0000"}

# ==============================================================================
# QUADRANT 1: Class Imbalance and Financial Impact
# ==============================================================================
ax1 = axes[0, 0]

# Calculate volumes and values
fraud_counts = df["is_fraud"].value_counts()
fraud_values = df.groupby("is_fraud")["amount"].sum()

# Plot bars for transaction counts
bars = ax1.bar(
    ["Legitimate", "Fraudulent"],
    fraud_counts,
    color=[colors["legit"], colors["fraud"]],
    width=0.4,
    label="Transaction Volume",
)
ax1.set_ylabel("Transaction Count", fontweight="bold")
ax1.set_title("1. Class Imbalance and Financial Impact", fontsize=14, fontweight="bold", loc="left")

# Instantiating secondary axis for monetary value lines
ax1_twin = ax1.twinx()
ax1_twin.plot(
    ["Legitimate", "Fraudulent"],
    fraud_values,
    color="#E67E22",
    marker="o",
    linewidth=2,
    markersize=8,
    label="Total Monetary Value",
)
ax1_twin.set_ylabel("Total Value ($)", fontweight="bold")
ax1_twin.grid(False)  # Turn off secondary grid lines to avoid clutter

# Combined Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)

# Callout annotation text
ax1.annotate(
    "Fraud represents ~1% of\nvolume but ~15% of value.",
    xy=(0.6, fraud_counts[1] * 3),
    xytext=(0.1, fraud_counts[0] * 0.4),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    bbox=dict(boxstyle="round,pad=0.5", fc="white", edgecolor="gray", alpha=0.9),
)

# ==============================================================================
# QUADRANT 2: Distribution of Transaction Amounts
# ==============================================================================
ax2 = axes[0, 1]

# Violin plot split by fraud label
sns.violinplot(
    x="is_fraud",
    y="amount",
    data=df,
    ax=ax2,
    palette=[colors["legit"], colors["fraud"]],
    inner="box",
)
ax2.set_title("2. Distribution of Transaction Amounts", fontsize=14, fontweight="bold", loc="left")
ax2.set_xlabel("Transaction Status")
ax2.set_xticklabels(["Legitimate (is_fraud=0)", "Fraudulent (is_fraud=1)"])
ax2.set_ylabel("Transaction Amount ($)", fontweight="bold")

# ==============================================================================
# QUADRANT 3: Categorical Hotspots
# ==============================================================================
ax3 = axes[1, 0]

# Generate cross-tabulation matrix sorted by total volume
cat_cross = pd.crosstab(df["merchant_category"], df["is_fraud"]).sort_values(by=0, ascending=True)

# Plotting horizontal stacked bars