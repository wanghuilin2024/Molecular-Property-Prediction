
# ============================================================
# Generate Figures for Molecular Property Prediction Report
# Datasets: ESOL and Lipophilicity
# Models: Random Forest and XGBoost
# Outputs: Figure 5.1, Figure 5.2, Figure 5.3
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Output directory and figure settings
# ============================================================

OUTPUT_DIR = Path("../results/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

# ============================================================
# 2. Experimental results from Table 5.1
# ============================================================

records = [
    # ESOL - Random
    ("ESOL", "Random", "A", "RF-Morgan", 0.9988, 1.5221, 1.2338, 0.5959),
    ("ESOL", "Random", "B", "RF-PhysChem", 0.5247, 0.5967, 0.7725, 0.8416),
    ("ESOL", "Random", "C", "RF-Fusion", 0.4905, 0.5202, 0.7213, 0.8619),
    ("ESOL", "Random", "D", "XGB-Morgan", 0.9166, 1.3719, 1.1713, 0.6358),
    ("ESOL", "Random", "E", "XGB-Fusion", 0.4570, 0.3877, 0.6227, 0.8971),

    # ESOL - Scaffold
    ("ESOL", "Scaffold", "A", "RF-Morgan", 1.3361, 2.9548, 1.7190, 0.3859),
    ("ESOL", "Scaffold", "B", "RF-PhysChem", 0.7178, 0.9599, 0.9797, 0.8005),
    ("ESOL", "Scaffold", "C", "RF-Fusion", 0.7045, 0.9366, 0.9678, 0.8053),
    ("ESOL", "Scaffold", "D", "XGB-Morgan", 1.1596, 2.3364, 1.5285, 0.5145),
    ("ESOL", "Scaffold", "E", "XGB-Fusion", 0.6911, 0.8813, 0.9388, 0.8168),

    # Lipophilicity - Random
    ("Lipophilicity", "Random", "A", "RF-Morgan", 0.6448, 0.6822, 0.8260, 0.5194),
    ("Lipophilicity", "Random", "B", "RF-PhysChem", 0.6989, 0.8175, 0.9041, 0.4241),
    ("Lipophilicity", "Random", "C", "RF-Fusion", 0.5521, 0.5225, 0.7229, 0.6319),
    ("Lipophilicity", "Random", "D", "XGB-Morgan", 0.5824, 0.5940, 0.7707, 0.5816),
    ("Lipophilicity", "Random", "E", "XGB-Fusion", 0.5170, 0.4844, 0.6960, 0.6587),

    # Lipophilicity - Scaffold
    ("Lipophilicity", "Scaffold", "A", "RF-Morgan", 0.8324, 1.0667, 1.0328, 0.3001),
    ("Lipophilicity", "Scaffold", "B", "RF-PhysChem", 0.7700, 0.9702, 0.9850, 0.3634),
    ("Lipophilicity", "Scaffold", "C", "RF-Fusion", 0.6878, 0.7627, 0.8733, 0.4996),
    ("Lipophilicity", "Scaffold", "D", "XGB-Morgan", 0.7651, 0.9368, 0.9679, 0.3854),
    ("Lipophilicity", "Scaffold", "E", "XGB-Fusion", 0.6368, 0.6819, 0.8257, 0.5526),
]

columns = [
    "Dataset", "Split", "Config", "Model",
    "MAE", "MSE", "RMSE", "R2"
]

df = pd.DataFrame(records, columns=columns)

# Verify the expected experimental design
assert len(df) == 20
assert not df.duplicated(["Dataset", "Split", "Config"]).any()

# Save source data
df.to_csv(
    OUTPUT_DIR / "table_5_1_data.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 3. Color scheme
# ============================================================

COLORS = {
    "A": "#8CA8C8",
    "B": "#91B8A4",
    "C": "#367E65",
    "D": "#D9A46C",
    "E": "#C65D57",
}

RANDOM_COLOR = "#5B86B1"
SCAFFOLD_COLOR = "#D58C62"

CONFIGS = ["A", "B", "C", "D", "E"]

MODEL_LABELS = {
    "A": "RF-Morgan",
    "B": "RF-PhysChem",
    "C": "RF-Fusion",
    "D": "XGB-Morgan",
    "E": "XGB-Fusion",
}

DATASETS = ["ESOL", "Lipophilicity"]
SPLITS = ["Random", "Scaffold"]


def get_rmse(dataset, split, config):
    """Return the test RMSE for a specified configuration."""
    result = df.loc[
        (df["Dataset"] == dataset)
        & (df["Split"] == split)
        & (df["Config"] == config),
        "RMSE"
    ]

    if len(result) != 1:
        raise ValueError(
            f"Invalid selection: {dataset}, {split}, {config}"
        )

    return float(result.iloc[0])


def save_figure(fig, filename):
    """Save both PNG and vector PDF."""
    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        OUTPUT_DIR / f"{filename}.pdf",
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)

    print(f"Saved: {filename}.png / .pdf")


# ============================================================
# 4. Figure 5.1
# Overall predictive performance across 20 configurations
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5.2),
    sharey=False
)

x = np.arange(len(CONFIGS))
bar_width = 0.34

for ax, dataset in zip(axes, DATASETS):

    random_values = [
        get_rmse(dataset, "Random", config)
        for config in CONFIGS
    ]

    scaffold_values = [
        get_rmse(dataset, "Scaffold", config)
        for config in CONFIGS
    ]

    bars_random = ax.bar(
        x - bar_width / 2,
        random_values,
        width=bar_width,
        color=RANDOM_COLOR,
        label="Random split"
    )

    bars_scaffold = ax.bar(
        x + bar_width / 2,
        scaffold_values,
        width=bar_width,
        color=SCAFFOLD_COLOR,
        label="Scaffold split"
    )

    ax.set_title(dataset, fontweight="bold", pad=12)
    ax.set_ylabel("Test RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS)
    ax.set_xlabel("Model-feature configuration")
    ax.set_ylim(
        0,
        max(random_values + scaffold_values) * 1.18
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for bars in [bars_random, bars_scaffold]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    height
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0
            )

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False
)

fig.text(
    0.5,
    0.01,
    "A: RF-Morgan   B: RF-PhysChem   C: RF-Fusion   "
    "D: XGB-Morgan   E: XGB-Fusion",
    ha="center",
    fontsize=9
)

fig.tight_layout(rect=[0, 0.06, 1, 0.92])

save_figure(
    fig,
    "Figure_5_1_Overall_RMSE"
)


# ============================================================
# 5. Figure 5.2
# Effect of molecular feature fusion (RQ1)
# Compare Morgan fingerprints and concatenated features
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12.5, 5),
    sharey=False
)

conditions = [
    ("ESOL", "Random"),
    ("ESOL", "Scaffold"),
    ("Lipophilicity", "Random"),
    ("Lipophilicity", "Scaffold")
]

x = np.arange(len(conditions))
bar_width = 0.34

comparisons = [
    ("Random Forest", "A", "C"),
    ("XGBoost", "D", "E")
]

for ax, (algorithm, baseline_config, fusion_config) in zip(
    axes, comparisons
):

    baseline_values = [
        get_rmse(dataset, split, baseline_config)
        for dataset, split in conditions
    ]

    fusion_values = [
        get_rmse(dataset, split, fusion_config)
        for dataset, split in conditions
    ]

    bars_baseline = ax.bar(
        x - bar_width / 2,
        baseline_values,
        bar_width,
        color="#8CA8C8",
        label="Morgan fingerprints"
    )

    bars_fusion = ax.bar(
        x + bar_width / 2,
        fusion_values,
        bar_width,
        color="#367E65",
        label="Concatenated features"
    )

    ax.set_title(algorithm, fontweight="bold", pad=12)
    ax.set_ylabel("Test RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "ESOL\nRandom",
        "ESOL\nScaffold",
        "Lipo\nRandom",
        "Lipo\nScaffold"
    ])
    ax.set_ylim(
        0,
        max(baseline_values + fusion_values) * 1.22
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for i, (baseline, fusion) in enumerate(
        zip(baseline_values, fusion_values)
    ):
        reduction = (baseline - fusion) / baseline * 100

        ax.annotate(
            f"-{reduction:.1f}%",
            xy=(x[i] + bar_width / 2, fusion),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#245C48"
        )

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.92])

save_figure(
    fig,
    "Figure_5_2_Feature_Fusion"
)


# ============================================================
# 6. Figure 5.3
# Effect of random vs scaffold-based splitting (RQ3)
# Show RMSE for all five configurations
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12.5, 5),
    sharey=False
)

for ax, dataset in zip(axes, DATASETS):

    random_values = np.array([
        get_rmse(dataset, "Random", config)
        for config in CONFIGS
    ])

    scaffold_values = np.array([
        get_rmse(dataset, "Scaffold", config)
        for config in CONFIGS
    ])

    for i, config in enumerate(CONFIGS):

        ax.plot(
            [0, 1],
            [random_values[i], scaffold_values[i]],
            marker="o",
            markersize=8,
            linewidth=2,
            color=COLORS[config],
            label=MODEL_LABELS[config]
        )

        ax.annotate(
            config,
            (1, scaffold_values[i]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=COLORS[config],
            fontweight="bold"
        )

    ax.set_title(dataset, fontweight="bold", pad=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "Random split",
        "Scaffold split"
    ])

    ax.set_xlim(-0.15, 1.25)
    ax.set_ylabel("Test RMSE")

    all_values = np.concatenate([
        random_values,
        scaffold_values
    ])

    margin = (all_values.max() - all_values.min()) * 0.2

    ax.set_ylim(
        max(0, all_values.min() - margin),
        all_values.max() + margin
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=5,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.90])

save_figure(
    fig,
    "Figure_5_3_Splitting_Strategy"
)


# ============================================================
# 7. Summary
# ============================================================

print("\nAll figures generated successfully.")
print("Output directory:", OUTPUT_DIR.resolve())

print("\nGenerated figures:")
print("1. Figure_5_1_Overall_RMSE")
print("2. Figure_5_2_Feature_Fusion")
print("3. Figure_5_3_Splitting_Strategy")
