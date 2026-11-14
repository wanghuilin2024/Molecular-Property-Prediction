
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. File paths
# ============================================================

from pathlib import Path

# 自动定位项目根目录 assignment3
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 原始数据集目录
DATASET_DIR = PROJECT_DIR / "dataset" / "raw"

ESOL_PATH = DATASET_DIR / "delaney-processed.csv"
LIPO_PATH = DATASET_DIR / "Lipophilicity.csv"

# 图片保存目录
OUTPUT_DIR = PROJECT_DIR / "results" / "report_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MoleculeNet原始数据集目标列
ESOL_TARGET = "measured log solubility in mols per litre"
LIPO_TARGET = "exp"


# ============================================================
# 2. Read datasets
# ============================================================

def load_target(file_path, target_column):

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path.resolve()}"
        )

    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Column '{target_column}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    target = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    target = target.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    return target.to_numpy(dtype=float)


esol_y = load_target(ESOL_PATH, ESOL_TARGET)
lipo_y = load_target(LIPO_PATH, LIPO_TARGET)

print("ESOL samples:", len(esol_y))
print("Lipophilicity samples:", len(lipo_y))


# ============================================================
# 3. Descriptive statistics
# ============================================================

def print_statistics(name, values):

    print(f"\n{name}")
    print("-" * 40)
    print(f"Samples: {len(values)}")
    print(f"Mean:    {np.mean(values):.4f}")
    print(f"Std:     {np.std(values):.4f}")
    print(f"Min:     {np.min(values):.4f}")
    print(f"Median:  {np.median(values):.4f}")
    print(f"Max:     {np.max(values):.4f}")


print_statistics("ESOL", esol_y)
print_statistics("Lipophilicity", lipo_y)


# ============================================================
# 4. Plot settings
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ESOL_COLOR = "#5B86B1"
LIPO_COLOR = "#D58C62"

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4.8)
)


# ============================================================
# 5. Histogram function
# ============================================================

def plot_distribution(
    ax,
    values,
    dataset_name,
    target_name,
    color
):

    # Percentages sum to 100%
    weights = np.ones(len(values)) * 100 / len(values)

    ax.hist(
        values,
        bins=25,
        weights=weights,
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7
    )

    # Mean and median reference lines
    mean_value = np.mean(values)
    median_value = np.median(values)

    ax.axvline(
        mean_value,
        color="#333333",
        linestyle="--",
        linewidth=1.6,
        label=f"Mean = {mean_value:.2f}"
    )

    ax.axvline(
        median_value,
        color="#666666",
        linestyle=":",
        linewidth=1.8,
        label=f"Median = {median_value:.2f}"
    )

    ax.set_title(
        f"{dataset_name} (n = {len(values):,})",
        fontweight="bold",
        pad=12
    )

    ax.set_xlabel(target_name)
    ax.set_ylabel("Percentage of molecules (%)")

    ax.yaxis.grid(
        True,
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9
    )


# ============================================================
# 6. ESOL
# ============================================================

plot_distribution(
    axes[0],
    esol_y,
    dataset_name="ESOL",
    target_name="Aqueous solubility (logS)",
    color=ESOL_COLOR
)


# ============================================================
# 7. Lipophilicity
# ============================================================

plot_distribution(
    axes[1],
    lipo_y,
    dataset_name="Lipophilicity",
    target_name="Distribution coefficient (logD)",
    color=LIPO_COLOR
)


# ============================================================
# 8. Save figures
# ============================================================

fig.tight_layout(w_pad=3)

png_path = OUTPUT_DIR / "Figure_4_1_Target_Distributions.png"
pdf_path = OUTPUT_DIR / "Figure_4_1_Target_Distributions.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("\nFigures saved:")
print(png_path.resolve())
print(pdf_path.resolve())
