from pathlib import Path
from datetime import datetime
import csv
import json
import time

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. 用户自定义设置（主要修改这里）
# ============================================================

SEED = 0

# 每组实验随机搜索次数
N_ITER = 300

# CPU并行
N_JOBS = -1

# 数据集：esol / freesolv / lipophilicity
DATASET_NAME = "lipophilicity"

# 划分：random / scaffold
SPLIT_METHOD = "random"

# 特征：morgan / fusion
FEATURE_TYPE = "morgan"

# 手动指定训练集和验证集
TRAIN_PATH = "../dataset/final/lipophilicity/random/train.csv"
VAL_PATH = "../dataset/final/lipophilicity/random/val.csv"

# 结果保存根目录
OUTPUT_ROOT = "../results/hyperparameter_search/xgboost"


# ============================================================
# 2. 超参数搜索空间
# ============================================================

PARAM_DISTRIBUTIONS = {
    # 决策树数量
    "n_estimators": [
        100, 200, 300, 500, 700
    ],

    # 单棵树最大深度
    "max_depth": [
        3, 4, 5, 6, 8, 10
    ],

    # 学习率
    "learning_rate": [
        0.01, 0.03, 0.05, 0.1, 0.2
    ],

    # 每棵树使用的样本比例
    "subsample": [
        0.6, 0.8, 1.0
    ],

    # 每棵树使用的特征比例
    "colsample_bytree": [
        0.5, 0.7, 0.9, 1.0
    ],

    # L1正则化
    "reg_alpha": [
        0, 0.01, 0.1, 1.0
    ],

    # L2正则化
    "reg_lambda": [
        0.1, 1.0, 5.0, 10.0
    ],

    # 子节点最小Hessian权重
    "min_child_weight": [
        1, 3, 5
    ]
}


# ============================================================
# 3. 文件路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

train_path = (BASE_DIR / TRAIN_PATH).resolve()
val_path = (BASE_DIR / VAL_PATH).resolve()

if not train_path.is_file():
    raise FileNotFoundError(
        f"找不到训练集：{train_path}"
    )

if not val_path.is_file():
    raise FileNotFoundError(
        f"找不到验证集：{val_path}"
    )

if train_path == val_path:
    raise ValueError(
        "训练集和验证集不能是同一个文件"
    )

if FEATURE_TYPE not in {"morgan", "fusion"}:
    raise ValueError(
        f"XGBoost不支持当前特征类型：{FEATURE_TYPE}"
    )

# 不同实验组合分别保存
OUTPUT_DIR = (
    (BASE_DIR / OUTPUT_ROOT).resolve()
    / DATASET_NAME
    / SPLIT_METHOD
    / FEATURE_TYPE
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# 保存文件名
CSV_PATH = (
    OUTPUT_DIR
    / f"xgb_{FEATURE_TYPE}_search_results.csv"
)

TXT_PATH = (
    OUTPUT_DIR
    / f"xgb_{FEATURE_TYPE}_search_log.txt"
)

JSON_PATH = (
    OUTPUT_DIR
    / f"xgb_{FEATURE_TYPE}_best_params.json"
)


# ============================================================
# 4. 读取训练集和验证集
# ============================================================

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

if train_df.empty or val_df.empty:
    raise ValueError(
        "训练集或验证集为空"
    )

required_columns = {
    "row_id", "smiles", "target"
}

for name, df in [
    ("Train", train_df),
    ("Validation", val_df)
]:
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{name}缺少必要列"
        )

    if df["row_id"].duplicated().any():
        raise ValueError(
            f"{name}存在重复row_id"
        )

if not set(train_df["smiles"]).isdisjoint(
    set(val_df["smiles"])
):
    raise ValueError(
        "Train与Validation存在重复SMILES"
    )


# ============================================================
# 5. 选择输入特征
# ============================================================

# 2048位Morgan指纹
MORGAN_COLUMNS = [
    f"Morgan_{i}"
    for i in range(2048)
]

# 11项RDKit理化描述符
PHYSCHEM_COLUMNS = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAtoms",
    "NumHeavyAtoms",
    "MolMR",
    "PSA",
    "FormalCharge",
    "NumRings"
]

# 方案D：Morgan
if FEATURE_TYPE == "morgan":
    feature_columns = MORGAN_COLUMNS

# 方案E：Morgan + PhysChem
elif FEATURE_TYPE == "fusion":
    feature_columns = (
        MORGAN_COLUMNS + PHYSCHEM_COLUMNS
    )

# 检查特征是否存在
for name, df in [
    ("Train", train_df),
    ("Validation", val_df)
]:
    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}缺少特征列：{missing[:10]}"
        )


# 构建模型输入
X_train = train_df[feature_columns].to_numpy(
    dtype=np.float32
)

y_train = train_df["target"].to_numpy(
    dtype=np.float64
)

X_val = val_df[feature_columns].to_numpy(
    dtype=np.float32
)

y_val = val_df["target"].to_numpy(
    dtype=np.float64
)

# 检查缺失值和无穷值
if not (
    np.isfinite(X_train).all()
    and np.isfinite(X_val).all()
    and np.isfinite(y_train).all()
    and np.isfinite(y_val).all()
):
    raise ValueError(
        "特征或标签中存在NaN/Inf"
    )


# ============================================================
# 6. 评价指标
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    return {
        "MAE": float(
            mean_absolute_error(y, predictions)
        ),
        "MSE": float(
            mean_squared_error(y, predictions)
        ),
        "R2": float(
            r2_score(y, predictions)
        )
    }


# ============================================================
# 7. 生成随机超参数组合
# ============================================================

sampler = list(
    ParameterSampler(
        PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        random_state=SEED
    )
)

PARAM_NAMES = list(
    PARAM_DISTRIBUTIONS.keys()
)

FIELDNAMES = [
    "trial",
    *PARAM_NAMES,
    "train_MAE",
    "train_MSE",
    "train_R2",
    "val_MAE",
    "val_MSE",
    "val_R2",
    "fit_seconds"
]


# ============================================================
# 8. 打印实验信息
# ============================================================

print("\n" + "=" * 70)
print("XGBoost Hyperparameter Search")
print("=" * 70)

print(f"Dataset: {DATASET_NAME}")
print(f"Split: {SPLIT_METHOD}")
print(f"Features: {FEATURE_TYPE}")
print(f"Feature dimension: {X_train.shape[1]}")
print(f"Train samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print(f"Search trials: {len(sampler)}")
print(f"Train path: {train_path}")
print(f"Validation path: {val_path}")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# 9. 随机超参数搜索
# ============================================================

best_mae = float("inf")
best_params = None
best_trial = None

# 每次运行重新创建CSV和TXT
with (
    CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file,
    TXT_PATH.open(
        "w",
        encoding="utf-8"
    ) as txt_file
):

    writer = csv.DictWriter(
        csv_file,
        fieldnames=FIELDNAMES
    )

    writer.writeheader()
    csv_file.flush()

    txt_file.write(
        "XGBoost Hyperparameter Search\n"
        f"Dataset: {DATASET_NAME}\n"
        f"Split: {SPLIT_METHOD}\n"
        f"Features: {FEATURE_TYPE}\n"
        f"Train: {train_path}\n"
        f"Validation: {val_path}\n"
        f"Trials: {len(sampler)}\n"
        f"Start: {datetime.now().isoformat()}\n\n"
    )
    txt_file.flush()

    for trial, params in enumerate(
        sampler,
        start=1
    ):

        start_time = time.perf_counter()

        # 创建XGBoost模型
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=N_JOBS,
            tree_method="hist"
        )

        # 仅使用训练集拟合
        model.fit(
            X_train,
            y_train
        )

        elapsed = (
            time.perf_counter() - start_time
        )

        # 训练集评价
        train_metrics = evaluate_model(
            model,
            X_train,
            y_train
        )

        # 验证集评价
        val_metrics = evaluate_model(
            model,
            X_val,
            y_val
        )

        # ----------------------------------------------------
        # 保存当前搜索结果
        # ----------------------------------------------------

        row = {
            "trial": trial,
            **params,
            "train_MAE": train_metrics["MAE"],
            "train_MSE": train_metrics["MSE"],
            "train_R2": train_metrics["R2"],
            "val_MAE": val_metrics["MAE"],
            "val_MSE": val_metrics["MSE"],
            "val_R2": val_metrics["R2"],
            "fit_seconds": elapsed
        }

        writer.writerow(row)
        csv_file.flush()

        # 保存TXT日志
        log_text = (
            f"Trial {trial}/{len(sampler)}\n"
            f"Parameters: {params}\n"
            f"Train: {train_metrics}\n"
            f"Validation: {val_metrics}\n"
            f"Fit time: {elapsed:.2f}s\n"
            + "-" * 60 + "\n"
        )

        txt_file.write(log_text)
        txt_file.flush()

        print(
            f"[{trial}/{len(sampler)}] "
            f"Train MAE={train_metrics['MAE']:.4f} | "
            f"Val MAE={val_metrics['MAE']:.4f} | "
            f"Val R2={val_metrics['R2']:.4f} | "
            f"{elapsed:.2f}s"
        )

        # ----------------------------------------------------
        # 更新最佳超参数
        # ----------------------------------------------------

        if val_metrics["MAE"] < best_mae:

            best_mae = val_metrics["MAE"]
            best_params = params.copy()
            best_trial = trial

            best_record = {
                "model": "XGBRegressor",
                "dataset": DATASET_NAME,
                "split": SPLIT_METHOD,
                "feature_type": FEATURE_TYPE,
                "train_path": str(train_path),
                "val_path": str(val_path),
                "best_trial": best_trial,
                "selection_metric": "val_MAE",
                "best_val_MAE": best_mae,
                "best_params": best_params,
                "random_state": SEED,
                "n_jobs": N_JOBS,
                "objective": "reg:squarederror",
                "tree_method": "hist"
            }

            with JSON_PATH.open(
                "w",
                encoding="utf-8"
            ) as json_file:

                json.dump(
                    best_record,
                    json_file,
                    indent=4,
                    ensure_ascii=False
                )

            print(
                f"  >>> New best! "
                f"Validation MAE={best_mae:.6f}"
            )

    # 写入搜索总结
    txt_file.write(
        "\n" + "=" * 60 + "\n"
        f"Best trial: {best_trial}\n"
        f"Best validation MAE: {best_mae:.6f}\n"
        f"Best parameters: {best_params}\n"
    )
    txt_file.flush()


# ============================================================
# 10. 输出最终结果
# ============================================================

print("\n" + "=" * 70)
print("XGBoost随机超参数搜索完成")
print("=" * 70)

print(f"Best trial: {best_trial}")
print(f"Best Validation MAE: {best_mae:.6f}")
print(f"Best parameters: {best_params}")

print(f"\nCSV: {CSV_PATH}")
print(f"TXT: {TXT_PATH}")
print(f"JSON: {JSON_PATH}")
