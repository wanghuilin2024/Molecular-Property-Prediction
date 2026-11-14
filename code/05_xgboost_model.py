from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. 用户自定义设置
# ============================================================
SEED = 0
DATASET_NAME = "esol"  # esol / lipophilicity
SPLIT_METHOD = "scaffold"        # random / scaffold
FEATURE_TYPE = "fusion"        # morgan / fusion
N_JOBS = -1
OVERWRITE = True

# ============================================================
# 2. 手动填写XGBoost超参数
# ============================================================
XGB_PARAMS = {
    "subsample": 0.5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "n_estimators": 300,
    "min_child_weight": 3,
    "max_depth": 4,
    "learning_rate": 0.01,
    "colsample_bytree": 0.9,
}

# ============================================================
# 3. 文件路径
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../dataset/final" / DATASET_NAME / SPLIT_METHOD).resolve()
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"
OUTPUT_DIR = (BASE_DIR / "../results/final_models/xgboost" / DATASET_NAME / SPLIT_METHOD / FEATURE_TYPE).resolve()
MODEL_PATH = OUTPUT_DIR / "xgb_model.joblib"
METRICS_PATH = OUTPUT_DIR / "xgb_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "xgb_predictions.csv"

# ============================================================
# 4. 检查配置与文件
# ============================================================
if DATASET_NAME not in {"esol", "lipophilicity"}:
    raise ValueError(f"不支持的数据集：{DATASET_NAME}")
if SPLIT_METHOD not in {"random", "scaffold"}:
    raise ValueError(f"不支持的数据划分：{SPLIT_METHOD}")
if FEATURE_TYPE not in {"morgan", "fusion"}:
    raise ValueError(f"不支持的特征类型：{FEATURE_TYPE}")
for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
if len({TRAIN_PATH.resolve(), VAL_PATH.resolve(), TEST_PATH.resolve()}) != 3:
    raise ValueError("Train、Validation、Test不能使用同一个文件")
if not OVERWRITE:
    existing_files = [p for p in [MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH] if p.exists()]
    if existing_files:
        raise FileExistsError(f"结果文件已存在：{existing_files}")

# ============================================================
# 5. 读取数据集
# ============================================================
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)
datasets = {"Train": train_df, "Validation": val_df, "Test": test_df}
required_columns = {"row_id", "smiles", "target"}
for name, df in datasets.items():
    if df.empty:
        raise ValueError(f"{name}数据集为空")
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{name}缺少必要列：{missing}")
    if df["row_id"].isna().any() or df["row_id"].duplicated().any():
        raise ValueError(f"{name}存在缺失或重复row_id")
    if df["smiles"].isna().any():
        raise ValueError(f"{name}存在缺失SMILES")

# ============================================================
# 6. 检查数据集重叠
# ============================================================
split_names = list(datasets.keys())
for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        name_a, name_b = split_names[i], split_names[j]
        df_a, df_b = datasets[name_a], datasets[name_b]
        overlap_ids = set(df_a["row_id"]) & set(df_b["row_id"])
        overlap_smiles = set(df_a["smiles"]) & set(df_b["smiles"])
        if overlap_ids:
            raise ValueError(f"{name_a}与{name_b}存在重复row_id：{len(overlap_ids)}条")
        if overlap_smiles:
            raise ValueError(f"{name_a}与{name_b}存在重复SMILES：{len(overlap_smiles)}条")
print("\n数据集重叠检查通过")

# ============================================================
# 7. 定义输入特征
# ============================================================
MORGAN_COLUMNS = [f"Morgan_{i}" for i in range(2048)]
PHYSCHEM_COLUMNS = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAtoms", "NumHeavyAtoms",
    "MolMR", "PSA", "FormalCharge", "NumRings"
]
feature_columns = MORGAN_COLUMNS if FEATURE_TYPE == "morgan" else MORGAN_COLUMNS + PHYSCHEM_COLUMNS
for name, df in datasets.items():
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name}缺少特征列：{missing[:10]}")

# ============================================================
# 8. 构建模型输入
# ============================================================
def prepare_data(df):
    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("输入特征或目标变量存在NaN/Inf")
    return X, y

X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)
X_test, y_test = prepare_data(test_df)

# ============================================================
# 9. 打印实验信息
# ============================================================
print("\n" + "=" * 70)
print("XGBoost Final Model Training")
print("=" * 70)
print(f"Dataset: {DATASET_NAME}")
print(f"Split: {SPLIT_METHOD}")
print(f"Features: {FEATURE_TYPE}")
print(f"Feature dimension: {X_train.shape[1]}")
print(f"Train samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print(f"Test samples: {len(y_test)}")
print(f"Random seed: {SEED}")
print("\nXGBoost Parameters:")
for key, value in XGB_PARAMS.items():
    print(f"  {key}: {value}")
print(f"\nOutput directory: {OUTPUT_DIR}")

# ============================================================
# 10. 创建并训练XGBoost模型
# ============================================================
model = XGBRegressor(
    **XGB_PARAMS,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=N_JOBS,
    tree_method="hist"
)
print("\n开始训练XGBoost模型...")
start_time = time.perf_counter()
model.fit(X_train, y_train)
training_seconds = time.perf_counter() - start_time
print(f"\n模型训练完成，耗时：{training_seconds:.2f}秒")

# ============================================================
# 11. 模型评价
# ============================================================
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)
    metrics = {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}
    return metrics, predictions

train_metrics, train_predictions = evaluate_model(model, X_train, y_train)
val_metrics, val_predictions = evaluate_model(model, X_val, y_val)
test_metrics, test_predictions = evaluate_model(model, X_test, y_test)
all_metrics = {"Train": train_metrics, "Validation": val_metrics, "Test": test_metrics}

print("\n" + "=" * 70)
print("Model Evaluation Results")
print("=" * 70)
for name, metrics in all_metrics.items():
    print(
        f"{name:<10} | MAE={metrics['MAE']:.4f} | "
        f"MSE={metrics['MSE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
        f"R2={metrics['R2']:.4f}"
    )

# ============================================================
# 12. 保存评价指标
# ============================================================
metrics_rows = []
for split_name, metrics in all_metrics.items():
    row = {
        "model": "XGBoost",
        "dataset": DATASET_NAME,
        "split_method": SPLIT_METHOD,
        "feature_type": FEATURE_TYPE,
        "feature_dimension": len(feature_columns),
        "data_split": split_name,
        "sample_count": len(datasets[split_name]),
        "MAE": metrics["MAE"],
        "MSE": metrics["MSE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "training_seconds": training_seconds,
        "random_state": SEED,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": N_JOBS,
        **XGB_PARAMS
    }
    metrics_rows.append(row)
metrics_df = pd.DataFrame(metrics_rows)

# ============================================================
# 13. 保存预测结果
# ============================================================
def build_predictions_df(df, predictions, split_name):
    result = pd.DataFrame({
        "row_id": df["row_id"].to_numpy(),
        "smiles": df["smiles"].to_numpy(),
        "data_split": split_name,
        "y_true": df["target"].to_numpy(),
        "y_pred": predictions
    })
    result["residual"] = result["y_true"] - result["y_pred"]
    result["absolute_error"] = result["residual"].abs()
    return result

predictions_df = pd.concat([
    build_predictions_df(train_df, train_predictions, "Train"),
    build_predictions_df(val_df, val_predictions, "Validation"),
    build_predictions_df(test_df, test_predictions, "Test")
], ignore_index=True)

# ============================================================
# 14. 保存模型和结果
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
predictions_df.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")


