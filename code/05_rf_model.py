from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. 用户自定义设置
# ============================================================
SEED = 0
N_JOBS = -1

# 数据集：esol / freesolv / lipophilicity
DATASET_NAME = "lipophilicity"

# 划分方式：random / scaffold
SPLIT_METHOD = "scaffold"

# 特征类型：morgan / physchem / fusion
FEATURE_TYPE = "fusion"

# 手动指定数据路径
TRAIN_PATH = "../dataset/final/lipophilicity/scaffold/train.csv"
VAL_PATH = "../dataset/final/lipophilicity/scaffold/val.csv"
TEST_PATH = "../dataset/final/lipophilicity/scaffold/test.csv"

# 最终模型保存目录
OUTPUT_ROOT = "../results/final_models/rf"

# ============================================================
# 2. 手动设置Random Forest超参数
# ============================================================
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "max_features": 1.0,
    "bootstrap": True
}

# ============================================================
# 3. 文件路径及参数检查
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

train_path = (BASE_DIR / TRAIN_PATH).resolve()
val_path = (BASE_DIR / VAL_PATH).resolve()
test_path = (BASE_DIR / TEST_PATH).resolve()

for name, path in [
    ("Train", train_path),
    ("Validation", val_path),
    ("Test", test_path)
]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到{name}数据集：{path}")

if len({train_path, val_path, test_path}) != 3:
    raise ValueError("Train、Validation和Test必须是三个不同文件")

if DATASET_NAME not in {"esol", "freesolv", "lipophilicity"}:
    raise ValueError(f"不支持的数据集：{DATASET_NAME}")

if SPLIT_METHOD not in {"random", "scaffold"}:
    raise ValueError(f"不支持的划分方式：{SPLIT_METHOD}")

if FEATURE_TYPE not in {"morgan", "physchem", "fusion"}:
    raise ValueError(f"不支持的特征类型：{FEATURE_TYPE}")

# 检查路径是否与当前实验设置一致
expected_folder = Path("dataset") / "final" / DATASET_NAME / SPLIT_METHOD

for name, path in [
    ("Train", train_path),
    ("Validation", val_path),
    ("Test", test_path)
]:
    if path.parent.parts[-4:] != expected_folder.parts:
        raise ValueError(
            f"{name}路径与DATASET_NAME或SPLIT_METHOD不匹配：{path}"
        )

# 检查手动输入的超参数
valid_params = RandomForestRegressor().get_params()
unknown_params = set(RF_PARAMS) - set(valid_params)

if unknown_params:
    raise ValueError(f"存在不支持的RF超参数：{unknown_params}")

if "random_state" in RF_PARAMS or "n_jobs" in RF_PARAMS:
    raise ValueError(
        "random_state和n_jobs请分别通过SEED和N_JOBS设置"
    )

# 固定输出目录，不再创建时间戳文件夹
OUTPUT_DIR = (
    (BASE_DIR / OUTPUT_ROOT).resolve()
    / DATASET_NAME
    / SPLIT_METHOD
    / FEATURE_TYPE
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "rf_model.joblib"
METRICS_PATH = OUTPUT_DIR / "rf_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "rf_predictions.csv"

# ============================================================
# 4. 读取Train、Validation和Test
# ============================================================
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

datasets = {
    "Train": train_df,
    "Validation": val_df,
    "Test": test_df
}

required_columns = {"row_id", "smiles", "target"}

for name, df in datasets.items():
    if df.empty:
        raise ValueError(f"{name}数据集为空")

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{name}缺少必要列：{missing}")

    if df["row_id"].duplicated().any():
        raise ValueError(f"{name}存在重复row_id")

    if df["smiles"].isna().any():
        raise ValueError(f"{name}存在缺失SMILES")

# 检查不同数据子集是否存在重复分子或row_id
for name_a, name_b in [
    ("Train", "Validation"),
    ("Train", "Test"),
    ("Validation", "Test")
]:
    df_a = datasets[name_a]
    df_b = datasets[name_b]

    smiles_overlap = set(df_a["smiles"]) & set(df_b["smiles"])
    if smiles_overlap:
        raise ValueError(
            f"{name_a}与{name_b}存在重复SMILES："
            f"{list(smiles_overlap)[:5]}"
        )

    id_overlap = set(df_a["row_id"]) & set(df_b["row_id"])
    if id_overlap:
        raise ValueError(f"{name_a}与{name_b}存在重复row_id")

# ============================================================
# 5. 选择输入特征
# ============================================================
# 2048位Morgan指纹
MORGAN_COLUMNS = [f"Morgan_{i}" for i in range(2048)]

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

if FEATURE_TYPE == "morgan":
    feature_columns = MORGAN_COLUMNS
elif FEATURE_TYPE == "physchem":
    feature_columns = PHYSCHEM_COLUMNS
else:
    feature_columns = MORGAN_COLUMNS + PHYSCHEM_COLUMNS

# 检查输入特征是否完整
for name, df in datasets.items():
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name}缺少特征列：{missing[:10]}")

# 提取X和y
X_train = train_df[feature_columns].to_numpy(dtype=np.float64)
y_train = train_df["target"].to_numpy(dtype=np.float64)

X_val = val_df[feature_columns].to_numpy(dtype=np.float64)
y_val = val_df["target"].to_numpy(dtype=np.float64)

X_test = test_df[feature_columns].to_numpy(dtype=np.float64)
y_test = test_df["target"].to_numpy(dtype=np.float64)

# 检查NaN和Inf
for name, X, y in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test)
]:
    if not np.isfinite(X).all():
        raise ValueError(f"{name}特征存在NaN或Inf")
    if not np.isfinite(y).all():
        raise ValueError(f"{name}目标值存在NaN或Inf")

# ============================================================
# 6. 打印实验信息
# ============================================================
print("\n" + "=" * 70)
print("Random Forest Final Model Training")
print("=" * 70)
print(f"Dataset: {DATASET_NAME}")
print(f"Split: {SPLIT_METHOD}")
print(f"Features: {FEATURE_TYPE}")
print(f"Feature dimension: {X_train.shape[1]}")
print(f"Train samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print(f"Test samples: {len(y_test)}")

print("\nManual hyperparameters:")
for key, value in RF_PARAMS.items():
    print(f"  {key}: {value}")

print(f"\nOutput directory: {OUTPUT_DIR}")

# ============================================================
# 7. 仅使用Train训练Random Forest模型
# ============================================================
print("\n开始训练Random Forest模型...")
start_time = time.perf_counter()

model = RandomForestRegressor(
    **RF_PARAMS,
    random_state=SEED,
    n_jobs=N_JOBS
)

model.fit(X_train, y_train)

training_seconds = time.perf_counter() - start_time
print(f"模型训练完成，耗时：{training_seconds:.2f}秒")

# ============================================================
# 8. Train / Validation / Test性能评价
# ============================================================
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    metrics = {
        "MAE": float(mean_absolute_error(y, predictions)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y, predictions))
    }
    return metrics, predictions


results = []
prediction_tables = []

for name, df, X, y in [
    ("Train", train_df, X_train, y_train),
    ("Validation", val_df, X_val, y_val),
    ("Test", test_df, X_test, y_test)
]:
    metrics, predictions = evaluate_model(model, X, y)

    # 保存评价指标，同时记录手动设置的超参数
    results.append({
        "dataset": DATASET_NAME,
        "split_method": SPLIT_METHOD,
        "feature_type": FEATURE_TYPE,
        "model": "RandomForest",
        "data_split": name,
        "n_samples": len(y),
        **metrics,
        "random_state": SEED,
        **RF_PARAMS
    })

    # 保存真实值、预测值、残差和绝对误差
    prediction_df = pd.DataFrame({
        "row_id": df["row_id"].to_numpy(),
        "smiles": df["smiles"].to_numpy(),
        "data_split": name,
        "y_true": y,
        "y_pred": predictions,
        "residual": y - predictions,
        "absolute_error": np.abs(y - predictions)
    })
    prediction_tables.append(prediction_df)

    print(
        f"{name:10s} | "
        f"MAE={metrics['MAE']:.4f} | "
        f"MSE={metrics['MSE']:.4f} | "
        f"RMSE={metrics['RMSE']:.4f} | "
        f"R2={metrics['R2']:.4f}"
    )

# ============================================================
# 9. 保存模型和结果
# ============================================================
joblib.dump(model, MODEL_PATH)

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(
    METRICS_PATH,
    index=False,
    encoding="utf-8-sig"
)

predictions_df = pd.concat(prediction_tables, ignore_index=True)
predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
    encoding="utf-8-sig"
)

