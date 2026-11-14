
# ============================================================
# 03_data_split.py
# Random Split + Bemis-Murcko Scaffold Split
#
# 数据集：
#   ESOL / FreeSolv / Lipophilicity
#
# 比例：
#   Train : Validation : Test = 80 : 10 : 10
#
# 输入：
#   dataset/processed/*_features.csv
#
# 输出：
#   dataset/final/{dataset}/{split}/train.csv
#   dataset/final/{dataset}/{split}/val.csv
#   dataset/final/{dataset}/{split}/test.csv
# ============================================================

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ============================================================
# 1. 基本参数
# ============================================================

SEED = 0

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DATASETS = [
    "esol",
    "freesolv",
    "lipophilicity"
]

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "../dataset/processed"
OUTPUT_DIR = BASE_DIR / "../dataset/final"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert np.isclose(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO,
    1.0
), "划分比例之和必须等于1"


# ============================================================
# 2. 生成Canonical SMILES
# ============================================================

def get_canonical_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(
            f"无效SMILES：{smiles}"
        )

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )


# ============================================================
# 3. 提取Bemis-Murcko骨架
# ============================================================

def get_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(
            f"无效SMILES：{smiles}"
        )

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
    )

    return scaffold


# ============================================================
# 4. 按分组构建索引
# ============================================================

def build_groups(group_keys):

    groups = defaultdict(list)

    for index, key in enumerate(group_keys):
        groups[key].append(index)

    return list(groups.values())


# ============================================================
# 5. Random Split
# ============================================================

def random_split(df):

    """
    随机划分。

    使用Canonical SMILES进行分组，
    防止同一分子出现在不同数据子集中。
    """

    canonical_smiles = [
        get_canonical_smiles(smiles)
        for smiles in df["smiles"]
    ]

    groups = build_groups(canonical_smiles)

    rng = np.random.default_rng(SEED)

    rng.shuffle(groups)

    n_samples = len(df)

    target_train = int(n_samples * TRAIN_RATIO)
    target_val = int(n_samples * VAL_RATIO)

    train_idx = []
    val_idx = []
    test_idx = []

    for group in groups:

        if len(train_idx) + len(group) <= target_train:

            train_idx.extend(group)

        elif len(val_idx) + len(group) <= target_val:

            val_idx.extend(group)

        else:

            test_idx.extend(group)

    return (
        np.asarray(train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
        np.asarray(test_idx, dtype=int)
    )


# ============================================================
# 6. Scaffold Split
# ============================================================

def scaffold_split(df):

    """
    Bemis-Murcko骨架划分。

    改进策略：
    1. 根据骨架进行分组。
    2. 随机打乱组顺序。
    3. 按组大小降序排列。
    4. 优先处理较大的骨架组。
    5. 将每组分配给使总体比例偏差最小的子集。

    同一骨架不能跨Train、Validation和Test。
    """

    scaffolds = [
        get_scaffold(smiles)
        for smiles in df["smiles"]
    ]

    groups = build_groups(scaffolds)

    rng = np.random.default_rng(SEED)

    # 随机化同样大小的骨架组顺序
    rng.shuffle(groups)

    # 优先处理大骨架组
    groups.sort(
        key=len,
        reverse=True
    )

    n_samples = len(df)

    targets = np.array([
        n_samples * TRAIN_RATIO,
        n_samples * VAL_RATIO,
        n_samples * TEST_RATIO
    ], dtype=float)

    splits = [
        [],
        [],
        []
    ]

    counts = np.zeros(3, dtype=int)

    for group in groups:

        group_size = len(group)

        scores = []

        for split_id in range(3):

            trial_counts = counts.copy()

            trial_counts[split_id] += group_size

            # 与目标样本数的平方偏差
            score = np.sum(
                (trial_counts - targets) ** 2
            )

            scores.append(score)

        best_split = int(
            np.argmin(scores)
        )

        splits[best_split].extend(group)

        counts[best_split] += group_size

    train_idx, val_idx, test_idx = [
        np.asarray(indices, dtype=int)
        for indices in splits
    ]

    # 禁止出现空集合
    if (
        len(train_idx) == 0
        or len(val_idx) == 0
        or len(test_idx) == 0
    ):
        raise ValueError(
            "Scaffold Split出现空数据集。"
            "请检查骨架分组大小及划分策略。"
        )

    return (
        train_idx,
        val_idx,
        test_idx
    )


# ============================================================
# 7. 检查数据划分
# ============================================================

def check_split(
    df,
    train_df,
    val_df,
    test_df,
    split_name
):

    # --------------------------------------------------------
    # 7.1 检查三个子集均非空
    # --------------------------------------------------------

    if min(
        len(train_df),
        len(val_df),
        len(test_df)
    ) == 0:
        raise ValueError(
            f"{split_name}划分存在空数据集"
        )

    # --------------------------------------------------------
    # 7.2 检查样本总数
    # --------------------------------------------------------

    total = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    if total != len(df):
        raise ValueError(
            "划分后样本数量与原始数据不一致"
        )

    # --------------------------------------------------------
    # 7.3 检查row_id是否跨集合重复
    # --------------------------------------------------------

    train_ids = set(train_df["row_id"])
    val_ids = set(val_df["row_id"])
    test_ids = set(test_df["row_id"])

    if not (
        train_ids.isdisjoint(val_ids)
        and train_ids.isdisjoint(test_ids)
        and val_ids.isdisjoint(test_ids)
    ):
        raise ValueError(
            "row_id在不同子集之间存在重复"
        )

    if (
        train_ids | val_ids | test_ids
    ) != set(df["row_id"]):
        raise ValueError(
            "划分后row_id与原始数据不一致"
        )

    # --------------------------------------------------------
    # 7.4 检查Canonical SMILES是否跨集合重复
    # --------------------------------------------------------

    train_smiles = {
        get_canonical_smiles(s)
        for s in train_df["smiles"]
    }

    val_smiles = {
        get_canonical_smiles(s)
        for s in val_df["smiles"]
    }

    test_smiles = {
        get_canonical_smiles(s)
        for s in test_df["smiles"]
    }

    if not (
        train_smiles.isdisjoint(val_smiles)
        and train_smiles.isdisjoint(test_smiles)
        and val_smiles.isdisjoint(test_smiles)
    ):
        raise ValueError(
            "Canonical SMILES跨集合重复"
        )

    # --------------------------------------------------------
    # 7.5 Scaffold Split额外检查
    # --------------------------------------------------------

    if split_name == "scaffold":

        train_scaffolds = {
            get_scaffold(s)
            for s in train_df["smiles"]
        }

        val_scaffolds = {
            get_scaffold(s)
            for s in val_df["smiles"]
        }

        test_scaffolds = {
            get_scaffold(s)
            for s in test_df["smiles"]
        }

        if not (
            train_scaffolds.isdisjoint(val_scaffolds)
            and train_scaffolds.isdisjoint(test_scaffolds)
            and val_scaffolds.isdisjoint(test_scaffolds)
        ):
            raise ValueError(
                "Scaffold跨集合重复"
            )

    print("数据完整性检查：通过")
    print("Canonical SMILES重叠检查：通过")

    if split_name == "scaffold":
        print("Scaffold重叠检查：通过")


# ============================================================
# 8. 保存划分结果
# ============================================================

def save_split(
    df,
    indices,
    dataset_name,
    split_name
):

    train_idx, val_idx, test_idx = indices

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # 保存前检查
    check_split(
        df,
        train_df,
        val_df,
        test_df,
        split_name
    )

    save_dir = (
        OUTPUT_DIR / dataset_name / split_name
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        save_dir / "train.csv",
        index=False,
        encoding="utf-8-sig"
    )

    val_df.to_csv(
        save_dir / "val.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_df.to_csv(
        save_dir / "test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    n_total = len(df)

    print(f"\n划分方式：{split_name}")

    print(
        f"Train: {len(train_df)} "
        f"({len(train_df) / n_total:.2%})"
    )

    print(
        f"Validation: {len(val_df)} "
        f"({len(val_df) / n_total:.2%})"
    )

    print(
        f"Test: {len(test_df)} "
        f"({len(test_df) / n_total:.2%})"
    )

    print(
        f"保存位置：{save_dir.resolve()}"
    )


# ============================================================
# 9. 依次处理三个数据集
# ============================================================

for dataset_name in DATASETS:

    print("\n" + "=" * 65)
    print(f"正在处理数据集：{dataset_name}")
    print("=" * 65)

    input_path = (
        INPUT_DIR / f"{dataset_name}_features.csv"
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"找不到特征文件：{input_path.resolve()}"
        )

    df = pd.read_csv(input_path)

    required_columns = {
        "row_id",
        "smiles",
        "target"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{input_path.name}缺少必要列"
        )

    if df.empty:
        raise ValueError(
            f"{input_path.name}为空"
        )

    if df["row_id"].isna().any():
        raise ValueError(
            "row_id存在缺失值"
        )

    if df["row_id"].duplicated().any():
        raise ValueError(
            "row_id存在重复值"
        )

    if df["smiles"].isna().any():
        raise ValueError(
            "SMILES存在缺失值"
        )

    if df["target"].isna().any():
        raise ValueError(
            "target存在缺失值"
        )

    print(
        f"原始样本数量：{len(df)}"
    )

    # --------------------------------------------------------
    # 9.1 Random Split
    # --------------------------------------------------------

    random_indices = random_split(df)

    save_split(
        df=df,
        indices=random_indices,
        dataset_name=dataset_name,
        split_name="random"
    )

    # --------------------------------------------------------
    # 9.2 Scaffold Split
    # --------------------------------------------------------

    scaffold_indices = scaffold_split(df)

    save_split(
        df=df,
        indices=scaffold_indices,
        dataset_name=dataset_name,
        split_name="scaffold"
    )


print("\n" + "=" * 65)
print("三个数据集的Random/Scaffold划分全部完成！")
print("=" * 65)
