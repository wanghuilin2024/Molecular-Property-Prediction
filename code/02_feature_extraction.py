
# ============================================================
# 02_feature_extraction.py
# 提取2048位Morgan指纹和11项理化描述符
# 输出CSV文件
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors
)


# ============================================================
# 1. 文件路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "../dataset/processed"
OUTPUT_DIR = BASE_DIR / "../dataset/processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "esol",
    "freesolv",
    "lipophilicity"
]


# ============================================================
# 2. Morgan指纹参数
# ============================================================

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
    includeChirality=False
)

MORGAN_COLUMNS = [
    f"Morgan_{i}"
    for i in range(MORGAN_BITS)
]


# ============================================================
# 3. 理化描述符名称
# ============================================================

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


# ============================================================
# 4. 理化描述符计算函数
# ============================================================

def calculate_physchem(mol):

    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        mol.GetNumAtoms(),
        mol.GetNumHeavyAtoms(),
        Crippen.MolMR(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Chem.GetFormalCharge(mol),
        rdMolDescriptors.CalcNumRings(mol)
    ]


# ============================================================
# 5. 提取三个数据集的分子特征
# ============================================================

for dataset_name in DATASETS:

    input_path = (
        INPUT_DIR / f"{dataset_name}_cleaned.csv"
    )

    output_path = (
        OUTPUT_DIR / f"{dataset_name}_features.csv"
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"找不到清洗后的数据集：{input_path.resolve()}"
        )

    print("\n" + "=" * 60)
    print(f"正在处理：{dataset_name}")

    # --------------------------------------------------------
    # 5.1 读取清洗后的CSV
    # --------------------------------------------------------

    df = pd.read_csv(input_path)

    required_columns = {
        "row_id",
        "smiles",
        "target"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{input_path.name} 缺少必要列"
        )

    if df.empty:
        raise ValueError(
            f"{input_path.name} 数据集为空"
        )

    if df["row_id"].isna().any() or df["row_id"].duplicated().any():
        raise ValueError("row_id存在缺失或重复")

    if df["target"].isna().any():
        raise ValueError("target存在缺失值")

    targets = pd.to_numeric(df["target"], errors="raise")

    if not np.isfinite(targets.to_numpy(dtype=float)).all():
        raise ValueError("target存在非有限数值")

    # --------------------------------------------------------
    # 5.2 提取Morgan指纹和理化描述符
    # --------------------------------------------------------

    fingerprints = []
    descriptors = []

    for index, smiles in enumerate(df["smiles"]):

        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(
                f"第{index}行SMILES为空"
            )

        mol = Chem.MolFromSmiles(smiles)

        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(
                f"第{index}行SMILES无效：{smiles}"
            )

        # Morgan指纹
        fingerprint = (
            morgan_generator.GetFingerprintAsNumPy(mol)
        )

        # 11项理化描述符
        physchem = calculate_physchem(mol)

        if not np.isfinite(physchem).all():
            raise ValueError(
                f"第{index}行理化描述符存在异常值"
            )

        fingerprints.append(fingerprint)
        descriptors.append(physchem)

    # --------------------------------------------------------
    # 5.3 转换为DataFrame
    # --------------------------------------------------------

    df_morgan = pd.DataFrame(
        np.asarray(fingerprints, dtype=np.uint8),
        columns=MORGAN_COLUMNS
    )

    df_physchem = pd.DataFrame(
        np.asarray(descriptors, dtype=float),
        columns=PHYSCHEM_COLUMNS
    )

    # --------------------------------------------------------
    # 5.4 合并原始信息与分子特征
    # --------------------------------------------------------

    info_columns = [
        col for col in [
            "row_id",
            "original_smiles",
            "smiles",
            "target"
        ]
        if col in df.columns
    ]

    df_info = df[info_columns].reset_index(drop=True)

    df_final = pd.concat(
        [
            df_info,
            df_morgan,
            df_physchem
        ],
        axis=1
    )

    # --------------------------------------------------------
    # 5.5 保存CSV
    # --------------------------------------------------------

    df_final.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 5.6 打印特征提取结果
    # --------------------------------------------------------

    print(f"分子数量：{len(df_final)}")
    print(f"Morgan指纹维度：{MORGAN_BITS}")
    print(f"理化描述符维度：{len(PHYSCHEM_COLUMNS)}")
    print(
        f"总特征维度："
        f"{MORGAN_BITS + len(PHYSCHEM_COLUMNS)}"
    )
    print(f"CSV总列数：{df_final.shape[1]}")
    print(f"文件保存至：{output_path.resolve()}")

print("\n三个数据集的特征提取全部完成！")
