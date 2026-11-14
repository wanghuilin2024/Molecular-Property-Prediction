
# ============================================================
# 01_data_clean.py
# ESOL、FreeSolv、Lipophilicity 数据清洗
# ============================================================

from pathlib import Path
from collections import Counter, defaultdict
import csv
import math

from rdkit import Chem


# ============================================================
# 1. 设置数据文件路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "ESOL": (
        BASE_DIR / "../dataset/raw/delaney-processed.csv",
        "measured log solubility in mols per litre"
    ),

    "FreeSolv": (
        BASE_DIR / "../dataset/raw/SAMPL.csv",
        "expt"
    ),

    "Lipophilicity": (
        BASE_DIR / "../dataset/raw/Lipophilicity.csv",
        "exp"
    ),
}

OUTPUT_DIR = BASE_DIR / "../dataset/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_COLUMNS = [
    "row_id",
    "original_smiles",
    "smiles",
    "target"
]


# ============================================================
# 2. 定义数据清洗函数
# ============================================================

def clean_dataset(source, destination, target_column):

    cleaned_rows = []
    issues = Counter()
    molecules = defaultdict(list)
    seen_ids = set()
    total_rows = 0

    # --------------------------------------------------------
    # 2.1 读取原始CSV
    # --------------------------------------------------------

    with source.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as input_file:

        reader = csv.DictReader(input_file)

        source_columns = reader.fieldnames or []

        required_columns = {"smiles", target_column}

        missing_columns = required_columns - set(source_columns)

        if missing_columns:
            raise ValueError(
                f"{source.name} 缺少列："
                f"{sorted(missing_columns)}"
            )

        if len(source_columns) != len(set(source_columns)):
            raise ValueError(
                f"{source.name} 存在重复列名"
            )

        has_row_ids = "row_id" in source_columns

        # ----------------------------------------------------
        # 2.2 逐行清洗数据
        # ----------------------------------------------------

        for record_index, row in enumerate(reader):

            total_rows += 1

            # 保留原始行编号
            row_id = record_index

            if has_row_ids:

                try:
                    row_id = int(row["row_id"] or "")

                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{source.name} 第 {reader.line_num} 行："
                        "row_id 必须为整数"
                    ) from exc

                if row_id < 0 or row_id in seen_ids:
                    raise ValueError(
                        f"{source.name} 存在负数或重复 row_id："
                        f"{row_id}"
                    )

                seen_ids.add(row_id)

            reasons = []

            # ------------------------------------------------
            # 2.3 检查SMILES
            # ------------------------------------------------

            original_smiles = row["smiles"]

            smiles = (original_smiles or "").strip()

            mol = None

            if not smiles:
                reasons.append("blank_smiles")

            else:
                mol = Chem.MolFromSmiles(smiles)

                if mol is None or mol.GetNumAtoms() == 0:
                    reasons.append("invalid_smiles")

            # ------------------------------------------------
            # 2.4 检查目标变量
            # ------------------------------------------------

            target_text = (
                row[target_column] or ""
            ).strip()

            target = None

            if not target_text:
                reasons.append("missing_target")

            else:

                try:
                    target = float(target_text)

                except ValueError:
                    reasons.append("non_numeric_target")

                else:
                    if not math.isfinite(target):
                        reasons.append("non_finite_target")

            # ------------------------------------------------
            # 2.5 跳过无效记录
            # ------------------------------------------------

            if reasons:
                issues.update(reasons)
                continue

            # ------------------------------------------------
            # 2.6 生成Canonical SMILES
            # ------------------------------------------------

            canonical_smiles = Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True
            )

            cleaned_rows.append([
                row_id,
                original_smiles,
                canonical_smiles,
                target
            ])

            # 记录重复分子
            molecules[canonical_smiles].append(target)

    # ========================================================
    # 3. 统计重复分子
    # ========================================================

    duplicate_groups = {
        smiles: targets
        for smiles, targets in molecules.items()
        if len(targets) > 1
    }

    differing_target_groups = sum(
        len(set(targets)) > 1
        for targets in duplicate_groups.values()
    )

    # ========================================================
    # 4. 保存清洗后的CSV
    # ========================================================

    with destination.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as output_file:

        writer = csv.writer(output_file)

        writer.writerow(CLEANED_COLUMNS)

        writer.writerows(cleaned_rows)

    # ========================================================
    # 5. 打印数据清洗结果
    # ========================================================

    print("\n" + "=" * 55)
    print(f"数据集：{destination.stem}")

    print(f"原始样本数：{total_rows}")
    print(f"清洗后样本数：{len(cleaned_rows)}")
    print(f"删除样本数：{total_rows - len(cleaned_rows)}")

    print(f"重复分子组数：{len(duplicate_groups)}")
    print(f"标签不同的重复分子组数：{differing_target_groups}")

    print("\n删除原因统计：")

    for reason in [
        "blank_smiles",
        "invalid_smiles",
        "missing_target",
        "non_numeric_target",
        "non_finite_target"
    ]:
        print(f"  {reason}: {issues[reason]}")

    print(f"\nCSV保存至：{destination.resolve()}")
    print("=" * 55)


# ============================================================
# 6. 直接运行数据清洗
# ============================================================

print("\n开始清洗三个数据集...")

for dataset_name, (input_path, target_column) in DATASETS.items():

    input_path = input_path.resolve()

    output_path = OUTPUT_DIR / f"{dataset_name.lower()}_cleaned.csv"

    if not input_path.is_file():
        raise FileNotFoundError(
            f"找不到数据文件：{input_path}"
        )

    clean_dataset(
        source=input_path,
        destination=output_path,
        target_column=target_column
    )

print("\n全部数据集清洗完成！")
print(f"结果保存目录：{OUTPUT_DIR.resolve()}")
