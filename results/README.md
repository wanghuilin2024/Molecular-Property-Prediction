# Molecular Property Prediction Using Machine Learning

## 1. Project Overview

This repository contains the Python implementation and experimental results for the final project of the course **Advanced Topics in Artificial Intelligence Drug Discovery**.

**Project Title:** Comparative Evaluation of Molecular Representations and Machine Learning Models for Molecular Property Prediction

This study investigates how molecular representations, machine learning algorithms, and data splitting strategies influence the prediction of molecular properties.

The experiments focus on three research questions:

1. How do different molecular representations affect predictive performance?
2. How does Random Forest compare with XGBoost when using the same molecular features?
3. How does scaffold-based splitting affect predictive performance compared with random splitting?

## 2. Datasets

Two benchmark datasets from MoleculeNet were used.

| Dataset | Number of Records | Prediction Target |
|---|---:|---|
| ESOL | 1,128 | Aqueous solubility (logS) |
| Lipophilicity | 4,200 | Octanol–water distribution coefficient (logD, pH 7.4) |

Molecular structures were represented using SMILES strings and processed using RDKit.

SMILES were canonicalized while retaining stereochemical information. Invalid molecular structures and records with missing or non-finite target values were excluded.

## 3. Molecular Feature Extraction

Three molecular representations were investigated:

**Morgan Fingerprints**
- Radius: 2 (ECFP4)
- Fingerprint length: 2,048 bits
- Chirality disabled

**Physicochemical Descriptors**

Eleven molecular descriptors were calculated using RDKit:

- Molecular weight (MolWt)
- Octanol–water partition coefficient (MolLogP)
- Hydrogen bond donors (NumHDonors)
- Hydrogen bond acceptors (NumHAcceptors)
- Rotatable bonds (NumRotatableBonds)
- Number of atoms (NumAtoms)
- Number of heavy atoms (NumHeavyAtoms)
- Molar refractivity (MolMR)
- Topological polar surface area (TPSA)
- Formal charge (FormalCharge)
- Number of rings (NumRings)

**Concatenated Features**

Morgan fingerprints and physicochemical descriptors were directly concatenated to form a 2,059-dimensional feature vector.

## 4. Machine Learning Models

Two regression algorithms were evaluated:

- Random Forest (RF)
- Extreme Gradient Boosting (XGBoost)

Five model–feature configurations were implemented:

| Configuration | Algorithm | Molecular Representation |
|---|---|---|
| A | Random Forest | Morgan fingerprints |
| B | Random Forest | Physicochemical descriptors |
| C | Random Forest | Concatenated features |
| D | XGBoost | Morgan fingerprints |
| E | XGBoost | Concatenated features |

## 5. Experimental Design

Two data splitting strategies were evaluated:

- **Random splitting:** Molecular records were assigned to training, validation, and test sets, with identical canonical SMILES grouped together.
- **Scaffold-based splitting:** Molecules were grouped according to their Bemis–Murcko scaffolds to assess prediction performance on structurally distinct molecules.

The target splitting ratio was approximately 80% training, 10% validation, and 10% testing.

A fixed random seed of 0 was used.

The experimental design comprised:

**2 datasets × 2 splitting strategies × 5 model–feature configurations = 20 experimental configurations.**

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

## 6. Repository Structure

```text
Molecular-Property-Prediction/
│
├── code/
│   ├── 01_data_clean.py
│   ├── 02_feature_extraction.py
│   ├── 03_data_split.py
│   ├── 04_hyperparameter_search_rf.py
│   ├── 04_hyperparameter_search_xgb.py
│   ├── 05_rf_model.py
│   ├── 05_xgboost_model.py
│   ├── 06_dataset_visualization.py
│   └── 06_visualization.py
│
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── final/
│
├── results/
│   ├── final_models/
│   ├── hyperparameter_search/
│   └── report_figures/
│
└── README.md
```

**Directory descriptions:**

- `code/`: Python scripts for preprocessing, feature extraction, data splitting, hyperparameter search, model training, evaluation, and visualization.
- `dataset/raw/`: Original benchmark datasets.
- `dataset/processed/`: Cleaned molecular data and extracted features.
- `dataset/final/`: Training, validation, and test datasets.
- `results/final_models/`: Final model outputs and evaluation results.
- `results/hyperparameter_search/`: Hyperparameter search records.
- `results/report_figures/`: Figures used in the final report.

## 7. Installation

The project was developed in Python using PyCharm.

The main Python dependencies include:

```bash
pip install numpy pandas scikit-learn rdkit xgboost matplotlib joblib
```

For reproducibility, the package versions used in the experiments should be retained in the Python environment.

## 8. Code Execution

The scripts are organized according to the experimental workflow.

**Step 1. Data Cleaning**

```bash
python code/01_data_clean.py
```

Clean and standardize the molecular datasets.

**Step 2. Molecular Feature Extraction**

```bash
python code/02_feature_extraction.py
```

Generate Morgan fingerprints, physicochemical descriptors, and concatenated molecular features.

**Step 3. Dataset Splitting**

```bash
python code/03_data_split.py
```

Generate random and scaffold-based training, validation, and test partitions.

**Step 4. Hyperparameter Search**

```bash
python code/04_hyperparameter_search_rf.py
python code/04_hyperparameter_search_xgb.py
```

Conduct random hyperparameter searches for Random Forest and XGBoost.

**Step 5. Final Model Training and Evaluation**

```bash
python code/05_rf_model.py
python code/05_xgboost_model.py
```

Train the final regression models and evaluate predictive performance.

**Step 6. Visualization**

```bash
python code/06_dataset_visualization.py
python code/06_visualization.py
```

Generate dataset distribution plots and model performance visualizations.

**Note:** These commands assume that the scripts resolve file paths relative to the repository root. If the scripts use paths relative to the current working directory, the execution directory must be adjusted accordingly.

## 9. Main Results

XGBoost with concatenated molecular features (Configuration E) achieved the lowest test RMSE across both datasets and both splitting strategies.

| Dataset | Splitting Strategy | Best Configuration | Test RMSE | Test R² |
|---|---|---|---:|---:|
| ESOL | Random | E | 0.6227 | 0.8971 |
| ESOL | Scaffold-based | E | 0.9388 | 0.8168 |
| Lipophilicity | Random | E | 0.6960 | 0.6587 |
| Lipophilicity | Scaffold-based | E | 0.8257 | 0.5526 |

The experiments showed that:

- Concatenated features achieved lower test RMSE than Morgan fingerprints alone in all eight matched comparisons.
- XGBoost achieved lower test RMSE than Random Forest in all eight comparisons using matched molecular representations.
- Scaffold-based splitting produced higher test RMSE than random splitting across all ten corresponding model–feature comparisons.

## 10. Limitations

Hyperparameter optimization combined random search with iterative manual tuning. Although validation MAE was used to identify candidate parameter settings, training, validation, and test performance were inspected during subsequent manual tuning.

Consequently, the reported test results should be interpreted as exploratory performance estimates rather than fully independent assessments of generalization.

The experiments also used fixed data partitions without repeated random seeds. Future work should use independent test data, standardized validation-based hyperparameter selection, and repeated evaluation across multiple partitions.

## 11. References

1. Wu, Z., et al. (2018). MoleculeNet: A benchmark for molecular machine learning. *Chemical Science*, 9, 513–530. https://doi.org/10.1039/C7SC02664A
2. Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742–754. https://doi.org/10.1021/ci100050t
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
5. Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887–2893. https://doi.org/10.1021/jm9602928
