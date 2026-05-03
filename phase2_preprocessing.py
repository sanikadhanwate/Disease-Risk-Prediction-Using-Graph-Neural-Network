import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys

HD_PATH   = "heart_disease_uci.csv"
PIMA_PATH = "diabetes.csv"

def section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")

def check_file(path):
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        sys.exit(1)

check_file(HD_PATH)
check_file(PIMA_PATH)

#  STEP 1: Load
section("Step 1 — Loading raw data")

hd   = pd.read_csv(HD_PATH)
pima = pd.read_csv(PIMA_PATH)

print(f"UCI loaded   : {hd.shape}")
print(f"Pima loaded  : {pima.shape}")


#  STEP 2: Clean
section("Step 2 — Cleaning UCI Heart Disease")

hd_clean = hd.copy()

# 2a. Binary label: 0 = no disease, 1 = disease (num 1–4)

hd_clean['label'] = (hd_clean['num'] > 0).astype(int)
print(f"Label binarised: 0={( hd_clean['label']==0).sum()}  1={(hd_clean['label']==1).sum()}")

# 2b. Drop columns: id, dataset (metadata), num (replaced by label),
#     ca (66% missing), thal (53% missing), slope (34% missing)
#     These are too sparse to reliably represent as node features.

drop_cols = ['id', 'dataset', 'num', 'ca', 'thal', 'slope']
hd_clean.drop(columns=drop_cols, inplace=True)
print(f"Dropped columns: {drop_cols}")

# 2c. Encode boolean columns (fbs, exang stored as True/False/nan strings)

for col in ['fbs', 'exang']:
    # map True-1, False-0, nan stays NaN
    hd_clean[col] = hd_clean[col].map({True: 1, False: 0, 'True': 1, 'False': 0})

# 2d. Encode sex: Male→1, Female→0
hd_clean['sex'] = hd_clean['sex'].map({'Male': 1, 'Female': 0})

# 2e. One-hot encode remaining categoricals: cp, restecg
cp_dummies = pd.get_dummies(hd_clean['cp'],      prefix='cp',      dummy_na=False)
re_dummies = pd.get_dummies(hd_clean['restecg'], prefix='restecg', dummy_na=False)
hd_clean = pd.concat([hd_clean.drop(columns=['cp', 'restecg']), cp_dummies, re_dummies], axis=1)
print(f"One-hot encoded: cp → {list(cp_dummies.columns)}")
print(f"One-hot encoded: restecg → {list(re_dummies.columns)}")

# 2f. Median impute remaining numeric NaNs
numeric_cols_hd = hd_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_hd = [c for c in numeric_cols_hd if c != 'label']
before_missing  = hd_clean[numeric_cols_hd].isnull().sum().sum()
medians_hd = hd_clean[numeric_cols_hd].median()
hd_clean[numeric_cols_hd] = hd_clean[numeric_cols_hd].fillna(medians_hd)
after_missing = hd_clean[numeric_cols_hd].isnull().sum().sum()
print(f"Imputed {before_missing} missing numeric values → {after_missing} remaining")

# 2g. Add source tag
hd_clean['source'] = 0   # 0 = heart disease dataset

print(f"\nUCI after cleaning: {hd_clean.shape}")
print(f"Columns: {list(hd_clean.columns)}")


#  STEP 3 — Clean

section("Step 3 — Cleaning Pima Diabetes")

pima_clean = pima.copy()

# 3a. Binary label (already binary)
pima_clean['label'] = pima_clean['Outcome']
pima_clean.drop(columns=['Outcome'], inplace=True)
print(f"Label kept as-is: 0={(pima_clean['label']==0).sum()}  1={(pima_clean['label']==1).sum()}")

# 3b. Replace biologically impossible zeros with NaN, then median impute
zero_impute_cols = ['Glucose', 'Insulin', 'BMI', 'BloodPressure', 'SkinThickness']
for col in zero_impute_cols:
    n_zeros = (pima_clean[col] == 0).sum()
    pima_clean[col] = pima_clean[col].replace(0, np.nan)
    pima_clean[col] = pima_clean[col].fillna(pima_clean[col].median())
    print(f"  Replaced {n_zeros} zeros in {col} with median={pima_clean[col].median():.1f}")

# 3c. Rename to align with UCI where possible
pima_clean.rename(columns={
    'Pregnancies'             : 'pregnancies',
    'Glucose'                 : 'glucose',
    'BloodPressure'           : 'trestbps',       # align with UCI
    'SkinThickness'           : 'skin_thickness',
    'Insulin'                 : 'insulin',
    'BMI'                     : 'bmi',
    'DiabetesPedigreeFunction': 'diabetes_pedigree',
    'Age'                     : 'age',             # align with UCI
}, inplace=True)

# 3d. Source tag
pima_clean['source'] = 1   # 1 = pima dataset

print(f"\nPima after cleaning: {pima_clean.shape}")
print(f"Columns: {list(pima_clean.columns)}")


#  STEP 4 — Merging with full NaN alignment

section("Step 4 — Merging datasets")

# pd.concat will auto-align on column names and fill missing columns with NaN
merged = pd.concat([hd_clean, pima_clean], ignore_index=True, sort=False)
print(f"Merged shape (before imputation): {merged.shape}")
print(f"Total patients: {len(merged)}  "
      f"(UCI={len(hd_clean)}, Pima={len(pima_clean)})")

# Fill NaN that arose from mismatched columns with column median
#   e.g. Pima patients have NaN for 'sex', 'chol', 'thalch' etc.
#        UCI patients have NaN for 'glucose', 'insulin', 'bmi' etc.

feature_cols_all = [c for c in merged.columns if c not in ['label', 'source']]

# Compute medians from the full merged set, then fill
for col in feature_cols_all:
    if merged[col].dtype in [np.float64, np.int64, float, int]:
        col_median = merged[col].median()
        merged[col] = merged[col].fillna(col_median)
    else:
        # boolean-like columns stored as object
        col_mode = merged[col].mode()
        if len(col_mode) > 0:
            merged[col] = merged[col].fillna(col_mode[0])

# Ensure all feature columns are numeric float32
for col in feature_cols_all:
    merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0).astype(np.float32)

remaining_nan = merged[feature_cols_all].isnull().sum().sum()
print(f"Remaining NaN after imputation: {remaining_nan}")

print(f"\nMerged label distribution:")
print(merged['label'].value_counts().sort_index().to_string())
print(f"  Imbalance ratio: {merged['label'].value_counts(normalize=True).round(3).to_string()}")


#  STEP 5 — Feature scaling

section("Step 5 — Feature scaling (StandardScaler)")

# We scale the features and save both the unscaled CSV and the scaler
# The GNN will use the SCALED features
feature_cols = [c for c in merged.columns if c not in ['label', 'source']]
print(f"Feature columns ({len(feature_cols)}):")
for i, c in enumerate(feature_cols):
    print(f"  {i:2d}. {c}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(merged[feature_cols].values)

# Save scaler for reproducibility
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"\nScaler saved → scaler.pkl")

# Add scaled features back
scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
scaled_df['label']  = merged['label'].values
scaled_df['source'] = merged['source'].values

#  STEP 6 — Saving everything
section("Step 6 — Saving outputs")

merged.to_csv("merged_patients.csv", index=False)
scaled_df.to_csv("merged_patients_scaled.csv", index=False)
print(f"Saved: merged_patients.csv        ({len(merged)} rows, unscaled)")
print(f"Saved: merged_patients_scaled.csv ({len(scaled_df)} rows, scaled)")

# Also save feature column names for downstream scripts
with open("feature_cols.txt", "w") as f:
    f.write("\n".join(feature_cols))
print(f"Saved: feature_cols.txt           ({len(feature_cols)} features)")


#  PLOT — Correlation heatmap

section("Plotting correlation heatmap")

# Only plot numeric non-dummy columns (too many dummies makes it unreadable)
core_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalch',
             'exang', 'oldpeak', 'glucose', 'bmi', 'insulin',
             'pregnancies', 'diabetes_pedigree', 'label']
core_cols = [c for c in core_cols if c in merged.columns]

corr = merged[core_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, annot_kws={"size": 8}
)
ax.set_title("Feature Correlation Matrix (Merged Dataset)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] eda_correlation.png")

section("Phase 2 Complete")
print(f"  Total patients  : {len(merged)}")
print(f"  Feature count   : {len(feature_cols)}")
print(f"  Label=0 (healthy): {(merged['label']==0).sum()}")
print(f"  Label=1 (disease): {(merged['label']==1).sum()}")
print("\nNext step: run  python phase3_graph_construction.py")