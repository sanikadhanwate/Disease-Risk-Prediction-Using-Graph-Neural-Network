"""
PHASE 1 — Data Loading & Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys

#Paths
HD_PATH   = "heart_disease_uci.csv"
PIMA_PATH = "diabetes.csv"

#Helpers
def section(title):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def check_file(path):
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        if path == PIMA_PATH:
            print("  Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
            print("  Save as 'diabetes.csv' in the same folder as this script.")
        sys.exit(1)

#Load
check_file(HD_PATH)
check_file(PIMA_PATH)

hd   = pd.read_csv(HD_PATH)
pima = pd.read_csv(PIMA_PATH)

#Section 1: Basic Info
section("UCI Heart Disease — Basic Info")
print(f"Shape            : {hd.shape}  ({hd.shape[0]} patients, {hd.shape[1]} columns)")
print(f"\nColumns          : {list(hd.columns)}")
print(f"\nData types:\n{hd.dtypes.to_string()}")
print(f"\nFirst 5 rows:\n{hd.head(5).to_string()}")

section("Pima Diabetes — Basic Info")
print(f"Shape            : {pima.shape}  ({pima.shape[0]} patients, {pima.shape[1]} columns)")
print(f"\nColumns          : {list(pima.columns)}")
print(f"\nData types:\n{pima.dtypes.to_string()}")
print(f"\nFirst 5 rows:\n{pima.head(5).to_string()}")

#Section 2: Missing Values
section("UCI Heart Disease — Missing Values")
hd_missing = hd.isnull().sum()
hd_missing_pct = (hd.isnull().mean() * 100).round(2)
hd_mv = pd.DataFrame({"Missing Count": hd_missing, "Missing %": hd_missing_pct})
hd_mv = hd_mv[hd_mv["Missing Count"] > 0].sort_values("Missing %", ascending=False)
print(hd_mv.to_string())
print("\n[NOTE] Columns with >50% missing (ca=66%, thal=53%) will be DROPPED in Phase 2.")
print("[NOTE] 'slope' has 34% missing — will be dropped too (too unreliable to impute for a graph).")

section("Pima Diabetes — Missing Values")
print("Structural NaN count:", pima.isnull().sum().sum())
print("\nBiologically-impossible zeros (treated as missing):")
zero_cols = ['Glucose', 'Insulin', 'BMI', 'BloodPressure', 'SkinThickness']
for col in zero_cols:
    n_zeros = (pima[col] == 0).sum()
    pct = n_zeros / len(pima) * 100
    print(f"  {col:18s}: {n_zeros:3d} zeros  ({pct:.1f}%)")

#Section 3: Target Distribution
section("UCI Heart Disease — Target Distribution (num)")
print(hd['num'].value_counts().sort_index().to_string())
binary_label = (hd['num'] > 0).astype(int)
print(f"\nAfter binarisation (0=healthy, 1=disease):")
print(binary_label.value_counts().sort_index().to_string())
hd_ratio = binary_label.value_counts(normalize=True).round(3)
print(f"  Class ratio — 0: {hd_ratio[0]:.1%}   1: {hd_ratio[1]:.1%}")

section("Pima Diabetes — Target Distribution (Outcome)")
print(pima['Outcome'].value_counts().sort_index().to_string())
pima_ratio = pima['Outcome'].value_counts(normalize=True).round(3)
print(f"  Class ratio — 0: {pima_ratio[0]:.1%}   1: {pima_ratio[1]:.1%}")
print("\n[NOTE] Both datasets are imbalanced. Phase 4 will use class weights in the loss function.")

#Section 4: Descriptive Statistics
section("UCI Heart Disease — Numeric Feature Stats")
numeric_hd = hd.select_dtypes(include=[np.number]).drop(columns=['id', 'num'])
print(numeric_hd.describe().round(2).to_string())

section("Pima Diabetes — Numeric Feature Stats")
print(pima.describe().round(2).to_string())

#Section 5: Categorical Columns in UCI
section("UCI Heart Disease — Categorical Column Unique Values")
cat_cols = hd.select_dtypes(include=['object', 'string']).columns.tolist()
for col in cat_cols:
    print(f"  {col:12s}: {sorted(hd[col].dropna().unique().tolist())}")

#Section 6: Sub-dataset breakdown
section("UCI Heart Disease — Sub-dataset Breakdown")
print(hd.groupby('dataset')['num'].value_counts().unstack(fill_value=0).to_string())

#Section 7: Shared Features (for graph construction)
section("Shared Features Available for Graph Edge Construction")
print("  UCI Heart Disease  | Pima Diabetes      | Aligned Name")
print("  ─────────────────────────────────────────────────────────")
print("  age                | Age                | age")
print("  trestbps           | BloodPressure      | blood_pressure")
print("  chol               | (BMI proxy)        | (partial)")
print("  thalch             | (no equivalent)    | heart_rate")
print("  oldpeak            | (no equivalent)    | —")
print("  (no equivalent)    | Glucose            | glucose")
print("  (no equivalent)    | BMI                | bmi")
print("  (no equivalent)    | Insulin            | insulin")
print("\n[NOTE] age + blood_pressure are the cleanest shared features.")
print("       All numeric features will be used in the node feature matrix.")
print("       Edge similarity will be computed on the FULL normalized feature vector.")


#  PLOTS

sns.set_style("whitegrid")
sns.set_palette("muted")

#Plot 1: Missing Value Heatmap (UCI)
fig, ax = plt.subplots(figsize=(10, 4))
missing_matrix = hd.isnull().astype(int)
cols_with_missing = hd.columns[hd.isnull().any()].tolist()
sns.heatmap(
    missing_matrix[cols_with_missing].T,
    cmap=["#f0f0f0", "#e24b4a"],
    cbar=False,
    ax=ax,
    linewidths=0,
    yticklabels=cols_with_missing
)
ax.set_title("UCI Heart Disease — Missing Value Map\n(red = missing)", fontsize=13, pad=12)
ax.set_xlabel("Patient index", fontsize=11)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("eda_missing_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved] eda_missing_heatmap.png")

#Plot 2: Class Distribution
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# UCI raw
hd['num'].value_counts().sort_index().plot(
    kind='bar', ax=axes[0], color='#378add', edgecolor='white', width=0.6)
axes[0].set_title("UCI Heart Disease\nRaw target (num 0–4)", fontsize=11)
axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=0)
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontsize=9)

# UCI binary
binary_label.value_counts().sort_index().plot(
    kind='bar', ax=axes[1], color=['#1d9e75', '#e24b4a'], edgecolor='white', width=0.5)
axes[1].set_title("UCI Heart Disease\nBinarised (0=healthy, 1=disease)", fontsize=11)
axes[1].set_xlabel("Class"); axes[1].set_ylabel("Count")
axes[1].set_xticklabels(['Healthy (0)', 'Disease (1)'], rotation=0)
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontsize=9)

# Pima
pima['Outcome'].value_counts().sort_index().plot(
    kind='bar', ax=axes[2], color=['#1d9e75', '#e24b4a'], edgecolor='white', width=0.5)
axes[2].set_title("Pima Diabetes\nOutcome (0=no diabetes, 1=diabetes)", fontsize=11)
axes[2].set_xlabel("Class"); axes[2].set_ylabel("Count")
axes[2].set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'], rotation=0)
for p in axes[2].patches:
    axes[2].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontsize=9)

plt.suptitle("Class Distribution — Both Datasets", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("eda_class_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] eda_class_dist.png")

#Plot 3: Feature Histograms
fig, axes = plt.subplots(2, 5, figsize=(18, 8))

# UCI numeric features
hd_num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
for i, col in enumerate(hd_num_cols):
    axes[0, i].hist(hd[col].dropna(), bins=25, color='#378add',
                    edgecolor='white', alpha=0.85)
    axes[0, i].set_title(f"UCI: {col}", fontsize=10)
    axes[0, i].set_ylabel("Count" if i == 0 else "")

# Pima numeric features
pima_num_cols = ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'Age']
for i, col in enumerate(pima_num_cols):
    axes[1, i].hist(pima[col].dropna(), bins=25, color='#1d9e75',
                    edgecolor='white', alpha=0.85)
    axes[1, i].set_title(f"Pima: {col}", fontsize=10)
    axes[1, i].set_ylabel("Count" if i == 0 else "")

plt.suptitle("Feature Distributions — UCI (blue) vs Pima (green)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("eda_feature_hist.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] eda_feature_hist.png")

section("Phase 1 Complete")
print("Next step: run  python phase2_preprocessing.py")
