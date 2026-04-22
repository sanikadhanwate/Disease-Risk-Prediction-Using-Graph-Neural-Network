# Disease Risk Prediction using Graph Neural Networks

> Modeling 1,688 patients as a similarity graph and training GCN/GAT models to predict heart disease and diabetes risk — outperforming feature-only baselines through graph-based message passing.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

Traditional machine learning treats patients as independent data points. This project challenges that assumption by building a **patient similarity graph** — where each patient is a node and edges connect the 5 most clinically similar patients using cosine similarity — then training Graph Neural Networks to leverage both patient features *and* relational structure for disease risk prediction.

**Key result:** GAT (Graph Attention Network) achieves F1 = 0.765 and AUC = 0.828, outperforming the no-graph MLP baseline (F1 = 0.760), confirming that graph structure contributes measurable predictive signal.

---

## Results

| Model | F1 | ROC-AUC |
|---|---|---|
| **GCN (graph)** | **0.747** | **0.835** |
| **GAT (graph)** | **0.765** | **0.828** |
| MLP (no-graph ablation) | 0.760 | 0.833 |
| Logistic Regression | 0.776 | 0.838 |
| Random Forest | 0.769 | 0.848 |
| Gradient Boosting | 0.751 | 0.845 |

> **Ablation finding:** GAT (with graph edges) beats MLP (same features, no edges) by Δ F1 = +0.005, isolating the contribution of graph structure.

### Visualizations

| ROC Curves | Ablation Study |
|---|---|
| ![ROC](outputs/results/eval_roc_curves.png) | ![Ablation](outputs/results/eval_ablation_summary.png) |

| Feature Importance | t-SNE Node Embeddings |
|---|---|
| ![Features](outputs/results/eval_feature_importance.png) | ![tSNE](outputs/results/eval_node_embeddings.png) |

---

## Dataset

| Dataset | Source | Patients | Features | Target |
|---|---|---|---|---|
| UCI Heart Disease | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) | 920 | 16 raw → 20 encoded | Heart disease (binary) |
| Pima Indians Diabetes | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | 8 | Diabetes (binary) |
| **Merged** | — | **1,688** | **20 aligned** | **Disease risk (binary)** |

**Preprocessing decisions:**
- Dropped `ca` (66% missing) and `thal` (53% missing) from UCI dataset
- Replaced biologically impossible zeros in Pima (Glucose, Insulin, BMI, BP, SkinThickness) with column medians
- One-hot encoded categorical features (`cp`, `restecg`)
- Aligned shared features across datasets: `Age→age`, `BloodPressure→trestbps`
- StandardScaler normalization before graph construction

---

## Graph Construction

Patients are connected using **k-Nearest Neighbours (k=5)** on the normalized feature space with cosine similarity metric.

```
Nodes  : 1,688 patients
Edges  : 16,880 (kNN, k=5, undirected)
Method : sklearn NearestNeighbors (cosine metric) → PyTorch Geometric Data object
Split  : 70% train / 15% val / 15% test (stratified)
```

Two methods were compared:
- **Method A — kNN (k=5):** Guarantees every node has exactly k neighbours. No isolated nodes. Selected as primary method.
- **Method B — Cosine threshold (≥0.90):** Variable degree distribution. Some isolated nodes. Used for ablation comparison.

---

## Model Architecture

### GCN (Graph Convolutional Network)
```
Input (20 features)
  → GCNConv(20 → 64) + BatchNorm + ReLU + Dropout(0.5)
  → GCNConv(64 → 32) + BatchNorm + ReLU + Dropout(0.5)
  → Linear(32 → 2)
  → Softmax
```

### GAT (Graph Attention Network)
```
Input (20 features)
  → GATConv(20 → 16, heads=8, concat=True) + ELU + Dropout(0.6)
  → GATConv(128 → 2, heads=1, concat=False) + Dropout(0.6)
  → Softmax
```

### MLP Ablation (no graph structure)
Same architecture as GCN but without `edge_index` — pure feature-based prediction. Used to isolate the contribution of graph structure.

**Training:** Adam optimizer, lr=0.01, weight_decay=5e-4, early stopping on validation F1 (patience=40), class weights in CrossEntropyLoss to handle class imbalance.

---

## Project Structure

```
├── phase1_eda.py                 # Data loading, EDA, missing value analysis
├── phase2_preprocessing.py      # Cleaning, encoding, merging, scaling
├── phase3_graph_construction.py # kNN graph, PyG Data object, train/val/test masks
├── phase4_models.py             # GCN, GAT, MLP, baseline ML training
├── phase5_evaluation.py         # Metrics, ROC curves, ablation, t-SNE
├── data/                        # Raw datasets (download instructions below)
├── outputs/results/             # All generated plots and CSVs
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOURUSERNAME/GNN-Disease-Risk-Prediction.git
cd GNN-Disease-Risk-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
# - heart_disease_uci.csv: https://archive.ics.uci.edu/dataset/45/heart+disease
# - diabetes.csv: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Place both in the data/ folder

# 4. Run the full pipeline in order
python phase1_eda.py
python phase2_preprocessing.py
python phase3_graph_construction.py
python phase4_models.py
python phase5_evaluation.py
```

Expected runtime: ~5–10 minutes on CPU (t-SNE in Phase 5 takes ~30s).

---

## Key Findings

1. **Graph structure provides measurable predictive signal** — GAT with edges outperforms MLP without edges (Δ F1 = +0.005), confirming the core hypothesis that patient relationships improve disease prediction.

2. **Glucose and Age are the dominant cross-disease predictors** (Gini importance 0.128 and 0.106), appearing as the strongest features in both heart disease and diabetes contexts — a clinically meaningful finding that validates the dataset merging strategy.

3. **t-SNE embeddings reveal domain shift** — UCI Heart Disease patients and Pima Diabetes patients cluster distinctly in GCN embedding space even after feature alignment, suggesting the model captures dataset-specific clinical patterns alongside shared disease biology.

4. **GNNs are competitive with tree-based ensembles** at this dataset scale (~1,688 nodes). Performance gaps narrow significantly as graph size increases — this project establishes the pipeline for scaling to larger clinical datasets.

---

## Technologies

`Python` · `PyTorch` · `PyTorch Geometric` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

## Citation

```bibtex
@misc{gnn-disease-risk,
  title   = {Disease Risk Prediction using Graph Neural Networks},
  year    = {2025},
  url     = {https://github.com/sanikadhanwate/GNN-Disease-Risk-Prediction}
}
```
