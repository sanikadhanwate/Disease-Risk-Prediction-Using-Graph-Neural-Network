# Patient Disease Risk Predictor — Flask App

A clinical decision support interface powered by a Graph Attention Network (GAT)
trained on the merged UCI Heart Disease + Pima Diabetes patient similarity graph.

## Features

- **Real-time GAT prediction** — disease risk probability from clinical inputs
- **Feature attribution** — gradient-based explanation of which features drove the prediction
- **5 nearest neighbours** — shows the most similar patients from the training graph and their outcomes
- **GAT attention weights** — which neighbours the model focused on most
- **Counterfactual explanation** — minimum feature changes to flip the prediction (connects to Prof. Rundensteiner's VLDB 2024 actionable recourse work)

## Setup

### 1. Place datasets in this folder

```
disease_risk_app/
├── app.py
├── model.py
├── heart_disease_uci.csv     ← your existing file
├── diabetes.csv              ← download from Kaggle (Pima Indians)
├── templates/
│   └── index.html
└── README.md
```

Pima dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### 2. Install dependencies

```bash
pip install flask torch torch_geometric scikit-learn pandas numpy
```

### 3. Run

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

**First run:** The app will preprocess both datasets, build the patient similarity
graph (kNN k=5, cosine similarity), and train the GAT model (~2–5 minutes on CPU).
After the first run, the model is cached so subsequent startups are instant.

## How it works

1. You enter any combination of clinical values (all fields optional)
2. Missing values are filled with the population median from the training data
3. The patient is injected as a new node into the existing 1,688-patient graph
4. Edges are drawn to the 5 most similar patients using cosine kNN
5. The GAT runs forward pass and returns:
   - Disease probability
   - Gradient attribution per feature
   - Attention weights per neighbour
   - Counterfactual: minimum changes to flip prediction

## Architecture

```
GAT:
  GATConv(20 → 16, heads=8) + ELU + Dropout(0.6)
  GATConv(128 → 2, heads=1) + Softmax

Graph:
  1,688 nodes (patients)
  16,880 edges (kNN k=5, cosine similarity)
  70/15/15 train/val/test split (stratified)
```

## Research connection

The counterfactual explanation module extends the work of:
- VanNostrand et al. "Counterfactual Explanation Analytics: Empowering Lay Users
  to Take Action Against Consequential Automated Decisions." VLDB 2024.
  (Prof. Rundensteiner's DAISY lab, WPI)

The open research gap: counterfactuals for tabular classifiers are well-studied,
but for GNNs, changing a patient's features also shifts their neighbourhood structure,
creating a compound explanation problem not yet addressed in the literature.
