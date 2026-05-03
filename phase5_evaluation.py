import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from torch_geometric.nn   import GCNConv, GATConv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics      import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, roc_curve,
                                  confusion_matrix, classification_report)
from sklearn.manifold     import TSNE

os.makedirs("results", exist_ok=True)
sns.set_style("whitegrid")

def section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")

def check_file(path):
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        print(f"  Run the earlier phases first.")
        sys.exit(1)

for f in ["graph_data.pt", "merged_patients_scaled.csv",
          "feature_cols.txt", "results/gcn_model.pt",
          "results/gat_model.pt", "results/mlp_model.pt"]:
    check_file(f)

torch.manual_seed(42)
np.random.seed(42)


#  MODEL DEFINITIONS (must match Phase 4 exactly)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1   = GCNConv(in_channels, hidden)
        self.bn1     = torch.nn.BatchNorm1d(hidden)
        self.conv2   = GCNConv(hidden, hidden // 2)
        self.bn2     = torch.nn.BatchNorm1d(hidden // 2)
        self.linear  = torch.nn.Linear(hidden // 2, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear(x)

    def get_embeddings(self, x, edge_index, edge_weight=None):
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
            x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight)))
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden=16, out_channels=2,
                 heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1   = GATConv(in_channels, hidden,
                                heads=heads, dropout=dropout, concat=True)
        self.conv2   = GATConv(hidden * heads, out_channels,
                                heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, out_channels=2, dropout=0.5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.BatchNorm1d(hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, out_channels),
        )
    def forward(self, x, edge_index=None, edge_weight=None):
        return self.net(x)


#  STEP 1: Load everything

section("Step 1 — Loading data & models")

data = torch.load("graph_data.pt", weights_only=False)
df   = pd.read_csv("merged_patients_scaled.csv")
with open("feature_cols.txt") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

X_np = df[feature_cols].values.astype(np.float32)
y_np = df['label'].values.astype(np.int64)
n_features = len(feature_cols)

train_idx = data.train_mask.numpy().nonzero()[0]
val_idx   = data.val_mask.numpy().nonzero()[0]
test_idx  = data.test_mask.numpy().nonzero()[0]

X_train = X_np[train_idx]; y_train = y_np[train_idx]
X_test  = X_np[test_idx];  y_test  = y_np[test_idx]

# Load GNN models
gcn = GCN(n_features, 64, 2, 0.5)
gcn.load_state_dict(torch.load("results/gcn_model.pt", weights_only=True))
gcn.eval()

gat = GAT(n_features, 16, 2, 8, 0.6)
gat.load_state_dict(torch.load("results/gat_model.pt", weights_only=True))
gat.eval()

mlp = MLP(n_features, 64, 2, 0.5)
mlp.load_state_dict(torch.load("results/mlp_model.pt", weights_only=True))
mlp.eval()

print("GCN, GAT, MLP models loaded.")

#Re-train sklearn baselines (quick, same seeds)
print("Re-training sklearn baselines for evaluation ...")
lr_model = LogisticRegression(
    max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight='balanced',
    random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42)
gb_model.fit(X_train, y_train)
print("Baselines trained.")


#  STEP 2: Gather predictions for all models

section("Step 2 — Collecting predictions on test set")

@torch.no_grad()
def gnn_predict(model, data, mask, uses_edge_weight=True):
    model.eval()
    if uses_edge_weight:
        out = model(data.x, data.edge_index, data.edge_weight)
    else:
        out = model(data.x, data.edge_index)
    logits = out[mask]
    preds  = logits.argmax(dim=1).numpy()
    proba  = F.softmax(logits, dim=1)[:, 1].numpy()
    return preds, proba

gcn_preds, gcn_proba = gnn_predict(gcn, data, data.test_mask, uses_edge_weight=True)
gat_preds, gat_proba = gnn_predict(gat, data, data.test_mask, uses_edge_weight=False)
mlp_preds, mlp_proba = gnn_predict(mlp, data, data.test_mask, uses_edge_weight=True)

lr_preds  = lr_model.predict(X_test);  lr_proba  = lr_model.predict_proba(X_test)[:,1]
rf_preds  = rf_model.predict(X_test);  rf_proba  = rf_model.predict_proba(X_test)[:,1]
gb_preds  = gb_model.predict(X_test);  gb_proba  = gb_model.predict_proba(X_test)[:,1]


#  STEP 3: Full metrics table

section("Step 3 — Full Metrics Table")

def compute_metrics(name, preds, proba, y_true):
    return {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_true, preds), 4),
        "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, preds, zero_division=0), 4),
        "F1"       : round(f1_score(y_true, preds, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_true, proba), 4),
    }

results = [
    compute_metrics("GCN (graph)",    gcn_preds, gcn_proba, y_test),
    compute_metrics("GAT (graph)",    gat_preds, gat_proba, y_test),
    compute_metrics("MLP (no-graph)", mlp_preds, mlp_proba, y_test),
    compute_metrics("Logistic Reg.",  lr_preds,  lr_proba,  y_test),
    compute_metrics("Random Forest",  rf_preds,  rf_proba,  y_test),
    compute_metrics("Grad. Boosting", gb_preds,  gb_proba,  y_test),
]

results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison_final.csv", index=False)

header = f"\n  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
print(header)
print("  " + "─" * 70)
for _, row in results_df.iterrows():
    print(f"  {row['Model']:<22} {row['Accuracy']:>9.4f} {row['Precision']:>10.4f} "
          f"{row['Recall']:>8.4f} {row['F1']:>8.4f} {row['ROC-AUC']:>9.4f}")


#  STEP 4: Per-class classification reports
section("Step 4 — Classification Reports (GCN & GAT)")
print("\n  ── GCN ──")
print(classification_report(y_test, gcn_preds,
      target_names=['Healthy (0)', 'Disease (1)']))
print("\n  ── GAT ──")
print(classification_report(y_test, gat_preds,
      target_names=['Healthy (0)', 'Disease (1)']))

#  STEP 5: Ablation analysis

section("Step 5 — Graph Ablation Analysis")
best_gnn_f1 = max(
    compute_metrics("GCN", gcn_preds, gcn_proba, y_test)["F1"],
    compute_metrics("GAT", gat_preds, gat_proba, y_test)["F1"],
)
mlp_f1 = compute_metrics("MLP", mlp_preds, mlp_proba, y_test)["F1"]
gain   = best_gnn_f1 - mlp_f1

print(f"\n  Best GNN F1 (graph):  {best_gnn_f1:.4f}")
print(f"  MLP F1 (no graph) :  {mlp_f1:.4f}")
print(f"  F1 gain from graph:  {gain:+.4f}")

if gain > 0:
    print(f"\n  [CONCLUSION] The graph structure provides a positive F1 gain of {gain:.4f}.")
    print(f"  This supports the project hypothesis that patient similarity graphs")
    print(f"  improve disease risk prediction beyond feature-only approaches.")
else:
    print(f"\n  [NOTE] The F1 gain is {gain:.4f}. The graph structure did not improve F1")
    print(f"  on this test split. Consider: different k, threshold, or feature subset.")
    print(f"  Report this honestly — negative results are valid scientific findings.")


#  PLOTS

section("Generating evaluation plots")

COLORS = {
    "GCN (graph)"   : "#378add",
    "GAT (graph)"   : "#1d9e75",
    "MLP (no-graph)": "#888780",
    "Logistic Reg." : "#EF9F27",
    "Random Forest" : "#D85A30",
    "Grad. Boosting": "#D4537E",
}

#Plot 1: ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))

for (name, preds, proba) in [
    ("GCN (graph)",    gcn_preds, gcn_proba),
    ("GAT (graph)",    gat_preds, gat_proba),
    ("MLP (no-graph)", mlp_preds, mlp_proba),
    ("Logistic Reg.",  lr_preds,  lr_proba),
    ("Random Forest",  rf_preds,  rf_proba),
    ("Grad. Boosting", gb_preds,  gb_proba),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    lw  = 2.5 if "GCN" in name or "GAT" in name else 1.5
    ls  = '-'  if "graph" in name or "no-graph" in name else '--'
    ax.plot(fpr, tpr, lw=lw, ls=ls,
            color=COLORS[name], label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5, label='Random')
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — All Models (Test Set)", fontsize=13)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("results/eval_roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] results/eval_roc_curves.png")

#Plot 2: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, preds) in zip(axes,
        [("GCN", gcn_preds), ("GAT", gat_preds)]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Healthy', 'Disease'],
                yticklabels=['Healthy', 'Disease'],
                linewidths=0.5, linecolor='white')
    ax.set_title(f"{name} — Confusion Matrix\n(Test Set)", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("results/eval_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] results/eval_confusion_matrices.png")

#Plot 3: Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
model_names = results_df["Model"].tolist()
x = np.arange(len(model_names))
bar_colors  = [COLORS.get(m, '#888780') for m in model_names]

for ax, metric in zip(axes, ["F1", "ROC-AUC"]):
    vals = results_df[metric].tolist()
    bars = ax.bar(x, vals, color=bar_colors, width=0.6, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{metric} Score — All Models", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', lw=0.8, alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontsize=8)

plt.suptitle("Model Performance Comparison — Test Set", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("results/eval_model_comparison_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] results/eval_model_comparison_bar.png")

#Plot 4: Feature Importance (Random Forest)
fi = pd.Series(rf_model.feature_importances_, index=feature_cols)
fi = fi.sort_values(ascending=True).tail(20)   # top 20

fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#378add' if 'cp_' not in c and 'restecg_' not in c else '#1d9e75'
          for c in fi.index]
ax.barh(fi.index, fi.values, color=colors, edgecolor='white', height=0.65)
ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
ax.set_title("Top 20 Feature Importances — Random Forest\n"
             "(blue=continuous/binary, green=one-hot)", fontsize=11)
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig("results/eval_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] results/eval_feature_importance.png")

#Plot 5: Ablation summary
ablation_models = ["GCN (graph)", "GAT (graph)", "MLP (no-graph)",
                   "Random Forest", "Grad. Boosting"]
ablation_df = results_df[results_df["Model"].isin(ablation_models)].copy()

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(ablation_df))
bar_c = [COLORS.get(m, '#888780') for m in ablation_df["Model"]]
bars  = ax.bar(x, ablation_df["F1"].values, color=bar_c,
               width=0.55, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(ablation_df["Model"].tolist(), rotation=15,
                   ha='right', fontsize=10)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("Graph Ablation — GNN with Graph vs. MLP without Graph\n"
             "(demonstrates contribution of graph structure)", fontsize=11)
ax.set_ylim(0, 1.05)
for bar, val in zip(bars, ablation_df["F1"].values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha='center', va='bottom', fontsize=9)

# Annotate the gain
gnn_x  = 0   # GCN bar index
mlp_x  = 2   # MLP bar index
gnn_f1 = ablation_df.iloc[0]["F1"]
mlp_f1_val = ablation_df.iloc[2]["F1"]
ax.annotate("", xy=(gnn_x, gnn_f1 + 0.05),
            xytext=(mlp_x, gnn_f1 + 0.05),
            arrowprops=dict(arrowstyle='<->', color='#e24b4a', lw=1.5))
ax.text((gnn_x + mlp_x)/2, gnn_f1 + 0.07,
        f"Δ={gnn_f1 - mlp_f1_val:+.3f}",
        ha='center', fontsize=9, color='#e24b4a')

plt.tight_layout()
plt.savefig("results/eval_ablation_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] results/eval_ablation_summary.png")

#Plot 6: t-SNE of GCN embeddings
section("t-SNE visualization of GCN node embeddings")
print("  Computing GCN embeddings for all nodes ...")

with torch.no_grad():
    embeddings = gcn.get_embeddings(
        data.x, data.edge_index, data.edge_weight).numpy()

print(f"  Embedding matrix: {embeddings.shape}")
print("  Running t-SNE (this may take ~30 seconds) ...")

tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            n_iter=1000, learning_rate='auto', init='pca')
emb_2d = tsne.fit_transform(embeddings)
print("  t-SNE done.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Colour by label
label_colors = np.where(y_np == 0, '#1d9e75', '#e24b4a')
axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1],
                c=label_colors, s=8, alpha=0.6, linewidths=0)
axes[0].set_title("GCN Embeddings — coloured by label\n(green=healthy, red=disease)",
                  fontsize=11)
axes[0].set_xlabel("t-SNE dim 1"); axes[0].set_ylabel("t-SNE dim 2")
axes[0].axis('off')

# Colour by data source
src = df['source'].values
source_colors = np.where(src == 0, '#378add', '#EF9F27')
axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1],
                c=source_colors, s=8, alpha=0.6, linewidths=0)
axes[1].set_title("GCN Embeddings — coloured by source\n"
                  "(blue=UCI heart, orange=Pima diabetes)", fontsize=11)
axes[1].set_xlabel("t-SNE dim 1"); axes[1].set_ylabel("t-SNE dim 2")
axes[1].axis('off')

from matplotlib.lines import Line2D
legend_label = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#1d9e75', markersize=7, label='Healthy'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#e24b4a', markersize=7, label='Disease'),
]
legend_src = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#378add', markersize=7, label='UCI Heart'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#EF9F27', markersize=7, label='Pima Diabetes'),
]
axes[0].legend(handles=legend_label, loc='lower right', fontsize=9)
axes[1].legend(handles=legend_src,   loc='lower right', fontsize=9)

plt.suptitle("t-SNE of GCN Node Embeddings", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("results/eval_node_embeddings.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] results/eval_node_embeddings.png")


#  FINAL PRINTED REPORT

section("FINAL REPORT — Disease Risk Prediction using GNNs")

print("""
  PROJECT SUMMARY
  ─────────────────────────────────────────────────────────────
  Approach   : Merged UCI Heart Disease + Pima Diabetes datasets
               into a single patient similarity graph.
               Each patient = node. Edges connect k=5 most
               similar patients by cosine similarity.

  Graph stats:""")
print(f"    Nodes       : {data.x.shape[0]}")
print(f"    Edges (kNN) : {data.edge_index.shape[1]}")
print(f"    Features    : {data.x.shape[1]}")

print("""
  RESULTS SUMMARY
  ─────────────────────────────────────────────────────────────""")
print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} "
      f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("  " + "─" * 70)
for _, row in results_df.iterrows():
    marker = " ◄" if row["Model"] in ("GCN (graph)", "GAT (graph)") else ""
    print(f"  {row['Model']:<22} {row['Accuracy']:>9.4f} "
          f"{row['Precision']:>10.4f} {row['Recall']:>8.4f} "
          f"{row['F1']:>8.4f} {row['ROC-AUC']:>9.4f}{marker}")

print(f"""
  ABLATION FINDING
  ─────────────────────────────────────────────────────────────
  GNN (with graph edges) vs MLP (no graph, same features):
    F1 gain = {best_gnn_f1 - mlp_f1_val:+.4f}
  {'Graph structure improved performance.' if best_gnn_f1 > mlp_f1_val else
   'Graph structure did not improve over MLP on this split.'}

  ALL OUTPUT FILES SAVED TO: results/
  ─────────────────────────────────────────────────────────────
    eval_roc_curves.png
    eval_confusion_matrices.png
    eval_model_comparison_bar.png
    eval_feature_importance.png
    eval_ablation_summary.png
    eval_node_embeddings.png
    model_comparison_final.csv
""")

section("Phase 5 Complete — Project pipeline finished!")