"""
PHASE 4 — Model Training: GNN (GCN + GAT) + Baseline ML
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from torch_geometric.data import Data
from torch_geometric.nn   import GCNConv, GATConv

from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,
                                      f1_score, roc_auc_score, classification_report)

os.makedirs("results", exist_ok=True)

def section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")

def check_file(path):
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        print(f"  Run the earlier phases first.")
        sys.exit(1)

check_file("graph_data.pt")
check_file("merged_patients_scaled.csv")
check_file("feature_cols.txt")

#Reproducibility
torch.manual_seed(42)
np.random.seed(42)

#  STEP 1 — Load data
section("Step 1 — Loading data")

data = torch.load("graph_data.pt", weights_only=False)
print(f"Graph loaded: {data}")

df = pd.read_csv("merged_patients_scaled.csv")
with open("feature_cols.txt") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

X_np = df[feature_cols].values.astype(np.float32)
y_np = df['label'].values.astype(np.int64)

train_idx = data.train_mask.numpy().nonzero()[0]
val_idx   = data.val_mask.numpy().nonzero()[0]
test_idx  = data.test_mask.numpy().nonzero()[0]

n_features = data.x.shape[1]
n_classes  = 2
class_weights = data.class_weights

print(f"Features         : {n_features}")
print(f"Train / Val / Test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
print(f"Class weights    : {class_weights.numpy().round(4)}")


#  MODEL DEFINITIONS

#GCN
class GCN(torch.nn.Module):
    """
    2-layer Graph Convolutional Network.
    Input → GCNConv(F→64) → BatchNorm → ReLU → Dropout →
            GCNConv(64→32) → BatchNorm → ReLU → Dropout →
            Linear(32→2)
    """
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
        """Return node embeddings from the second conv layer (for visualization)."""
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
            x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight)))
        return x


#GAT
class GAT(torch.nn.Module):
    """
    2-layer Graph Attention Network.
    Input → GATConv(F→16, heads=8) → ELU → Dropout →
            GATConv(128→2, heads=1, concat=False) → return logits
    """
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


# MLP (no-graph ablation)
class MLP(torch.nn.Module):
    """
    Plain MLP — same features, NO graph structure.
    Used to show that graph edges provide extra signal.
    """
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


#  TRAINING UTILITY

def train_gnn(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    # GCN uses edge_weight; GAT does not
    if isinstance(model, GCN) or isinstance(model, MLP):
        out = model(data.x, data.edge_index, data.edge_weight)
    else:
        out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_gnn(model, data, mask):
    model.eval()
    if isinstance(model, GCN) or isinstance(model, MLP):
        out    = model(data.x, data.edge_index, data.edge_weight)
    else:
        out    = model(data.x, data.edge_index)
    logits = out[mask]
    preds  = logits.argmax(dim=1).numpy()
    proba  = F.softmax(logits, dim=1)[:, 1].numpy()
    true   = data.y[mask].numpy()
    metrics = {
        "accuracy" : round(accuracy_score(true, preds), 4),
        "precision": round(precision_score(true, preds, zero_division=0), 4),
        "recall"   : round(recall_score(true, preds, zero_division=0), 4),
        "f1"       : round(f1_score(true, preds, zero_division=0), 4),
        "roc_auc"  : round(roc_auc_score(true, proba), 4),
    }
    return metrics, preds, proba


def train_loop(model, data, model_name, epochs=300, lr=0.01,
               weight_decay=5e-4, patience=30, save_path=None):
    """
    Full training loop with early stopping on validation F1.
    Returns: (best_val_metrics, test_metrics, history)
    """
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-5)

    history = {"train_loss": [], "val_f1": [], "val_loss": []}
    best_val_f1    = -1.0
    best_state     = None
    patience_count = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_gnn(model, data, optimizer, criterion)

        # Validation
        model.eval()
        with torch.no_grad():
            if isinstance(model, GCN) or isinstance(model, MLP):
                val_out  = model(data.x, data.edge_index, data.edge_weight)
            else:
                val_out  = model(data.x, data.edge_index)
            val_loss = criterion(val_out[data.val_mask],
                                 data.y[data.val_mask]).item()

        val_metrics, _, _ = eval_gnn(model, data, data.val_mask)
        val_f1 = val_metrics["f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        scheduler.step(val_f1)

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            print(f"  [Early stop] Epoch {epoch}  best val F1={best_val_f1:.4f}")
            break

        if epoch % 50 == 0:
            print(f"  [{model_name}] Epoch {epoch:3d} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"val_f1={val_f1:.4f}")

    # Restore best weights & evaluate on test set
    model.load_state_dict(best_state)
    if save_path:
        torch.save(best_state, save_path)
        print(f"  Model saved → {save_path}")

    test_metrics, test_preds, test_proba = eval_gnn(model, data, data.test_mask)
    elapsed = time.time() - t0
    print(f"  Training time : {elapsed:.1f}s")
    print(f"  Best val F1   : {best_val_f1:.4f}")
    print(f"  Test metrics  : {test_metrics}")

    return best_val_f1, test_metrics, history, test_preds, test_proba



#  STEP 2 — Train GCN
section("Step 2 — Training GCN")

gcn = GCN(in_channels=n_features, hidden=64, out_channels=2, dropout=0.5)
print(f"GCN parameters: {sum(p.numel() for p in gcn.parameters()):,}")

gcn_val_f1, gcn_test, gcn_history, gcn_preds, gcn_proba = train_loop(
    gcn, data,
    model_name="GCN",
    epochs=400,
    lr=0.01,
    weight_decay=5e-4,
    patience=40,
    save_path="results/gcn_model.pt"
)


#  STEP 3 — Train GAT
section("Step 3 — Training GAT")

gat = GAT(in_channels=n_features, hidden=16, out_channels=2,
          heads=8, dropout=0.6)
print(f"GAT parameters: {sum(p.numel() for p in gat.parameters()):,}")

gat_val_f1, gat_test, gat_history, gat_preds, gat_proba = train_loop(
    gat, data,
    model_name="GAT",
    epochs=400,
    lr=0.005,
    weight_decay=1e-3,
    patience=40,
    save_path="results/gat_model.pt"
)


#  STEP 4 — Train MLP ablation (no graph structure)
section("Step 4 — Training MLP ablation (no graph edges)")

mlp = MLP(in_channels=n_features, hidden=64, out_channels=2, dropout=0.5)
print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters()):,}")

mlp_val_f1, mlp_test, mlp_history, mlp_preds, mlp_proba = train_loop(
    mlp, data,
    model_name="MLP",
    epochs=400,
    lr=0.01,
    weight_decay=5e-4,
    patience=40,
    save_path="results/mlp_model.pt"
)


#  STEP 5 — Train baseline ML models (no graph)
section("Step 5 — Training baseline ML models")

X_train = X_np[train_idx]
y_train = y_np[train_idx]
X_val   = X_np[val_idx]
y_val   = y_np[val_idx]
X_test  = X_np[test_idx]
y_test  = y_np[test_idx]

def eval_sklearn(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model"    : model_name,
        "accuracy" : round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall"   : round(recall_score(y_test, preds, zero_division=0), 4),
        "f1"       : round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc"  : round(roc_auc_score(y_test, proba), 4),
    }
    print(f"\n  [{model_name}] {metrics}")
    print(classification_report(y_test, preds, target_names=['Healthy', 'Disease']))
    return metrics

# Logistic Regression
print("\n  Training Logistic Regression ...")
lr_model = LogisticRegression(
    max_iter=1000, class_weight='balanced', random_state=42, C=1.0)
lr_model.fit(X_train, y_train)
lr_metrics = eval_sklearn(lr_model, X_test, y_test, "Logistic Regression")

# Random Forest
print("\n  Training Random Forest ...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight='balanced',
    random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_metrics = eval_sklearn(rf_model, X_test, y_test, "Random Forest")

# Gradient Boosting
print("\n  Training Gradient Boosting ...")
gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42)
gb_model.fit(X_train, y_train)
gb_metrics = eval_sklearn(gb_model, X_test, y_test, "Gradient Boosting")


#  STEP 6 — Compile results
section("Step 6 — Compiling all results")

all_results = pd.DataFrame([
    {"model": "GCN (graph)",       **gcn_test},
    {"model": "GAT (graph)",       **gat_test},
    {"model": "MLP (no-graph)",    **mlp_test},
    lr_metrics,
    rf_metrics,
    gb_metrics,
])

# reorder columns
cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
all_results = all_results[cols]
all_results.to_csv("results/model_comparison.csv", index=False)

print("\n  ┌─────────────────────────────────────────────────────────────┐")
print("  │                  MODEL COMPARISON — TEST SET                 │")
print("  ├──────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┤")
print(f"  │ {'Model':<20} │ {'Accuracy':>8} │ {'Precision':>8} │ {'Recall':>8} │ {'F1':>8} │ {'ROC-AUC':>8} │")
print("  ├──────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
for _, row in all_results.iterrows():
    print(f"  │ {row['model']:<20} │ {row['accuracy']:>8.4f} │ "
          f"{row['precision']:>8.4f} │ {row['recall']:>8.4f} │ "
          f"{row['f1']:>8.4f} │ {row['roc_auc']:>8.4f} │")
print("  └──────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

print(f"\n  [Saved] results/model_comparison.csv")


#  PLOTS

sns.set_style("whitegrid")

# ── Plot 1: GCN training curves ───────────────────────────────
def plot_training(history, model_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], color='#378add', label='Train loss')
    axes[0].plot(history["val_loss"],   color='#e24b4a', label='Val loss', linestyle='--')
    axes[0].set_title(f"{model_name} — Loss curve", fontsize=11)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["val_f1"], color='#1d9e75', label='Val F1')
    axes[1].axhline(max(history["val_f1"]), color='#e24b4a',
                    linestyle=':', alpha=0.7, label=f"Best={max(history['val_f1']):.4f}")
    axes[1].set_title(f"{model_name} — Validation F1", fontsize=11)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1 Score")
    axes[1].legend()

    plt.suptitle(f"{model_name} Training Curves", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")

plot_training(gcn_history, "GCN", "results/gcn_training_curve.png")
plot_training(gat_history, "GAT", "results/gat_training_curve.png")
plot_training(mlp_history, "MLP (ablation)", "results/mlp_training_curve.png")

section("Phase 4 Complete")
print("  All models trained. Outputs in results/")
print("  Next step: run  python phase5_evaluation.py")
