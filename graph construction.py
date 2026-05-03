import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from collections import Counter
import os, sys

#Paths
SCALED_CSV    = "merged_patients_scaled.csv"
FEAT_COLS_TXT = "feature_cols.txt"
GRAPH_OUT     = "graph_data.pt"

def section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")

def check_file(path):
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        print(f"  Make sure Phase 2 ran successfully first.")
        sys.exit(1)

check_file(SCALED_CSV)
check_file(FEAT_COLS_TXT)

#  STEP 1: Load preprocessed data

section("Step 1 — Loading preprocessed data")

df = pd.read_csv(SCALED_CSV)
with open(FEAT_COLS_TXT) as f:
    feature_cols = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(df)} patients, {len(feature_cols)} features")

X = df[feature_cols].values.astype(np.float32)   # node feature matrix
y = df['label'].values.astype(np.int64)           # node labels

n_nodes     = len(X)
n_features  = X.shape[1]
n_class0    = int((y == 0).sum())
n_class1    = int((y == 1).sum())

print(f"Node feature matrix X : {X.shape}")
print(f"Label vector y        : {y.shape}   "
      f"(0={n_class0}, 1={n_class1})")


#  STEP 2: Method A: kNN Graph (k=5)

section("Step 2 — Method A: kNN Graph (k=5, cosine metric)")

K = 5
nbrs = NearestNeighbors(n_neighbors=K + 1, metric='cosine', algorithm='brute')
nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)

# Build edge lists — skip self-loop (index 0 is the node itself)
src_list_knn, dst_list_knn, weights_knn = [], [], []
for i in range(n_nodes):
    for j_idx in range(1, K + 1):           # skip index 0 (self)
        j   = indices[i, j_idx]
        sim = 1.0 - distances[i, j_idx]     # cosine similarity from distance
        src_list_knn.append(i)
        dst_list_knn.append(j)
        weights_knn.append(sim)

edge_index_knn = torch.tensor([src_list_knn, dst_list_knn], dtype=torch.long)
edge_weight_knn = torch.tensor(weights_knn, dtype=torch.float)

# Make undirected (add reverse edges)
edge_index_knn, edge_weight_knn = to_undirected(
    edge_index_knn, edge_attr=edge_weight_knn, num_nodes=n_nodes, reduce='mean')

n_edges_knn = edge_index_knn.shape[1]
avg_degree_knn = n_edges_knn / n_nodes

print(f"  Edges (directed→undirected): {K * n_nodes} → {n_edges_knn}")
print(f"  Average node degree        : {avg_degree_knn:.2f}")
print(f"  Min edge weight            : {edge_weight_knn.min():.4f}")
print(f"  Max edge weight            : {edge_weight_knn.max():.4f}")
print(f"  Mean edge weight           : {edge_weight_knn.mean():.4f}")

#  STEP 3: Method B: Cosine Threshold Graph

section("Step 3 — Method B: Cosine Threshold Graph (threshold=0.90)")

# NOTE: Computing full pairwise cosine similarity is O(n^2).
# For n≈1688 patients and ~20 features, the matrix is 1688×1688 ≈ 11M floats = ~45MB. Fine.
THRESHOLD = 0.90

print(f"  Computing {n_nodes}×{n_nodes} cosine similarity matrix ...")
sim_matrix = cosine_similarity(X)
print(f"  Matrix shape: {sim_matrix.shape}")

# Zero out diagonal and values below threshold
np.fill_diagonal(sim_matrix, 0)
src_thresh, dst_thresh = np.where(sim_matrix >= THRESHOLD)

# Keep upper triangle only to avoid duplicates, then make undirected
mask_upper = src_thresh < dst_thresh
src_thresh = src_thresh[mask_upper]
dst_thresh = dst_thresh[mask_upper]
weights_thresh = sim_matrix[src_thresh, dst_thresh]

edge_index_thresh = torch.tensor(
    np.array([np.concatenate([src_thresh, dst_thresh]),
              np.concatenate([dst_thresh, src_thresh])]), dtype=torch.long)
edge_weight_thresh = torch.tensor(
    np.concatenate([weights_thresh, weights_thresh]), dtype=torch.float)

n_edges_thresh    = edge_index_thresh.shape[1]
avg_degree_thresh = n_edges_thresh / n_nodes

print(f"  Edges (above threshold={THRESHOLD}): {n_edges_thresh}")
print(f"  Average node degree               : {avg_degree_thresh:.2f}")

# Check for isolated nodes (degree 0)
degrees_thresh = torch.zeros(n_nodes, dtype=torch.long)
for node in edge_index_thresh[0]:
    degrees_thresh[node] += 1
n_isolated_thresh = (degrees_thresh == 0).sum().item()
print(f"  Isolated nodes (degree=0)         : {n_isolated_thresh}")

if n_isolated_thresh > n_nodes * 0.05:
    print(f"  [WARNING] >{n_isolated_thresh/n_nodes:.0%} nodes are isolated — "
          f"threshold may be too strict. kNN graph is more robust.")

#  STEP 4: Compare & select best method

section("Step 4 — Comparing methods & selecting graph")

print(f"\n  {'Method':<30} {'Edges':>8} {'Avg Degree':>12} {'Isolated':>10}")
print(f"  {'-'*62}")

degrees_knn = torch.zeros(n_nodes, dtype=torch.long)
for node in edge_index_knn[0]:
    degrees_knn[node] += 1
n_isolated_knn = (degrees_knn == 0).sum().item()

print(f"  {'A: kNN (k=5)':<30} {n_edges_knn:>8} {avg_degree_knn:>12.2f} {n_isolated_knn:>10}")
print(f"  {'B: Cosine thresh (0.90)':<30} {n_edges_thresh:>8} {avg_degree_thresh:>12.2f} {n_isolated_thresh:>10}")

# Select: prefer kNN because it guarantees every node has exactly k neighbours
# (no isolated nodes), making it more robust for GNN message passing.
# We'll include both edge indices in the saved object so you can swap in Phase 4.
print(f"\n  [SELECTED] Method A — kNN graph")
print(f"  Reason: guaranteed connectivity (no isolated nodes),")
print(f"          uniform degree distribution, robust for GNN message passing.")
print(f"  Both edge_index variants are saved in graph_data.pt for ablation.")


#  STEP 5: Build train/val/test masks

section("Step 5 — Stratified train / val / test split (70/15/15)")

from sklearn.model_selection import train_test_split

indices_all = np.arange(n_nodes)

# Stratified split: 70% train, 15% val, 15% test
idx_train, idx_temp = train_test_split(
    indices_all, test_size=0.30, stratify=y, random_state=42)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, stratify=y[idx_temp], random_state=42)

train_mask = torch.zeros(n_nodes, dtype=torch.bool)
val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
test_mask  = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val]     = True
test_mask[idx_test]   = True

print(f"  Train : {train_mask.sum().item():4d} nodes "
      f"(label 0: {int((y[idx_train]==0).sum())}, label 1: {int((y[idx_train]==1).sum())})")
print(f"  Val   : {val_mask.sum().item():4d} nodes "
      f"(label 0: {int((y[idx_val]==0).sum())}, label 1: {int((y[idx_val]==1).sum())})")
print(f"  Test  : {test_mask.sum().item():4d} nodes "
      f"(label 0: {int((y[idx_test]==0).sum())}, label 1: {int((y[idx_test]==1).sum())})")


#  STEP 6: Compute class weights for imbalanced loss

section("Step 6 — Class weights for imbalanced training")

# Weight inversely proportional to class frequency
counts      = np.bincount(y)
class_weights = torch.tensor(
    [n_nodes / (2.0 * c) for c in counts], dtype=torch.float)
print(f"  Class counts : {counts}")
print(f"  Class weights: {class_weights.numpy().round(4)}")
print(f"  (Used in CrossEntropyLoss(weight=...) in Phase 4)")


#  STEP 7: Build & save PyG Data object

section("Step 7 — Building PyTorch Geometric Data object")

data = Data(
    x            = torch.tensor(X, dtype=torch.float),
    edge_index   = edge_index_knn,          # kNN selected as primary
    edge_weight  = edge_weight_knn,
    y            = torch.tensor(y, dtype=torch.long),
    train_mask   = train_mask,
    val_mask     = val_mask,
    test_mask    = test_mask,
    # Also store the threshold graph for ablation experiments
    edge_index_thresh  = edge_index_thresh,
    edge_weight_thresh = edge_weight_thresh,
    class_weights      = class_weights,
    n_features         = torch.tensor(n_features),
    n_nodes            = torch.tensor(n_nodes),
)

torch.save(data, GRAPH_OUT)
print(f"\n  Saved: {GRAPH_OUT}")
print(f"\n  Data object summary:")
print(f"    x            : {data.x.shape}  — node features")
print(f"    edge_index   : {data.edge_index.shape}  — kNN edges")
print(f"    edge_weight  : {data.edge_weight.shape}")
print(f"    y            : {data.y.shape}   — labels")
print(f"    train_mask   : {data.train_mask.sum().item()} True")
print(f"    val_mask     : {data.val_mask.sum().item()} True")
print(f"    test_mask    : {data.test_mask.sum().item()} True")


#  PLOTS

section("Plotting graphs")
sns.set_style("whitegrid")

# ── Plot 1: Edge weight distributions ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(edge_weight_knn.numpy(), bins=40, color='#378add',
             edgecolor='white', alpha=0.85)
axes[0].set_title(f"Method A — kNN (k={K})\nEdge weight distribution", fontsize=11)
axes[0].set_xlabel("Cosine similarity"); axes[0].set_ylabel("Count")
axes[0].axvline(edge_weight_knn.mean().item(), color='#e24b4a',
                linestyle='--', label=f"Mean={edge_weight_knn.mean():.3f}")
axes[0].legend()

axes[1].hist(edge_weight_thresh.numpy(), bins=40, color='#1d9e75',
             edgecolor='white', alpha=0.85)
axes[1].set_title(f"Method B — Cosine Threshold (≥{THRESHOLD})\nEdge weight distribution",
                  fontsize=11)
axes[1].set_xlabel("Cosine similarity"); axes[1].set_ylabel("Count")
axes[1].axvline(edge_weight_thresh.mean().item(), color='#e24b4a',
                linestyle='--', label=f"Mean={edge_weight_thresh.mean():.3f}")
axes[1].legend()

plt.suptitle("Edge Weight Distributions — Method A vs B", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("graph_edge_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] graph_edge_dist.png")

# ── Plot 2: Node degree distributions ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

deg_knn    = degrees_knn.numpy()
deg_thresh = degrees_thresh.numpy()

axes[0].hist(deg_knn, bins=range(deg_knn.min(), deg_knn.max() + 2),
             color='#378add', edgecolor='white', align='left', alpha=0.85)
axes[0].set_title(f"Method A — kNN Degree Distribution\n"
                  f"mean={deg_knn.mean():.1f}, min={deg_knn.min()}, max={deg_knn.max()}", fontsize=11)
axes[0].set_xlabel("Node degree"); axes[0].set_ylabel("Count")

axes[1].hist(deg_thresh, bins=range(deg_thresh.min(), min(deg_thresh.max(), 80) + 2),
             color='#1d9e75', edgecolor='white', align='left', alpha=0.85)
axes[1].set_title(f"Method B — Threshold Degree Distribution\n"
                  f"mean={deg_thresh.mean():.1f}, min={deg_thresh.min()}, max={deg_thresh.max()}", fontsize=11)
axes[1].set_xlabel("Node degree"); axes[1].set_ylabel("Count")

plt.suptitle("Node Degree Distributions — Method A vs B", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("graph_degree_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] graph_degree_dist.png")

section("Phase 3 Complete")
print(f"  Graph saved to : {GRAPH_OUT}")
print(f"  {n_nodes} nodes  |  {n_edges_knn} edges (kNN)  |  "
      f"{n_features} features per node")
print("\nNext step: run  python phase4_models.py")