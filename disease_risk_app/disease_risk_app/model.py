"""
model.py  —  GAT model definition, data preprocessing, graph construction,
             training, and inference utilities.

All the heavy-lifting lives here so app.py stays clean.
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
HD_PATH     = os.path.join(BASE_DIR, "heart_disease_uci.csv")
PIMA_PATH   = os.path.join(BASE_DIR, "diabetes.csv")
MODEL_CACHE = os.path.join(BASE_DIR, "gat_cached.pt")
DATA_CACHE  = os.path.join(BASE_DIR, "graph_cached.pkl")

# ── Feature columns (after preprocessing) ────────────────────────────────────
FEATURE_DISPLAY_NAMES = {
    "age"              : "Age (years)",
    "sex"              : "Sex (1=male, 0=female)",
    "trestbps"         : "Resting blood pressure (mmHg)",
    "chol"             : "Cholesterol (mg/dL)",
    "fbs"              : "Fasting blood sugar >120 (1=yes)",
    "thalch"           : "Max heart rate achieved",
    "exang"            : "Exercise-induced angina (1=yes)",
    "oldpeak"          : "ST depression (oldpeak)",
    "glucose"          : "Glucose (mg/dL)",
    "bmi"              : "BMI",
    "insulin"          : "Insulin (µU/mL)",
    "pregnancies"      : "Number of pregnancies",
    "diabetes_pedigree": "Diabetes pedigree function",
    "skin_thickness"   : "Skin thickness (mm)",
}

# Healthy reference ranges for counterfactual display
HEALTHY_RANGES = {
    "glucose"  : (70, 99),
    "bmi"      : (18.5, 24.9),
    "trestbps" : (90, 120),
    "chol"     : (0, 200),
    "thalch"   : (100, 170),
    "oldpeak"  : (0, 1.0),
    "age"      : (0, 999),
}

# ── GAT model ─────────────────────────────────────────────────────────────────
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden=16, out_channels=2, heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden, heads=heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, out_channels, heads=1,
                             dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    def forward_with_attention(self, x, edge_index):
        """Return logits + attention weights from layer 1."""
        x = F.dropout(x, p=self.dropout, training=False)
        x_out, (edge_idx, attn) = self.conv1(x, edge_index,
                                              return_attention_weights=True)
        x_out = F.elu(x_out)
        x_out = F.dropout(x_out, p=self.dropout, training=False)
        logits = self.conv2(x_out, edge_index)
        return logits, edge_idx, attn


# ── Preprocessing ──────────────────────────────────────────────────────────────
def _preprocess_heart(df):
    hd = df.copy()
    hd["label"] = (hd["num"] > 0).astype(int)
    drop = [c for c in ["id","dataset","num","ca","thal","slope"] if c in hd.columns]
    hd.drop(columns=drop, inplace=True)
    for col in ["fbs","exang"]:
        if col in hd.columns:
            hd[col] = hd[col].map({True:1, False:0, "True":1, "False":0})
    if "sex" in hd.columns:
        hd["sex"] = hd["sex"].map({"Male":1, "Female":0})
    if "cp" in hd.columns:
        cp_dummies = pd.get_dummies(hd["cp"], prefix="cp", dummy_na=False)
        hd = pd.concat([hd.drop(columns=["cp"]), cp_dummies], axis=1)
    if "restecg" in hd.columns:
        re_dummies = pd.get_dummies(hd["restecg"], prefix="restecg", dummy_na=False)
        hd = pd.concat([hd.drop(columns=["restecg"]), re_dummies], axis=1)
    num_cols = [c for c in hd.select_dtypes(include=[np.number]).columns if c != "label"]
    hd[num_cols] = hd[num_cols].fillna(hd[num_cols].median())
    hd["source"] = 0
    return hd


def _preprocess_pima(df):
    p = df.copy()
    p["label"] = p["Outcome"]
    p.drop(columns=["Outcome"], inplace=True)
    for col in ["Glucose","Insulin","BMI","BloodPressure","SkinThickness"]:
        if col in p.columns:
            p[col] = p[col].replace(0, np.nan)
            p[col] = p[col].fillna(p[col].median())
    p.rename(columns={
        "Pregnancies":"pregnancies","Glucose":"glucose",
        "BloodPressure":"trestbps","SkinThickness":"skin_thickness",
        "Insulin":"insulin","BMI":"bmi",
        "DiabetesPedigreeFunction":"diabetes_pedigree","Age":"age"
    }, inplace=True)
    p["source"] = 1
    return p


def build_data(hd_path, pima_path):
    """Full preprocessing + graph construction. Returns (data, scaler, df, feature_cols)."""
    hd   = _preprocess_heart(pd.read_csv(hd_path))
    pima = _preprocess_pima(pd.read_csv(pima_path))

    merged = pd.concat([hd, pima], ignore_index=True, sort=False)
    feature_cols = [c for c in merged.columns if c not in ["label","source"]]

    # Fill cross-dataset NaNs with column median
    for col in feature_cols:
        if merged[col].dtype in [np.float64, np.int64, float, int]:
            merged[col] = merged[col].fillna(merged[col].median())
        else:
            m = merged[col].mode()
            if len(m): merged[col] = merged[col].fillna(m[0])

    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(merged[feature_cols].values).astype(np.float32)
    y = merged["label"].values.astype(np.int64)
    n = len(X)

    # kNN graph (k=5, cosine)
    nbrs = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute").fit(X)
    distances, indices = nbrs.kneighbors(X)
    src, dst, wts = [], [], []
    for i in range(n):
        for j_idx in range(1, 6):
            j = indices[i, j_idx]
            sim = float(1.0 - distances[i, j_idx])
            src += [i, j]; dst += [j, i]; wts += [sim, sim]

    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(wts, dtype=torch.float)

    # Masks
    idx_all = np.arange(n)
    i_tr, i_tmp = train_test_split(idx_all, test_size=0.30, stratify=y, random_state=42)
    i_va, i_te  = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=42)

    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[i_tr] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[i_va]   = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[i_te]  = True

    counts = np.bincount(y)
    class_weights = torch.tensor([n/(2*c) for c in counts], dtype=torch.float)

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        class_weights=class_weights,
        n_features=torch.tensor(X.shape[1]),
    )

    # Store original (unscaled) df alongside for neighbour display
    merged["_row_idx"] = np.arange(n)
    return data, scaler, merged, feature_cols, nbrs, X


# ── Training ───────────────────────────────────────────────────────────────────
def train_gat(data, n_features, epochs=350, patience=40):
    model = GAT(n_features, hidden=16, out_channels=2, heads=8, dropout=0.6)
    opt   = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.5, patience=15, min_lr=1e-5)
    crit  = torch.nn.CrossEntropyLoss(weight=data.class_weights)

    best_f1, best_state, wait = -1, None, 0
    for epoch in range(1, epochs+1):
        model.train(); opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = crit(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(data.x, data.edge_index)
            val_pred = val_out[data.val_mask].argmax(1)
            val_true = data.y[data.val_mask]
            tp = ((val_pred==1)&(val_true==1)).sum().float()
            fp = ((val_pred==1)&(val_true==0)).sum().float()
            fn = ((val_pred==0)&(val_true==1)).sum().float()
            p  = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
            f1 = 2*p*r/(p+r+1e-8)

        sch.step(f1)
        if f1 > best_f1:
            best_f1 = f1; best_state = {k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait += 1
        if wait >= patience:
            break
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}  val_f1={f1:.4f}  best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Training complete. Best val F1={best_f1:.4f}")
    return model


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_patient(raw_values: dict, data, scaler, merged_df,
                    feature_cols, model, nbrs, X_scaled):
    """
    Given a dict of raw clinical values from the form, return a prediction dict.
    raw_values keys = feature_cols entries that have clinical meaning.
    Missing keys are filled with population median.
    """
    # Build raw row in correct column order
    median_vals = merged_df[feature_cols].median().to_dict()
    row = {col: float(raw_values.get(col, median_vals[col])) for col in feature_cols}

    # One-hot: if cp / restecg variants present, fill zeros for others
    row_df = pd.DataFrame([row])[feature_cols].astype(np.float32)
    row_scaled = scaler.transform(row_df.values).astype(np.float32)

    # --- kNN neighbours from training graph ---
    dists, idxs = nbrs.kneighbors(row_scaled, n_neighbors=5)
    neighbours = []
    for rank, (dist, idx) in enumerate(zip(dists[0], idxs[0])):
        nb_row  = merged_df.iloc[idx]
        nb_unscaled = {col: float(merged_df[feature_cols].iloc[idx][col])
                       for col in ["age","trestbps","glucose","bmi","chol"] if col in feature_cols}
        neighbours.append({
            "rank"      : rank + 1,
            "patient_id": int(idx),
            "similarity": round(float(1.0 - dist), 3),
            "label"     : int(merged_df["label"].iloc[idx]),
            "label_str" : "Disease" if merged_df["label"].iloc[idx] == 1 else "Healthy",
            "source"    : "UCI Heart" if merged_df["source"].iloc[idx] == 0 else "Pima Diabetes",
            "key_vals"  : nb_unscaled,
        })

    # --- Inject new node into graph for inference ---
    # Append to feature matrix, add edges to its k neighbours
    n_existing   = X_scaled.shape[0]
    new_idx      = n_existing
    X_aug        = np.vstack([X_scaled, row_scaled])
    new_src      = [n_existing]*5 + list(idxs[0])
    new_dst      = list(idxs[0]) + [n_existing]*5
    aug_edges    = torch.cat([data.edge_index,
                              torch.tensor([new_src, new_dst], dtype=torch.long)], dim=1)
    x_aug_t      = torch.tensor(X_aug, dtype=torch.float)

    with torch.no_grad():
        model.eval()
        logits, edge_idx_ret, attn_weights = model.forward_with_attention(x_aug_t, aug_edges)
        probs = F.softmax(logits[new_idx], dim=0).numpy()

    prob_disease = float(probs[1])
    prob_healthy = float(probs[0])
    prediction   = "High Risk" if prob_disease >= 0.5 else "Low Risk"
    risk_level   = "high" if prob_disease >= 0.5 else "low"

    # --- Attention weights to new node ---
    # Find edges where dst == new_idx
    edge_arr = edge_idx_ret.numpy()
    attn_arr = attn_weights.numpy()  # shape [E, heads]
    attn_mean = attn_arr.mean(axis=1) if attn_arr.ndim == 2 else attn_arr
    mask      = edge_arr[1] == new_idx
    if mask.any():
        src_nodes  = edge_arr[0][mask]
        src_attns  = attn_mean[mask]
        top_order  = np.argsort(src_attns)[::-1][:5]
        attn_info  = [{"patient_id": int(src_nodes[i]),
                       "attention" : round(float(src_attns[i]), 4)}
                      for i in top_order]
    else:
        attn_info = []

    # --- Feature contributions (gradient-based approximation) ---
    x_tensor = torch.tensor(row_scaled, dtype=torch.float, requires_grad=True)
    X_aug2   = torch.cat([data.x, x_tensor], dim=0)
    logits2  = model(X_aug2, aug_edges)
    score    = logits2[new_idx, 1]
    score.backward()
    grads    = x_tensor.grad.numpy()[0]
    contrib  = []
    for i, col in enumerate(feature_cols):
        display = FEATURE_DISPLAY_NAMES.get(col, col)
        raw_val = row[col]
        g       = float(grads[i])
        contrib.append({
            "feature"    : col,
            "display"    : display,
            "value"      : round(raw_val, 2),
            "gradient"   : round(g, 4),
            "abs_grad"   : abs(g),
            "direction"  : "risk" if g > 0 else "protective",
        })
    contrib.sort(key=lambda x: x["abs_grad"], reverse=True)
    contrib = contrib[:8]   # top 8 features

    # Normalize abs_grads to [0,1] for bar widths
    max_g = max(c["abs_grad"] for c in contrib) if contrib else 1
    for c in contrib:
        c["bar_pct"] = round(c["abs_grad"] / max_g * 100, 1)

    # --- Counterfactual (simple gradient descent in feature space) ---
    counterfactual = _compute_counterfactual(
        row_scaled, row, feature_cols, data, aug_edges,
        new_idx, model, prob_disease, scaler)

    return {
        "prediction"   : prediction,
        "risk_level"   : risk_level,
        "prob_disease" : round(prob_disease * 100, 1),
        "prob_healthy" : round(prob_healthy * 100, 1),
        "neighbours"   : neighbours,
        "contributions": contrib,
        "attention"    : attn_info,
        "counterfactual": counterfactual,
        "n_disease_neighbours": sum(1 for nb in neighbours if nb["label"] == 1),
    }


def _compute_counterfactual(row_scaled, row_raw, feature_cols, data,
                             aug_edges, new_idx, model, prob_disease, scaler):
    """
    Gradient-descent counterfactual: find minimum feature change to flip prediction.
    Only moves continuous clinical features (not binary flags or one-hot columns).
    """
    if prob_disease < 0.5:
        return None   # already healthy, no counterfactual needed

    MUTABLE = {"age","trestbps","chol","thalch","oldpeak",
               "glucose","bmi","insulin","skin_thickness","diabetes_pedigree","pregnancies"}
    mutable_idx = [i for i, c in enumerate(feature_cols) if c in MUTABLE]

    cf = torch.tensor(row_scaled.copy(), dtype=torch.float, requires_grad=False)
    cf_var = cf.clone().detach()
    cf_var = cf_var.requires_grad_(True)
    opt = torch.optim.Adam([cf_var], lr=0.05)

    for step in range(300):
        opt.zero_grad()
        X_cf = torch.cat([data.x, cf_var.view(1, -1)], dim=0)
        logits = model(X_cf, aug_edges)
        prob_d = F.softmax(logits[new_idx], dim=0)[1]
        # Loss: push toward healthy + regularise against big changes
        orig_t = torch.tensor(row_scaled[0], dtype=torch.float)
        loss   = prob_d + 0.3 * ((cf_var[0] - orig_t) ** 2).mean()
        loss.backward()
        # Zero out gradient for immutable features
        with torch.no_grad():
            mask = torch.ones_like(cf_var[0])
            for i in range(len(feature_cols)):
                if i not in mutable_idx:
                    mask[i] = 0
            cf_var.grad[0] *= mask
        opt.step()
        with torch.no_grad():
            X_check = torch.cat([data.x, cf_var.view(1, -1)], dim=0)
            p_d = F.softmax(model(X_check, aug_edges)[new_idx], dim=0)[1].item()
            if p_d < 0.45:
                break

    # Unscale counterfactual
    cf_np    = cf_var.detach().numpy()
    cf_unscaled = scaler.inverse_transform(cf_np)[0]
    orig_unscaled= scaler.inverse_transform(row_scaled)[0]

    changes = []
    for i, col in enumerate(feature_cols):
        if col not in MUTABLE: continue
        delta = cf_unscaled[i] - orig_unscaled[i]
        if abs(delta) > 0.5:
            changes.append({
                "feature"  : col,
                "display"  : FEATURE_DISPLAY_NAMES.get(col, col),
                "original" : round(float(orig_unscaled[i]), 1),
                "suggested": round(float(cf_unscaled[i]), 1),
                "delta"    : round(float(delta), 1),
            })
    changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return changes[:4] if changes else []


# ── Cache helpers ──────────────────────────────────────────────────────────────
def load_or_build(hd_path, pima_path):
    """Load cached model+data or rebuild from scratch."""
    if os.path.exists(MODEL_CACHE) and os.path.exists(DATA_CACHE):
        print("Loading from cache...")
        with open(DATA_CACHE, "rb") as f:
            cache = pickle.load(f)
        n_feat  = cache["n_features"]
        model   = GAT(n_feat, 16, 2, 8, 0.6)
        model.load_state_dict(torch.load(MODEL_CACHE, weights_only=True))
        model.eval()
        return model, cache["data"], cache["scaler"], cache["merged_df"], \
               cache["feature_cols"], cache["nbrs"], cache["X_scaled"]

    print("Building graph and training GAT...")
    data, scaler, merged_df, feature_cols, nbrs, X_scaled = \
        build_data(hd_path, pima_path)
    n_feat = int(data.n_features.item())
    model  = train_gat(data, n_feat)

    torch.save(model.state_dict(), MODEL_CACHE)
    with open(DATA_CACHE, "wb") as f:
        pickle.dump({
            "n_features" : n_feat,
            "data"       : data,
            "scaler"     : scaler,
            "merged_df"  : merged_df,
            "feature_cols": feature_cols,
            "nbrs"       : nbrs,
            "X_scaled"   : X_scaled,
        }, f)
    print("Model and data cached.")
    return model, data, scaler, merged_df, feature_cols, nbrs, X_scaled
