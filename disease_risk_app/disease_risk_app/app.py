"""
app.py  —  Flask application for Patient Disease Risk Prediction
           with GAT-based explainability.

Run:
    python app.py

Then open:  http://127.0.0.1:5000
"""

import os, json
from flask import Flask, render_template, request, jsonify
from model import load_or_build, predict_patient, HD_PATH, PIMA_PATH, FEATURE_DISPLAY_NAMES

app = Flask(__name__)

# ── Global state (loaded once at startup) ─────────────────────────────────────
print("\n" + "="*55)
print("  Disease Risk Predictor — loading model...")
print("="*55)

if not os.path.exists(HD_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] heart_disease_uci.csv not found at {HD_PATH}\n"
        "Place it in the same folder as app.py and restart."
    )
if not os.path.exists(PIMA_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] diabetes.csv not found at {PIMA_PATH}\n"
        "Download from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
        "Place it in the same folder as app.py and restart."
    )

MODEL, DATA, SCALER, MERGED_DF, FEATURE_COLS, NBRS, X_SCALED = \
    load_or_build(HD_PATH, PIMA_PATH)

# Stats for the dashboard header
TOTAL_PATIENTS = len(MERGED_DF)
N_DISEASE      = int((MERGED_DF["label"] == 1).sum())
N_HEALTHY      = int((MERGED_DF["label"] == 0).sum())
N_FEATURES     = len(FEATURE_COLS)

print(f"\n  Ready!  {TOTAL_PATIENTS} patients · {N_FEATURES} features · http://127.0.0.1:5000\n")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Main UI — patient input form."""
    return render_template("index.html",
        total_patients=TOTAL_PATIENTS,
        n_disease=N_DISEASE,
        n_healthy=N_HEALTHY,
        n_features=N_FEATURES,
        feature_names=FEATURE_DISPLAY_NAMES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form values, run GAT prediction, return JSON."""
    form = request.get_json()

    # Parse and type-cast inputs
    raw = {}
    float_fields = [
        "age","trestbps","chol","thalch","oldpeak",
        "glucose","bmi","insulin","skin_thickness",
        "diabetes_pedigree","pregnancies",
    ]
    binary_fields = ["sex","fbs","exang"]
    for f in float_fields:
        if f in form and form[f] != "":
            try: raw[f] = float(form[f])
            except: pass
    for f in binary_fields:
        if f in form and form[f] != "":
            try: raw[f] = float(form[f])
            except: pass

    result = predict_patient(
        raw_values=raw,
        data=DATA,
        scaler=SCALER,
        merged_df=MERGED_DF,
        feature_cols=FEATURE_COLS,
        model=MODEL,
        nbrs=NBRS,
        X_scaled=X_SCALED,
    )
    return jsonify(result)


@app.route("/population_stats")
def population_stats():
    """Return class balance and feature stats for the dashboard."""
    stats = {
        "total": TOTAL_PATIENTS,
        "disease": N_DISEASE,
        "healthy": N_HEALTHY,
        "disease_pct": round(N_DISEASE / TOTAL_PATIENTS * 100, 1),
        "healthy_pct": round(N_HEALTHY / TOTAL_PATIENTS * 100, 1),
    }
    return jsonify(stats)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
