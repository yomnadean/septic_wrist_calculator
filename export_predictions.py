import numpy as np
import pandas as pd
import joblib

DATA_PATH = "/Users/yomnadean/Desktop/Research/septic_wrist_v2/RK_AI.xlsx"
MODEL_PATH = "septic_wrist_model.joblib"
FEATURES_PATH = "final_features.csv"   # created by your training script

# ==============================
# 1. Load data & model
# ==============================

print("Loading data and model...")

df = pd.read_excel(DATA_PATH)
pipe = joblib.load(MODEL_PATH)

# Load the final feature list used in the model
features = pd.read_csv(FEATURES_PATH)["feature"].tolist()
print("\nFinal model features:", features)

# ==============================
# 2. Clean septic outcome (for info; not needed for prediction)
# ==============================

if "septic" in df.columns:
    df["septic"] = pd.to_numeric(df["septic"], errors="coerce")
    n_before = len(df)
    # We won't drop rows based on septic here; we can keep them even if missing
    print(f"\nSeptic column present. Non-missing septic rows: "
          f"{df['septic'].notna().sum()} / {n_before}")
else:
    print("\nNo 'septic' column found; proceeding without outcome.")

# ==============================
# 3. Prepare features & handle missingness
# ==============================

# Ensure all required feature columns exist
missing_feats = [f for f in features if f not in df.columns]
if missing_feats:
    raise ValueError(f"The following model features are missing in the data: {missing_feats}")

X = df[features].copy()

# Identify rows where ALL model features are non-missing
complete_mask = ~X.isna().any(axis=1)
n_complete = complete_mask.sum()
n_total = len(df)

print(f"\nRows with complete data for all model features: {n_complete} / {n_total}")

# ==============================
# 4. Predict only on complete rows
# ==============================

# Initialize output column with NaN
df["septic_prob"] = np.nan

if n_complete > 0:
    X_complete = X[complete_mask]
    probs = pipe.predict_proba(X_complete)[:, 1]
    df.loc[complete_mask, "septic_prob"] = probs
    print(f"Predicted septic probabilities for {n_complete} patients.")
else:
    print("No rows with complete feature data; no predictions made.")

# ==============================
# 5. Save results
# ==============================

OUTPUT_PATH = "septic_wrist_with_predictions.xlsx"
df.to_excel(OUTPUT_PATH, index=False)
print(f"\n✅ Saved predictions to '{OUTPUT_PATH}'")
