import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

import joblib

# Try to import XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("xgboost not installed; skipping XGBoost model.")

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False
    print("lightgbm not installed; skipping LightGBM model.")


# ==============================
# CONFIG
# ==============================

DATA_PATH = "/Users/yomnadean/Desktop/Research/septic_wrist_v2/RK_AI.xlsx"
FINAL_FEATURES_PATH = "final_features.csv"  # from your previous script
TARGET = "septic"

CV_SPLITS = 5
RANDOM_STATE = 42


# ==============================
# LOAD DATA & FEATURES
# ==============================

print("\nLoading data and feature list...")

df = pd.read_excel(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Outcome column '{TARGET}' not found in data!")

# 1) Clean septic outcome
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
n_before = len(df)
df = df[~df[TARGET].isna()].copy()
df[TARGET] = df[TARGET].astype(int)
n_after = len(df)
print(f"Dropped {n_before - n_after} rows with missing septic outcome.")

# 2) Load final features (locks feature set used for comparison)
if not Path(FINAL_FEATURES_PATH).exists():
    raise FileNotFoundError(
        f"'{FINAL_FEATURES_PATH}' not found. "
        "Run your previous feature-selection script first."
    )

features = pd.read_csv(FINAL_FEATURES_PATH)["feature"].tolist()
print("\nFinal model features:", features)

# Ensure all features exist in the dataframe
missing_feats = [f for f in features if f not in df.columns]
if missing_feats:
    raise ValueError(f"The following features are missing from the data: {missing_feats}")

# 3) Complete-case restriction for predictors
X_full = df[features].copy()
y_full = df[TARGET].copy()

complete_mask = ~X_full.isna().any(axis=1)
X = X_full[complete_mask].copy()
y = y_full[complete_mask].copy()

print(f"\nRows with complete data for all model features: {len(X)} / {len(df)}")

# Outcome imbalance info
n_pos = int(y.sum())
n_neg = int(len(y) - n_pos)
print(f"Septic events: {n_pos}, Non-septic: {n_neg}")

if n_pos == 0 or n_neg == 0:
    raise ValueError("Need both septic and non-septic cases for modeling.")

scale_pos_weight = n_neg / n_pos  # for XGBoost

# All features are numeric/binary; treat them as numeric
numeric_features = features

# Preprocessor for models that need scaling (logistic)
preprocess_standardize = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ],
    remainder="drop"
)

# For tree-based models, scaling is not needed
preprocess_passthrough = "passthrough"


# ==============================
# DEFINE MODELS
# ==============================

models = {}

# 1) Logistic regression (L1)
models["logistic_l1"] = Pipeline(steps=[
    ("prep", preprocess_standardize),
    ("clf", LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        max_iter=500,
        random_state=RANDOM_STATE
    ))
])

# 2) Random Forest
models["random_forest"] = Pipeline(steps=[
    ("prep", preprocess_passthrough),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])

# 3) ExtraTrees
models["extra_trees"] = Pipeline(steps=[
    ("prep", preprocess_passthrough),
    ("clf", ExtraTreesClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])

# 4) XGBoost (if available)
if HAVE_XGB:
    models["xgboost"] = Pipeline(steps=[
        ("prep", preprocess_passthrough),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

# 5) LightGBM (if available)
if HAVE_LGBM:
    models["lightgbm"] = Pipeline(steps=[
        ("prep", preprocess_passthrough),
        ("clf", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

print("\nModels to be evaluated:")
for name in models.keys():
    print(" -", name)


# ==============================
# CROSS-VALIDATION COMPARISON
# ==============================

cv = StratifiedKFold(
    n_splits=CV_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []

for model_name, model in models.items():
    print(f"\n=== Evaluating model: {model_name} ===")
    fold_aucs = []
    fold_pr_aucs = []
    fold_briers = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        fold_aucs.append(auc)
        fold_pr_aucs.append(pr_auc)
        fold_briers.append(brier)

        print(f"Fold {fold_idx}: ROC AUC={auc:.3f}, PR AUC={pr_auc:.3f}, Brier={brier:.3f}")

    res = {
        "model": model_name,
        "mean_roc_auc": np.mean(fold_aucs),
        "std_roc_auc": np.std(fold_aucs),
        "mean_pr_auc": np.mean(fold_pr_aucs),
        "std_pr_auc": np.std(fold_pr_aucs),
        "mean_brier": np.mean(fold_briers),
        "std_brier": np.std(fold_briers),
    }
    results.append(res)

# ==============================
# SAVE AND DISPLAY RESULTS
# ==============================

results_df = pd.DataFrame(results).sort_values(
    by="mean_roc_auc", ascending=False
)

print("\n=== Model Comparison (sorted by mean ROC AUC) ===")
print(results_df)

results_df.to_csv("model_comparison.csv", index=False)
print("\nSaved model comparison table to 'model_comparison.csv'.")

# ==============================
# SELECT BEST MODEL & TRAIN ON FULL DATA
# ==============================

best_model_name = results_df.iloc[0]["model"]
print(f"\nBest model by mean ROC AUC: {best_model_name}")

best_model = models[best_model_name]

# Fit best model on all complete-case data
best_model.fit(X, y)
joblib.dump(best_model, "septic_wrist_best_model.joblib")

print("\n✅ Saved best model pipeline as 'septic_wrist_best_model.joblib'")
print("   (Features used:", features, ")")
