import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# SHAP is optional but recommended
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Septic Wrist Risk Calculator",
    page_icon="🦴",
    layout="centered"
)

# ==============================
# CONSTANTS
# ==============================

LOGISTIC_MODEL_PATH = "septic_wrist_model.joblib"          # L1 logistic
BEST_MODEL_PATH = "septic_wrist_best_model.joblib"         # Random forest (best CV AUC)

FEATURES = [
    "blood_culture",
    "crystals",
    "multiple_joints",
    "age",
    "ivda",
    "hx_genital_infection",
    "hx__septic_arthritis",
    "symptom_duration",
    "hx_crystalline_arthropathy",
]


# ==============================
# HELPERS
# ==============================

@st.cache_resource
def load_models():
    models = {}
    if Path(LOGISTIC_MODEL_PATH).exists():
        models["Logistic regression (L1)"] = joblib.load(LOGISTIC_MODEL_PATH)
    if Path(BEST_MODEL_PATH).exists():
        models["Random forest (best CV AUC)"] = joblib.load(BEST_MODEL_PATH)

    if not models:
        raise FileNotFoundError(
            "No model files found. Make sure 'septic_wrist_model.joblib' and/or "
            "'septic_wrist_best_model.joblib' are in the same folder as app.py."
        )

    return models


def yn_to_int(x: str) -> int:
    return 1 if x == "Yes" else 0


def build_input_row(
    age,
    multiple_joints,
    blood_culture,
    crystals,
    symptom_duration,
    ivda,
    hx_genital_infection,
    hx_septic_arthritis,
    hx_crystalline_arthropathy,
) -> pd.DataFrame:
    row = pd.DataFrame([{
        "blood_culture": yn_to_int(blood_culture),
        "crystals": yn_to_int(crystals),
        "multiple_joints": yn_to_int(multiple_joints),
        "age": age,
        "ivda": yn_to_int(ivda),
        "hx_genital_infection": yn_to_int(hx_genital_infection),
        "hx__septic_arthritis": yn_to_int(hx_septic_arthritis),
        "symptom_duration": symptom_duration,
        "hx_crystalline_arthropathy": yn_to_int(hx_crystalline_arthropathy),
    }])
    # Ensure consistent feature order
    return row[FEATURES]


@st.cache_resource
def get_shap_explainer(model_name: str, model_pipeline):
    """
    Build a SHAP explainer for the selected model.
    For simplicity, we use:
      - TreeExplainer for tree-based models (RandomForest, ExtraTrees, XGB, etc.)
      - LinearExplainer for LogisticRegression
    We treat all features as numeric; pipeline handles any preprocessing.
    """
    if not HAVE_SHAP:
        return None

    # model_pipeline is a sklearn Pipeline(prep, clf) or just an estimator
    clf = None
    prep = None

    if hasattr(model_pipeline, "named_steps"):
        # It's a Pipeline
        steps = model_pipeline.named_steps
        clf = steps.get("clf", model_pipeline)
        prep = steps.get("prep", None)
    else:
        # Just a model
        clf = model_pipeline
        prep = None

    # For background, create a "neutral" reference patient
    # All binaries = 0, age ~ 50, symptom_duration ~ 3 days
    bg_row = pd.DataFrame([{
        "blood_culture": 0,
        "crystals": 0,
        "multiple_joints": 0,
        "age": 50,
        "ivda": 0,
        "hx_genital_infection": 0,
        "hx__septic_arthritis": 0,
        "symptom_duration": 3.0,
        "hx_crystalline_arthropathy": 0,
    }])[FEATURES]

    if prep is not None and prep != "passthrough":
        bg_trans = prep.transform(bg_row)
    else:
        bg_trans = bg_row.values

    # Tree-based models
    if isinstance(clf, (RandomForestClassifier, ExtraTreesClassifier)):
        explainer = shap.TreeExplainer(clf)
        # we will pass transformed data each time
        return explainer, prep

    # Logistic regression -> LinearExplainer
    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, bg_trans)
        return explainer, prep

    # Fallback: general SHAP explainer
    explainer = shap.Explainer(lambda x: model_pipeline.predict_proba(x)[:, 1], bg_row)
    return explainer, None


def compute_shap_for_row(model_name: str, model_pipeline, input_row: pd.DataFrame):
    """
    Compute SHAP values for a single input_row (1 x n_features).
    Returns a DataFrame with columns: feature, shap_value, abs_shap
    """
    if not HAVE_SHAP:
        return None

    explainer, prep = get_shap_explainer(model_name, model_pipeline)
    if explainer is None:
        return None

    # transform input if needed
    if prep is not None and prep != "passthrough":
        X_row = prep.transform(input_row)
    else:
        X_row = input_row.values

    shap_values = explainer(X_row)
    # For some explainers, shap_values is a numpy array; for newer versions it's an object
    if hasattr(shap_values, "values"):
        sv = shap_values.values[0]
    else:
        sv = shap_values[0]

    df_shap = pd.DataFrame({
        "feature": FEATURES,
        "shap_value": sv,
    })
    df_shap["abs_shap"] = df_shap["shap_value"].abs()
    df_shap = df_shap.sort_values("abs_shap", ascending=False)

    return df_shap


# ==============================
# LOAD MODELS
# ==============================

models = load_models()

# ==============================
# HEADER WITH LAB BRANDING
# ==============================

logo_path = Path("bassiri_lab_logo.png")  # add your lab logo as this file

cols = st.columns([1, 3])

with cols[0]:
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.write("")

with cols[1]:
    st.markdown(
        """
        ### Septic Wrist Risk Calculator  
        **Dr. Bahar Bassiri Gharb's Microsurgery & Perfusion Lab**  
        Cleveland Clinic
        """.strip()
    )

st.markdown("---")

# ==============================
# SIDEBAR: MODEL SELECTION
# ==============================

st.sidebar.header("Model Selection")

model_name = st.sidebar.radio(
    "Choose prediction model:",
    list(models.keys()),
    index=0
)

selected_model = models[model_name]

st.sidebar.info(
    "💡 Logistic regression provides more interpretable coefficients and slightly "
    "better calibration. Random forest can capture nonlinear effects and interactions."
)

if HAVE_SHAP:
    st.sidebar.success("SHAP explanations enabled.")
else:
    st.sidebar.warning("SHAP not installed. Install 'shap' to enable explanations.")

# ==============================
# MAIN FORM: INPUTS
# ==============================

st.markdown("#### Patient & Clinical Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=50, step=1)

    multiple_joints = st.selectbox(
        "Multiple joints involved?",
        ["No", "Yes"]
    )

    blood_culture = st.selectbox(
        "Positive blood culture?",
        ["No", "Yes"]
    )

    crystals = st.selectbox(
        "Crystals in synovial fluid?",
        ["No", "Yes"]
    )

with col2:
    symptom_duration = st.number_input(
        "Symptom duration (days)",
        min_value=0.0,
        max_value=60.0,
        value=2.0,
        step=0.5
    )

    ivda = st.selectbox(
        "IV drug use?",
        ["No", "Yes"]
    )

    hx_genital_infection = st.selectbox(
        "History of genital infection?",
        ["No", "Yes"]
    )

    hx_septic_arthritis = st.selectbox(
        "History of septic arthritis?",
        ["No", "Yes"]
    )

    hx_crystalline_arthropathy = st.selectbox(
        "History of crystalline arthropathy?",
        ["No", "Yes"]
    )

input_row = build_input_row(
    age,
    multiple_joints,
    blood_culture,
    crystals,
    symptom_duration,
    ivda,
    hx_genital_infection,
    hx_septic_arthritis,
    hx_crystalline_arthropathy,
)

st.markdown("---")

# ==============================
# PREDICT BUTTON
# ==============================

if st.button("Calculate Septic Wrist Risk"):
    try:
        prob = selected_model.predict_proba(input_row)[0, 1]
    except Exception as e:
        st.error(f"Error while predicting: {e}")
    else:
        risk_pct = prob * 100

        if risk_pct < 10:
            risk_cat = "Low"
            color = "🟢"
        elif risk_pct < 30:
            risk_cat = "Intermediate"
            color = "🟡"
        else:
            risk_cat = "High"
            color = "🔴"

        st.subheader("Estimated Risk")
        st.metric(
            label=f"{model_name}",
            value=f"{risk_pct:.1f} %",
            delta=None
        )

        st.markdown(
            f"**Risk category:** {color} **{risk_cat}**  "
            "(thresholds can be refined based on clinical judgment or decision-curve analysis)."
        )

        with st.expander("Show input summary"):
            st.write(input_row)

        # ==============================
        # SHAP EXPLANATION SECTION
        # ==============================
        if HAVE_SHAP:
            st.markdown("#### SHAP Explanation (Feature Contributions)")

            shap_df = compute_shap_for_row(model_name, selected_model, input_row)
            if shap_df is None:
                st.info("Unable to compute SHAP values for this model.")
            else:
                st.write(
                    "The plot below shows how each feature influenced this prediction "
                    "(positive values push the risk up, negative values push it down)."
                )

                # Show table
                st.dataframe(
                    shap_df[["feature", "shap_value"]],
                    use_container_width=True
                )

                # Simple bar chart of SHAP values
                st.bar_chart(
                    shap_df.set_index("feature")["shap_value"],
                    use_container_width=True
                )
        else:
            st.info("Install the 'shap' package to see feature-level explanations.")

st.markdown("---")

st.caption(
    "This tool is intended for research and educational purposes only and is not a substitute "
    "for clinical judgment or institutional guidelines."
)
