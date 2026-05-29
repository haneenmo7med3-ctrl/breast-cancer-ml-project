import streamlit as st
import pickle
import numpy as np
import os

# ── Load saved model artifacts ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

model   = pickle.load(open(os.path.join(BASE, "models/best_model.pkl"),        "rb"))
scaler  = pickle.load(open(os.path.join(BASE, "models/scaler.pkl"),             "rb"))
features = pickle.load(open(os.path.join(BASE, "models/selected_features.pkl"), "rb"))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Breast Cancer Prediction System")
st.markdown("**Biotech ML Final Project** — Haneen Mohamed Ismail | ID: 221000582")
st.divider()

st.subheader("Enter Patient Clinical Measurements")
st.caption("Fill in all fields and click Predict. All measurements come from Fine Needle Aspirate (FNA) analysis.")

# ── Input form — one field per selected feature ──────────────────────────────
col1, col2 = st.columns(2)

inputs = {}

field_config = {
    "radius_mean":            ("Radius Mean",             col1, 0.0,  30.0,  14.0),
    "texture_mean":           ("Texture Mean",            col2, 0.0,  40.0,  19.0),
    "area_mean":              ("Area Mean",               col1, 0.0, 2500.0, 654.0),
    "smoothness_mean":        ("Smoothness Mean",         col2, 0.0,  0.2,   0.096),
    "compactness_mean":       ("Compactness Mean",        col1, 0.0,  0.4,   0.104),
    "concavity_mean":         ("Concavity Mean",          col2, 0.0,  0.5,   0.089),
    "concavity_se":           ("Concavity SE",            col1, 0.0,  0.2,   0.030),
    "fractal_dimension_mean": ("Fractal Dimension Mean",  col2, 0.0,  0.1,   0.063),
    "age":                    ("Patient Age",             col1, 18,   100,   55),
    "age_group_encoded":      ("Age Group (0=<40, 1=40-59, 2=60+)", col2, 0, 2, 1),
    "compactness_ratio":      ("Compactness Ratio\n(compactness / fractal_dim)", col1, 0.0, 10.0, 1.6),
    "area_radius_ratio":      ("Area / Radius Ratio",    col2, 0.0, 200.0, 46.0),
}

for feat, (label, col, mn, mx, default) in field_config.items():
    with col:
        if feat == "age" or feat == "age_group_encoded":
            inputs[feat] = float(col.number_input(label, min_value=int(mn), max_value=int(mx), value=int(default)))
        else:
            inputs[feat] = col.number_input(label, min_value=float(mn), max_value=float(mx), value=float(default), format="%.4f")

st.divider()

# ── Predict button ───────────────────────────────────────────────────────────
if st.button("🔍 Predict Diagnosis", use_container_width=True, type="primary"):

    # Build input array in the exact feature order the model expects
    X = np.array([[inputs[f] for f in features]])

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Malignant Tumor Detected")
        prob = probability[1]
    else:
        st.success("✅ Benign Tumor")
        prob = probability[0]

    st.metric("Model Confidence", f"{prob * 100:.1f}%")
    st.progress(float(prob))

    st.caption("⚠️ This tool is for educational purposes only. Always consult a qualified medical professional.")