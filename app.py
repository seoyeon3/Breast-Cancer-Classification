import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# ----------------------------
# Load model
# ----------------------------
model = joblib.load("xgb_breast_cancer.pkl")

st.title("Breast Cancer Prediction App")
st.write("Enter the feature values below to predict whether the tumor is benign or malignant.")
st.caption("For educational purposes only. Not for clinical diagnosis.")

# ----------------------------
# Feature names from sklearn breast cancer dataset
# ----------------------------
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Default values: simple reasonable placeholders
default_values = {
    'mean radius': 14.0,
    'mean texture': 19.0,
    'mean perimeter': 91.0,
    'mean area': 650.0,
    'mean smoothness': 0.10,
    'mean compactness': 0.10,
    'mean concavity': 0.09,
    'mean concave points': 0.05,
    'mean symmetry': 0.18,
    'mean fractal dimension': 0.06,
    'radius error': 0.40,
    'texture error': 1.20,
    'perimeter error': 2.80,
    'area error': 40.0,
    'smoothness error': 0.007,
    'compactness error': 0.025,
    'concavity error': 0.03,
    'concave points error': 0.012,
    'symmetry error': 0.02,
    'fractal dimension error': 0.003,
    'worst radius': 16.0,
    'worst texture': 25.0,
    'worst perimeter': 105.0,
    'worst area': 850.0,
    'worst smoothness': 0.14,
    'worst compactness': 0.25,
    'worst concavity': 0.27,
    'worst concave points': 0.12,
    'worst symmetry': 0.29,
    'worst fractal dimension': 0.08
}

# ----------------------------
# Input form
# ----------------------------
with st.form("prediction_form"):
    st.subheader("Input Features")

    col1, col2 = st.columns(2)
    input_data = {}

    for i, feature in enumerate(feature_names):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            input_data[feature] = st.number_input(
                label=feature,
                value=float(default_values[feature]),
                format="%.6f"
            )

    submitted = st.form_submit_button("Predict")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    input_df = pd.DataFrame([input_data])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    # sklearn breast cancer dataset:
    # 0 = malignant, 1 = benign
    label = "Benign" if pred == 1 else "Malignant"

    st.subheader("Prediction Result")

    if pred == 1:
        st.success(f"Prediction: {label}")
    else:
        st.error(f"Prediction: {label}")

    st.write(f"Probability of Malignant: **{prob[0]:.4f}**")
    st.write(f"Probability of Benign: **{prob[1]:.4f}**")

    st.subheader("Input Summary")
    st.dataframe(input_df)
