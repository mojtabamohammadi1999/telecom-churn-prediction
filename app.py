import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_DIR / "churn_pipeline.joblib")
    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return pipeline, metadata


pipeline, metadata = load_artifacts()
THRESHOLD = metadata["threshold"]
YES_NO = {"Yes": 1, "No": 0}

st.title("Customer Churn Predictor")
st.write("Enter a customer's information to estimate the probability that they will leave.")

st.subheader("Customer profile")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    partner = st.selectbox("Has a partner", ["No", "Yes"])
with col2:
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])

st.subheader("Account and billing")
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0,
                          value=70.0, step=0.5)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"])
paperless = st.selectbox("Paperless billing", ["Yes", "No"])

st.subheader("Services")
phone = st.selectbox("Phone service", ["Yes", "No"])
multiple_lines = "No"
if phone == "Yes":
    multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])

internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
addon_labels = {
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}
addons = {}
for column, label in addon_labels.items():
    if internet == "No":
        addons[column] = "No"
    else:
        addons[column] = st.selectbox(label, ["No", "Yes"], key=column)

if st.button("Predict churn risk", type="primary"):
    row = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": YES_NO[senior],
        "Partner": YES_NO[partner],
        "Dependents": YES_NO[dependents],
        "tenure": tenure,
        "PhoneService": YES_NO[phone],
        "MultipleLines": YES_NO[multiple_lines],
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": YES_NO[paperless],
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
    }
    for column, value in addons.items():
        row[column] = YES_NO[value]

    input_df = pd.DataFrame([row])[metadata["feature_columns"]]
    proba = pipeline.predict_proba(input_df)[0, 1]

    st.divider()
    st.metric("Churn probability", f"{proba:.1%}")
    st.progress(float(proba))

    if proba >= THRESHOLD:
        st.error("High churn risk: consider contacting this customer with a retention offer.")
    else:
        st.success("Low churn risk: no action needed for now.")

    st.caption(
        f"Decision threshold: {THRESHOLD:.2f}. It was chosen to minimize an assumed cost "
        f"({metadata['cost_missed_churn']} per lost customer vs. "
        f"{metadata['cost_false_alarm']} per unnecessary offer). "
        "Total charges are estimated as tenure × monthly charges."
    )

with st.sidebar:
    st.header("About the model")
    st.write("Logistic regression with a full preprocessing pipeline.")
    st.write(f"Test ROC-AUC: **{metadata['test_roc_auc']}**")
    st.write(f"Test PR-AUC: **{metadata['test_pr_auc']}**")