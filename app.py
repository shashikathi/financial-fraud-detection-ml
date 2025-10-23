import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components

# -------------------------------------------------------------
# ⚙️ Load Model & Metadata
# -------------------------------------------------------------
MODEL_PATH = "fraud_model_slim.pkl"
META_PATH = "metadata.json"

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        import json
        metadata = json.load(f)
    return model, metadata

model, metadata = load_assets()
FEATURES = metadata.get("features", [])

# -------------------------------------------------------------
# 🧠 Predict Function
# -------------------------------------------------------------
def predict_fraud(df):
    X = df[FEATURES]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    df["Fraud_Probability"] = np.round(probs, 3)
    df["Prediction"] = np.where(preds == 1, "🚨 Fraud", "✅ Legit")
    return df

# -------------------------------------------------------------
# 🎨 Streamlit UI
# -------------------------------------------------------------
st.set_page_config(page_title="💸 Financial Fraud Detection", layout="wide")
st.title("💸 Financial Fraud Detection Dashboard")
st.write("Detect and analyze fraudulent transactions using Machine Learning.")

tabs = st.tabs(["📊 EDA & Visuals", "🔮 Fraud Prediction"])

# -------------------------------------------------------------
# TAB 1 — EDA VISUALS
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("Interactive Dashboards")

    graphs = {
        "Transaction Type Distribution": "Graphs/transaction_types_distribution.html",
        "Fraud Rate by Type": "Graphs/fraud_rate_by_type.html",
        "Transaction Amount (Animated)": "Graphs/transaction_amount_distribution_animated.html",
        "Origin Account Balances (Animated)": "Graphs/origin_account_balances_animated.html",
        "Feature Importance": "Graphs/feature_importance.html"
    }

    for title, path in graphs.items():
        st.markdown(f"### {title}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                components.html(f.read(), height=480)
        except FileNotFoundError:
            st.warning(f"⚠️ Missing file: {path}")

# -------------------------------------------------------------
# TAB 2 — FRAUD PREDICTION
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Upload Transaction CSV to Predict Fraud")
    uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("✅ File uploaded successfully!")

        df_pred = predict_fraud(df.copy())

        fraud_count = (df_pred["Prediction"] == "🚨 Fraud").sum()
        total = len(df_pred)
        ratio = (fraud_count / total) * 100

        st.metric("Total Transactions", f"{total:,}")
        st.metric("Detected Frauds", f"{fraud_count}")
        st.metric("Fraud Ratio", f"{ratio:.2f}%")

        fig = px.histogram(df_pred, x="Fraud_Probability", color="Prediction",
                           nbins=20, title="Fraud Probability Distribution",
                           color_discrete_map={"🚨 Fraud": "red", "✅ Legit": "green"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔍 Top 50 Predictions")
        st.dataframe(df_pred.head(50))

# -------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ by K. Shashi Preetham — Data Analyst & ML Enthusiast")
