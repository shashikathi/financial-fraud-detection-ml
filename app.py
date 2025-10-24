# --- Install required packages if not already ---
# !pip install streamlit plotly shap pandas scikit-learn matplotlib

import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ------------------ Load Data ------------------
df = pd.read_csv(r"E:\ml projects\Fruad\fraud-detection-dashboard\Fraud.csv")

st.set_page_config(page_title="🚨 Fraud Detection Dashboard", layout="wide")
st.title("🚨 Fraud Detection Dashboard")

# ------------------ Sidebar Filters ------------------
st.sidebar.header("Filters")
transaction_type = st.sidebar.multiselect(
    "Transaction Type",
    df['type'].unique(),
    default=df['type'].unique()
)

filtered_df = df[df['type'].isin(transaction_type)]

st.sidebar.markdown(f"**Total Transactions:** {filtered_df.shape[0]}")
st.sidebar.markdown(f"**Total Frauds:** {filtered_df['isFraud'].sum()}")
st.sidebar.markdown(f"**Fraud %:** {round(filtered_df['isFraud'].mean()*100,2)}%")

# ------------------ Section 1: Transaction Distribution ------------------
st.subheader("📊 Transaction Type Distribution (Fraud vs Non-Fraud)")
fig1 = px.histogram(
    filtered_df,
    x='type',
    color='isFraud',
    barmode='group',
    title='Distribution of Transaction Types',
    labels={'isFraud': 'Fraud'},
    color_discrete_map={0: 'blue', 1: 'red'}
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------ Section 2: Fraud % by Transaction Type ------------------
st.subheader("💥 Fraud Percentage by Transaction Type")
fraud_rate = filtered_df.groupby('type')['isFraud'].mean().reset_index()
fraud_rate['isFraud'] *= 100
fig2 = px.bar(
    fraud_rate,
    x='type',
    y='isFraud',
    text='isFraud',
    color='isFraud',
    color_continuous_scale='Reds',
    title='Fraud % by Transaction Type'
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------ Section 3: Model & Feature Importance ------------------
st.subheader("🛠 Model Training & Feature Importance")

# Select features (exclude target)
feature_cols = [col for col in df.columns if col != 'isFraud']
X = df[feature_cols]
y = df['isFraud']

# Simple train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=True)

fig3 = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    color_continuous_scale='Blues',
    title='Feature Importance'
)
st.plotly_chart(fig3, use_container_width=True)

# ------------------ Section 4: SHAP Explanations ------------------
st.subheader("🔍 SHAP Explanations")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# SHAP summary plot
st.markdown("**SHAP Summary Plot**")
plt.figure()
shap.summary_plot(shap_values, X_val, plot_type="dot", show=False)
st.pyplot(plt.gcf())
plt.clf()

# SHAP for single transaction
st.markdown("**SHAP Single Transaction**")
single_index = st.number_input(
    "Choose Transaction Index for Explanation",
    min_value=0, max_value=len(X_val)-1, value=0
)
single = X_val.iloc[[single_index]]
shap_single = explainer.shap_values(single)[0]

plt.figure()
shap.plots.bar(shap.Explanation(values=shap_single, base_values=explainer.expected_value, data=single), show=False)
st.pyplot(plt.gcf())
plt.clf()

# ------------------ Section 5: Download Filtered Data ------------------
st.subheader("💾 Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode()
st.download_button(
    label="Download Filtered CSV",
    data=csv,
    file_name='filtered_transactions.csv',
    mime='text/csv'
)

st.success("✅ Dashboard ready! Interact with filters, hover over plots, explore SHAP explanations, and impress recruiters 😎")
