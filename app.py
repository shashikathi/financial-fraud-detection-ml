import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Fraud Detection Dashboard")

# ------------------ Load Data in Chunks ------------------
@st.cache_data
def load_filtered_data(path, chunksize=10000, types=None):
    if not os.path.exists(path):
        st.error(f"CSV file not found at {path}. Please check the file path.")
        st.stop()

    filtered_chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if types:
            filtered_chunk = chunk[chunk['type'].isin(types)]
        else:
            filtered_chunk = chunk
        filtered_chunks.append(filtered_chunk)
    return pd.concat(filtered_chunks, ignore_index=True)

data_path = "Fraud.csv"  # Keep CSV in same folder for deployment
transaction_type_filter = st.sidebar.multiselect(
    "Transaction Type Filter (initial, optional)",
    options=["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN"],  # you can adjust dynamically later
    default=None
)

df = load_filtered_data(data_path, types=transaction_type_filter if transaction_type_filter else None)

st.sidebar.markdown(f"Total Transactions: {df.shape[0]}")
st.sidebar.markdown(f"Total Frauds: {df['isFraud'].sum()}")
st.sidebar.markdown(f"Fraud %: {round(df['isFraud'].mean()*100,2)}%")

# ------------------ Transaction Distribution ------------------
st.subheader("Transaction Type Distribution (Fraud vs Non-Fraud)")
fig1 = px.histogram(
    df,
    x='type',
    color='isFraud',
    barmode='group',
    labels={'isFraud':'Fraud'},
    color_discrete_map={0:'blue',1:'red'}
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------ Fraud % by Transaction Type ------------------
st.subheader("Fraud Percentage by Transaction Type")
fraud_rate = df.groupby('type')['isFraud'].mean().reset_index()
fraud_rate['isFraud'] *= 100
fig2 = px.bar(fraud_rate, x='type', y='isFraud', text='isFraud', color='isFraud', color_continuous_scale='Reds')
st.plotly_chart(fig2, use_container_width=True)

# ------------------ Model Training ------------------
st.subheader("Model Training & Feature Importance")

# Use numeric features only
feature_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
if 'isFraud' in feature_cols:
    feature_cols.remove('isFraud')

# Optional: sample for huge datasets
SAMPLE_SIZE = 50000  # adjust depending on memory
if len(df) > SAMPLE_SIZE:
    df_sample = df.sample(SAMPLE_SIZE, random_state=42)
else:
    df_sample = df.copy()

X = df_sample[feature_cols]
y = df_sample['isFraud']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=True)
fig3 = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
st.plotly_chart(fig3, use_container_width=True)

# ------------------ SHAP Explanations (Lazy) ------------------
st.subheader("SHAP Explanations")
explainer = shap.TreeExplainer(model)

shap_option = st.selectbox("Choose SHAP Visualization", ["Summary Plot", "Single Transaction"])
if shap_option == "Summary Plot":
    # Optional: use sample for SHAP summary to save memory
    shap_sample = X_val.sample(min(5000, len(X_val)), random_state=42)
    shap_values = explainer.shap_values(shap_sample)
    plt.figure()
    shap.summary_plot(shap_values, shap_sample, plot_type="dot", show=False)
    st.pyplot(plt.gcf())
    plt.clf()
else:
    single_index = st.number_input("Choose Transaction Index", min_value=0, max_value=len(X_val)-1, value=0)
    single = X_val.iloc[[single_index]]
    shap_single = explainer.shap_values(single)[0]
    plt.figure()
    shap.plots.bar(shap.Explanation(values=shap_single, base_values=explainer.expected_value, data=single), show=False)
    st.pyplot(plt.gcf())
    plt.clf()

# ------------------ Download Filtered Data ------------------
st.subheader("Download Filtered Data")
csv = df.to_csv(index=False).encode()
st.download_button("Download Filtered CSV", data=csv, file_name="filtered_transactions.csv", mime="text/csv")

st.success("Dashboard ready. Use filters and explore SHAP explanations.")
