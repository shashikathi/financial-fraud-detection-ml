# --- Install required packages if not already ---
# !pip install streamlit plotly shap pandas scikit-learn matplotlib pyarrow

import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Fraud Detection Dashboard")

# ------------------ File Upload ------------------
st.subheader("Upload Your CSV File")
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_filtered_data(file, chunksize=10000):
        filtered_chunks = []
        for chunk in pd.read_csv(file, chunksize=chunksize):
            filtered_chunks.append(chunk)
        return pd.concat(filtered_chunks, ignore_index=True)

    df = load_filtered_data(uploaded_file)

    # ------------------ Sidebar Filters ------------------
    st.sidebar.header("Filters")
    transaction_type_filter = st.sidebar.multiselect(
        "Transaction Type Filter",
        options=df['type'].unique(),
        default=df['type'].unique()
    )

    filtered_df = df[df['type'].isin(transaction_type_filter)]

    st.sidebar.markdown(f"Total Transactions: {filtered_df.shape[0]}")
    st.sidebar.markdown(f"Total Frauds: {filtered_df['isFraud'].sum()}")
    st.sidebar.markdown(f"Fraud %: {round(filtered_df['isFraud'].mean()*100,2)}%")

    # ------------------ Transaction Distribution ------------------
    st.subheader("Transaction Type Distribution (Fraud vs Non-Fraud)")
    fig1 = px.histogram(
        filtered_df,
        x='type',
        color='isFraud',
        barmode='group',
        labels={'isFraud':'Fraud'},
        color_discrete_map={0:'blue',1:'red'}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ------------------ Fraud % by Transaction Type ------------------
    st.subheader("Fraud Percentage by Transaction Type")
    fraud_rate = filtered_df.groupby('type')['isFraud'].mean().reset_index()
    fraud_rate['isFraud'] *= 100
    fig2 = px.bar(fraud_rate, x='type', y='isFraud', text='isFraud', color='isFraud', color_continuous_scale='Reds')
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------ Model Training ------------------
    st.subheader("Model Training & Feature Importance")
    feature_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if 'isFraud' in feature_cols:
        feature_cols.remove('isFraud')

    SAMPLE_SIZE = 50000
    if len(filtered_df) > SAMPLE_SIZE:
        df_sample = filtered_df.sample(SAMPLE_SIZE, random_state=42)
    else:
        df_sample = filtered_df.copy()

    X = df_sample[feature_cols]
    y = df_sample['isFraud']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=True)
    fig3 = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig3, use_container_width=True)

    # ------------------ SHAP Explanations ------------------
    st.subheader("SHAP Explanations")
    explainer = shap.TreeExplainer(model)

    shap_option = st.selectbox("Choose SHAP Visualization", ["Summary Plot", "Single Transaction"])
    if shap_option == "Summary Plot":
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
    csv = filtered_df.to_csv(index=False).encode()
    st.download_button("Download Filtered CSV", data=csv, file_name="filtered_transactions.csv", mime="text/csv")

    st.success("Dashboard ready. Use filters and explore SHAP explanations.")
else:
    st.info("Upload a CSV file to start the dashboard.")
