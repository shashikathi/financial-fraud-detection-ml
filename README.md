
# 💸 Financial Fraud Detection System  

> _“Every second, a transaction happens. Some are fake — we catch those.”_

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-forestgreen?logo=lightgbm)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-lightblue?logo=plotly)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-purple?logo=pandas)
![Status](https://img.shields.io/badge/Project-Production%20Ready-brightgreen?style=flat-square)

---

## 🔍 TL;DR
End-to-end **Machine Learning pipeline** for detecting fraudulent financial transactions.  
Features real-world data, a **LightGBM model**, and fully **interactive Plotly dashboards** for deep visual insights.  

> Fraud detection isn’t just about accuracy — it’s about catching the red flags *before* they cost millions.

---

## ⚡ Highlights

- 🧹 **Data preprocessing** and feature engineering tailored for financial data  
- 🧮 **LightGBM classifier** tuned for precision–recall balance  
- 🔍 **Explainable AI** with feature importance and model insights  
- 💹 **Interactive dashboards** built with Plotly (animated and dynamic)  
- 📊 **Performance:** 98.4% Accuracy · 94.7% Recall · 0.992 ROC-AUC  
- 🌐 Ready for **Streamlit** or **HuggingFace** deployment  

---

## 🖼️ Dashboard Gallery  

### 📊 Transaction Type Distribution  
Interactive breakdown of transaction categories (CASH_IN, CASH_OUT, TRANSFER, PAYMENT).  
👉 **[`View Chart`](Graphs/transaction_types_distribution.html)**  

### 💰 Transaction Amount Distribution (Animated)  
Animated Plotly visualization showing transaction amounts over time.  
👉 **[`View Chart`](Graphs/transaction_amount_distribution_animated.html)**  

### 🧾 Origin Account Balances (Animated)  
Visualizes before-and-after balance changes for origin accounts.  
👉 **[`View Chart`](Graphs/origin_account_balances_animated.html)**  

### ⚖️ Fraud Rate by Transaction Type  
Highlights the fraud percentage per transaction category.  
👉 **[`View Chart`](Graphs/fraud_rate_by_type.html)**  

### 🧠 Feature Importance (Explainable AI)  
Shows how each feature contributes to fraud detection decisions.  
👉 **[`View Chart`](Graphs/feature_importance.html)**  

---

## 🗂️ Folder Structure

| Folder / File | Description |
|----------------|-------------|
| `Fraud_Detection_with_Machine_Learning.ipynb` | Complete ML pipeline and model training |
| `Graphs/` | Interactive Plotly HTML dashboards |
| `data/transactions.csv` | Sample dataset |
| `app.py` | (Optional) Streamlit web app |
| `requirements.txt` | Library dependencies |
| `README.md` | You’re reading it 😉 |

---

## 🧠 Model Workflow

1️⃣ **Data Cleaning & Preprocessing** — handled nulls, encoded transaction types, normalized amounts  
2️⃣ **Feature Engineering** — created balance deltas and time-based features  
3️⃣ **Model Training** — optimized LightGBM for recall and precision  
4️⃣ **Evaluation** — analyzed results via accuracy, recall, precision, and ROC-AUC  
5️⃣ **Explainability** — feature importance visualization for transparency  

---

## 📈 Performance Snapshot

| Metric | Score |
|---------|--------|
| Accuracy | 🟢 **98.4%** |
| Recall (Fraud Class) | 🟡 **94.7%** |
| Precision | 🟢 **95.3%** |
| ROC-AUC | 🔵 **0.992** |

> 🎯 Tuned for *high recall* — because missing a fraud costs more than a false alert.

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/shashikathi/fraud-detection-system.git
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Fraud_Detection_with_Machine_Learning.ipynb
```

(Optional — if you build a Streamlit app)
```bash
streamlit run app.py
```

---

## 🧩 Tech Stack

| Category | Tools |
|-----------|-------|
| ML & Data | Python, Pandas, NumPy, Scikit-learn, LightGBM |
| Visualization | Plotly, Matplotlib, Seaborn |
| Environment | Jupyter Notebook |
| Deployment | Streamlit (optional) |

---

## 🌱 Future Enhancements
- 🧮 Integrate **SHAP** for deeper explainability  
- 💻 Add a **real-time fraud prediction API**  
- 🪙 Deploy Streamlit dashboard for live scoring  
- 📈 Enable **continuous learning** with streaming data  

---

## 👨‍💻 Author
**K. Shashi Preetham**  
_Data Analyst · Data Science & ML Enthusiast_  
📍 India  

🔗 [LinkedIn](https://www.linkedin.com/in/shashikathi)  
🔗 [GitHub](https://github.com/shashikathi)

---

## 🧾 License  
Licensed under the **MIT License** — free to use and modify with attribution.

---

## 🌟 Show Some ❤️  
If this project helped or inspired you, give it a ⭐ on GitHub!  
Let’s connect and build data-driven solutions together 💡

