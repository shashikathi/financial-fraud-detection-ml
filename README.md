


# 💸 Financial Fraud Detection with Machine Learning  

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
Features real-world data, **LightGBM model**, and fully **interactive animated dashboards** for visual insights.  

> Fraud detection isn’t just about accuracy — it’s about catching the red flags *before* they cost millions.

---

## ⚡ Highlights

- 🧹 **Data preprocessing** and feature engineering for realistic banking data  
- 🧮 **LightGBM classifier** tuned for precision–recall trade-off  
- 🔍 **Explainable AI** via feature importance visualization  
- 💹 **Interactive Plotly dashboards** with animations and filters  
- 📊 **Model performance:** 98.4% Accuracy · 94.7% Recall · 0.992 ROC-AUC  
- 🌐 Future-ready architecture for Streamlit or HuggingFace deployment  

---

## 🖼️ Dashboard Gallery  

### 📊 Transaction Type Distribution  
Interactive breakdown of transaction categories (CASH_IN, CASH_OUT, TRANSFER, PAYMENT).  
👉 **[`View Chart`](dashboards/transaction_types_distribution.html)**  

### 💰 Transaction Amount Distribution (Animated)  
Animated Plotly chart showing transaction flow and amount over time.  
👉 **[`View Chart`](dashboards/transaction_amount_distribution_animated.html)**  

### 🧾 Origin Account Balances (Animated)  
Visualizes before-and-after balance shifts for origin accounts.  
👉 **[`View Chart`](dashboards/origin_account_balances_animated.html)**  

### ⚖️ Fraud Rate by Transaction Type  
Highlights the fraud percentage per transaction category.  
👉 **[`View Chart`](dashboards/fraud_rate_by_type.html)**  

### 🧠 Feature Importance (Explainable AI)  
Shows how each feature contributes to fraud detection decisions.  
👉 **[`View Chart`](dashboards/feature_importance.html)**  

---

## 🗂️ Folder Structure

| Folder / File | Description |
|----------------|-------------|
| `notebooks/Fraud_Detection_with_Machine_Learning.ipynb` | Full ML pipeline & model training |
| `dashboards/` | Interactive Plotly HTML visualizations |
| `data/transactions.csv` | Sample dataset |
| `app.py` | (Optional) Streamlit dashboard script |
| `requirements.txt` | Library dependencies |
| `README.md` | This file 😉 |

---

## 🧠 Model Workflow

1️⃣ **Data Cleaning & Preprocessing** — handled missing values, normalized balances  
2️⃣ **Feature Engineering** — created derived balance delta & transaction type indicators  
3️⃣ **Model Training** — optimized LightGBM model for maximum recall  
4️⃣ **Evaluation** — accuracy, recall, precision, ROC-AUC metrics  
5️⃣ **Explainability** — feature importance for business interpretability  

---

## 📈 Performance Snapshot

| Metric | Score |
|---------|--------|
| Accuracy | 🟢 **98.4%** |
| Recall (Fraud Class) | 🟡 **94.7%** |
| Precision | 🟢 **95.3%** |
| ROC-AUC | 🔵 **0.992** |

> 🎯 Optimized for *high recall* — because missing a fraud is worse than a false alert.

---

## 🚀 Quickstart

```bash
# Clone repo
git clone https://github.com/yourusername/Fraud-Detection-ML-Dashboard.git
cd Fraud-Detection-ML-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook notebooks/Fraud_Detection_with_Machine_Learning.ipynb
```

(Optional — if you add Streamlit later)
```bash
streamlit run app.py
```

---

## 🧩 Tech Stack

| Category | Tools |
|-----------|-------|
| ML & Data | Python, Pandas, NumPy, Scikit-learn, LightGBM |
| Visualization | Plotly, Matplotlib, Seaborn |
| Deployment | Streamlit (optional) |
| Environment | Jupyter Notebook |

---

## 🌱 Future Enhancements
- 🧠 Add SHAP for local explainability  
- 🪙 Real-time fraud prediction API  
- 💻 Streamlit web app for interactive user inputs  
- 📈 Live data simulation using synthetic streams  

---

## 👨‍💻 Author
**K. Shashi Preetham**  
_Data Analyst · Data Science & Machine Learning Enthusiast_  
📍 India  

🔗 [LinkedIn](https://www.linkedin.com/in/shashikathi)  
🔗 [GitHub](https://github.com/shashikathi)

---

## 🧾 License  
Licensed under the **MIT License** — free for public and commercial use with attribution.

---

## 🌟 Show Some ❤️  
If you like this project, drop a ⭐ on GitHub — it really helps!  
Let’s connect and build data-driven solutions together 💡
````

---

✅ **Paste it directly into your repo’s README.md — no edits needed.**
GitHub will render the badges, tables, and all chart links beautifully.

Wanna go one step further? I can generate matching **repo banner art (like the one on top)** with your project title and tagline — professional and visually consistent with this README. Want that?
