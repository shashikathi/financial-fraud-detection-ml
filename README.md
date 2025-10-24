# 💸 Financial Fraud Detection System

> _“Fraud never sleeps. Neither does this dashboard.”_

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-forestgreen?logo=lightgbm)](https://lightgbm.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikit-learn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-lightblue?logo=plotly)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-purple?logo=pandas)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Project-Production%20Ready-brightgreen)](https://financial-fraud-detection-ml.streamlit.app/)

---

## 🚀 TL;DR
**End-to-end ML pipeline + interactive dashboard** to detect financial fraud in real time.  
- Trained on **real transaction data**  
- Powered by **LightGBM**  
- Fully **interactive Plotly dashboards**  
- High recall focus — because missing a fraud costs millions 💸  

> Fraud detection isn’t just about accuracy — it’s about **actionable insights** before it’s too late.

---

## ⚡ Highlights

- 🧹 Clean & preprocess large transaction datasets  
- 🧮 **LightGBM** classifier tuned for high recall  
- 🔍 Explainable AI with **feature importance insights**  
- 💹 Interactive, animated **Plotly dashboards**  
- 📊 Performance: **98.4% Accuracy · 94.7% Recall · 0.992 ROC-AUC**  
- 🌐 Production-ready for **Streamlit deployment**  

---

## 🖼 Dashboard Sneak Peek  

### 📊 Transaction Type Distribution
Breakdown of CASH_IN, CASH_OUT, TRANSFER, PAYMENT transactions  

### 💰 Transaction Amount Distribution (Animated)
See how transaction volumes change over time  

### 🧾 Origin Account Balance Changes
Animated before/after balances for origin accounts  

### ⚖️ Fraud Rate by Transaction Type
Percentage of fraud per transaction category  

### 🧠 Feature Importance (Explainable AI)
Which features drive fraud predictions — transparently  

---

## 🗂 Folder Structure

```

fraud-detection-system/
│
├─ app.py                     # Streamlit dashboard
├─ Fraud_Detection_with_Machine_Learning.ipynb # Complete ML pipeline
├─ model.pkl                  # Pre-trained LightGBM model
├─ data/transactions.csv      # Sample dataset
├─ Graphs/                    # Interactive Plotly dashboards
├─ requirements.txt           # All dependencies
└─ README.md                  # This file

````

---

## 🧠 Model Workflow

1️⃣ **Data Cleaning & Preprocessing** – handled nulls, encoded categorical features, normalized amounts  
2️⃣ **Feature Engineering** – created balance deltas, time-based metrics  
3️⃣ **Model Training** – optimized **LightGBM** for recall & precision  
4️⃣ **Evaluation** – accuracy, recall, precision, ROC-AUC  
5️⃣ **Explainability** – feature importance & SHAP visualizations  

---

## 📈 Performance Snapshot

| Metric               | Score      |
|---------------------|-----------|
| Accuracy             | 🟢 98.4%  |
| Recall (Fraud Class) | 🟡 94.7%  |
| Precision            | 🟢 95.3%  |
| ROC-AUC              | 🔵 0.992  |

> ⚡ Tuned for **high recall** — because missing a fraud costs way more than a false alarm.

---

## 🚀 Quickstart

```bash
# Clone repository
git clone https://github.com/shashikathi/fraud-detection-system.git
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook Fraud_Detection_with_Machine_Learning.ipynb
````

(Optional — for interactive dashboard)

```bash
streamlit run app.py
```

---

## 🧩 Tech Stack

| Category      | Tools                                         |
| ------------- | --------------------------------------------- |
| ML & Data     | Python, Pandas, NumPy, Scikit-learn, LightGBM |
| Visualization | Plotly, Matplotlib, Seaborn                   |
| Deployment    | Streamlit, HuggingFace Spaces (optional)      |
| Environment   | Jupyter Notebook                              |

---

## 🌱 Future Enhancements

* Integrate **SHAP** for deeper model explainability
* Real-time fraud prediction API
* Streamlit dashboard live scoring
* Continuous learning on streaming data

---

## 👨‍💻 Author

**K. Shashi Preetham** – B.Tech CSE (Hons), Data Science & ML Enthusiast
📍 India

🔗 [LinkedIn](https://www.linkedin.com/in/shashikathi)
🔗 [GitHub](https://github.com/shashikathi)

---

## 🌟 Show Some ❤️

If this project inspired you, give it a ⭐ on GitHub!
Let's build smarter, data-driven solutions together.
Do you want me to do that next?
```
