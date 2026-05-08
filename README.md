# 💳 Real-Time Credit Card Fraud Detection System
### B.E. (AI & ML) — Major Project

---

## 📌 Project Overview
A complete end-to-end machine learning system to detect fraudulent credit card
transactions in real time. The system handles extreme class imbalance using
**SMOTE**, trains two models (**Random Forest** and **XGBoost**), provides
**SHAP-based explainability**, and exposes a live **Streamlit web application**.

---

## 🗂️ Project Structure
```
fraud_detection/
│
├── data_preprocessing.py   # Data loading, scaling, SMOTE balancing
├── model_training.py       # Train RF & XGBoost, save best model, plots
├── explainability.py       # SHAP summary / bar / waterfall plots
├── app.py                  # Streamlit web application
├── utils.py                # Shared helper functions
├── requirements.txt        # Python dependencies
├── artifacts/              # Generated after training (models, plots)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.csv
│   ├── X_test.npy / y_test.npy
│   ├── roc_curves.png
│   ├── shap_summary.png
│   └── ...
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download
```bash
git clone <your-repo-url>
cd fraud_detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Download **creditcard.csv** and place it in the `fraud_detection/` folder.

---

## 🚀 Run the Pipeline

### Step 1 — Preprocess Data
```bash
python data_preprocessing.py
```

### Step 2 — Train Models
```bash
python model_training.py
```
This saves `artifacts/best_model.pkl` and all evaluation plots.

### Step 3 — Generate SHAP Explanations
```bash
python explainability.py
```

### Step 4 — Launch Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📊 Models & Techniques

| Component         | Detail                                  |
|-------------------|-----------------------------------------|
| Dataset           | Kaggle Credit Card Fraud (284,807 txns) |
| Imbalance Handling| SMOTE (Synthetic Minority Oversampling) |
| Model 1           | Random Forest (200 trees)               |
| Model 2           | XGBoost (300 estimators, LR=0.05)       |
| Explainability    | SHAP TreeExplainer                      |
| Evaluation        | F1, ROC-AUC, Confusion Matrix, PR curve |
| Deployment        | Streamlit Web App                       |

---

## 📈 Expected Results

| Metric     | Random Forest | XGBoost  |
|------------|---------------|----------|
| Accuracy   | ~99.9%        | ~99.9%   |
| Recall     | ~82–86%       | ~84–88%  |
| Precision  | ~88–92%       | ~90–94%  |
| F1 Score   | ~85–89%       | ~87–91%  |
| ROC-AUC    | ~97–98%       | ~98–99%  |

> Note: Actual results vary slightly with each run due to random seeds.

---

## 🎓 Academic Notes

- **Class Imbalance**: Only ~0.17% of transactions are fraudulent. SMOTE creates
  synthetic minority samples to balance the training set.
- **Why XGBoost**: Gradient boosting with `scale_pos_weight` handles imbalance
  natively and typically outperforms RF on tabular financial data.
- **SHAP**: SHAP values decompose each prediction into per-feature contributions,
  making the model interpretable to non-technical stakeholders.

---

## 👨‍💻 Tech Stack
`Python 3.10+` | `Scikit-learn` | `XGBoost` | `imbalanced-learn`
`SHAP` | `Streamlit` | `Matplotlib` | `Seaborn` | `Pandas` | `NumPy`

---
*Real-Time Fraud Detection — B.E. AI & ML Major Project*
