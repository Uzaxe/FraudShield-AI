"""
data_preprocessing.py
=====================
Loads the Credit Card Fraud dataset, scales features,
and applies SMOTE to handle class imbalance.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place 'creditcard.csv' in the same directory before running.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

DATA_PATH = "creditcard.csv"
OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Fraud cases   : {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")
    print(f"[INFO] Legit cases   : {(df['Class'] == 0).sum()}")
    return df


def preprocess(df: pd.DataFrame):
    """
    Steps:
      1. Drop 'Time' (not useful for tabular model).
      2. Scale 'Amount' with StandardScaler.
      3. Split into train/test (80/20, stratified).
      4. Apply SMOTE on training set only.

    Returns
    -------
    X_train_res, X_test, y_train_res, y_test, scaler
    """
    # 1. Drop Time
    df = df.drop(columns=["Time"])

    # 2. Scale Amount
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # 3. Split
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 4. SMOTE (only on train)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE — Train size: {X_train_res.shape[0]}")
    print(f"[INFO] Fraud in resampled train: {y_train_res.sum()}")

    # Save scaler & test set for app reuse
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test.values)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test.values)
    pd.Series(X.columns.tolist()).to_csv(
        os.path.join(OUTPUT_DIR, "feature_names.csv"), index=False
    )

    return X_train_res, X_test, y_train_res, y_test, scaler


if __name__ == "__main__":
    df = load_data()
    X_train_res, X_test, y_train_res, y_test, scaler = preprocess(df)
    print("[DONE] Preprocessing complete.")
