"""
model_training.py
=================
Trains two models — Random Forest and XGBoost — on the
preprocessed + SMOTE-balanced data. Saves best model.

Metrics reported: Accuracy, Precision, Recall, F1, ROC-AUC,
                  Confusion Matrix, Classification Report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from data_preprocessing import load_data, preprocess

OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall"   : round(recall_score(y_test, y_pred),    4),
        "F1 Score" : round(f1_score(y_test, y_pred),        4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_proba),  4),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return metrics, y_pred, y_proba


def plot_confusion_matrix(name: str, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"cm_{name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved confusion matrix → {path}")


def plot_roc_curves(results):
    """Overlay ROC curves for all models."""
    plt.figure(figsize=(7, 5))
    for name, fpr, tpr, auc in results:
        plt.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved ROC curves → {path}")


def plot_feature_importance(model, feature_names, model_name: str, top_n: int = 20):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_vals     = importances[indices]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_vals, y=top_features, palette="viridis")
    plt.title(f"Top {top_n} Feature Importances — {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"feature_importance_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved feature importance → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    # Load & preprocess
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess(df)
    feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else \
                    pd.read_csv(os.path.join(OUTPUT_DIR, "feature_names.csv")).iloc[:, 0].tolist()

    # ── Model 1 : Random Forest ────────────────────────────────────────────
    print("\n[TRAINING] Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_metrics, rf_pred, rf_proba = evaluate("Random Forest", rf, X_test, y_test)

    # ── Model 2 : XGBoost ─────────────────────────────────────────────────
    print("\n[TRAINING] XGBoost …")
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_metrics, xgb_pred, xgb_proba = evaluate("XGBoost", xgb, X_test, y_test)

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_confusion_matrix("Random Forest", y_test, rf_pred)
    plot_confusion_matrix("XGBoost",       y_test, xgb_pred)

    roc_results = []
    for name, proba in [("Random Forest", rf_proba), ("XGBoost", xgb_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        roc_results.append((name, fpr, tpr, auc))
    plot_roc_curves(roc_results)

    feat_arr = np.array(feature_names)
    plot_feature_importance(rf,  feat_arr, "Random Forest")
    plot_feature_importance(xgb, feat_arr, "XGBoost")

    # ── Save best model ────────────────────────────────────────────────────
    best_model = xgb if xgb_metrics["ROC-AUC"] >= rf_metrics["ROC-AUC"] else rf
    best_name  = "XGBoost" if best_model is xgb else "Random Forest"
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.pkl"))
    print(f"\n[INFO] Best model ({best_name}) saved to artifacts/best_model.pkl")

    # ── Comparison table ───────────────────────────────────────────────────
    comparison = pd.DataFrame([rf_metrics, xgb_metrics])
    comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\n[INFO] Model Comparison:")
    print(comparison.to_string(index=False))

    return best_model, feature_names


if __name__ == "__main__":
    train()
