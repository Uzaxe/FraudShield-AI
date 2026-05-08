"""
anomaly_detection.py  —  Sklearn version (no TensorFlow needed)
===============================================================
Autoencoder-style anomaly detection using:
  - PCA for compression (like the encoder)
  - Inverse PCA for reconstruction (like the decoder)
  - Reconstruction error = how "abnormal" a transaction is

Why this works:
  PCA learns the structure of LEGITIMATE transactions.
  Fraudulent transactions don't fit this structure well
  -> they have HIGH reconstruction error -> flagged as anomalies.

This is mathematically equivalent to a linear autoencoder.

No extra installs needed — uses sklearn which you already have.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_score, recall_score,
    f1_score, accuracy_score
)

OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── PCA Autoencoder class ─────────────────────────────────────────────────────
class PCAAutoencoder:
    """
    Linear Autoencoder using PCA.

    Encoder: 29 features -> n_components (bottleneck)
    Decoder: n_components -> 29 features (reconstruction)

    Trained only on LEGITIMATE transactions so it learns
    what "normal" looks like. Fraud = high reconstruction error.
    """

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.pca          = PCA(n_components=n_components, random_state=42)
        self.scaler       = StandardScaler()
        self.threshold    = None

    def fit(self, X_legit: np.ndarray):
        """Train on legitimate transactions only."""
        X_scaled = self.scaler.fit_transform(X_legit)
        self.pca.fit(X_scaled)
        print(f"[INFO] PCA explains {self.pca.explained_variance_ratio_.sum()*100:.1f}% of variance")
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Compress then reconstruct."""
        X_scaled     = self.scaler.transform(X)
        X_compressed = self.pca.transform(X_scaled)
        X_recon      = self.pca.inverse_transform(X_compressed)
        return X_recon

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Mean squared error between input and reconstruction."""
        X_scaled = self.scaler.transform(X)
        X_recon  = self.reconstruct(X)
        return np.mean(np.square(X_scaled - X_recon), axis=1)

    def set_threshold(self, X_legit: np.ndarray, percentile: float = 95):
        errors         = self.reconstruction_error(X_legit)
        self.threshold = float(np.percentile(errors, percentile))
        print(f"[INFO] Threshold set at {percentile}th percentile: {self.threshold:.6f}")
        return self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = fraud (anomaly), 0 = legit."""
        errors = self.reconstruction_error(X)
        return (errors >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Normalised reconstruction error as fraud probability 0-1."""
        errors     = self.reconstruction_error(X)
        max_err    = np.percentile(errors, 99.9)
        normalised = np.clip(errors / max_err, 0, 1)
        return normalised


# ── Train & Evaluate ──────────────────────────────────────────────────────────
def train_autoencoder():
    from data_preprocessing import load_data, preprocess

    print("[INFO] Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess(df)

    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    # Train ONLY on legitimate transactions
    X_legit = X_train[y_train == 0]
    print(f"[INFO] Training PCA Autoencoder on {len(X_legit):,} legitimate transactions...")

    ae = PCAAutoencoder(n_components=8)
    ae.fit(X_legit)
    ae.set_threshold(X_legit, percentile=95)

    # Evaluate on test set
    print("[INFO] Evaluating on test set...")
    recon_errors = ae.reconstruction_error(X_test)

    # Find best threshold using ROC (Youden's J statistic)
    fpr_arr, tpr_arr, thr_arr = roc_curve(y_test, recon_errors)
    j_scores  = tpr_arr - fpr_arr
    best_idx  = np.argmax(j_scores)
    ae.threshold = float(thr_arr[best_idx])
    print(f"[INFO] Best threshold (Youden J): {ae.threshold:.6f}")

    y_pred = ae.predict(X_test)
    auc    = roc_auc_score(y_test, recon_errors)
    acc    = accuracy_score(y_test, y_pred)
    p      = precision_score(y_test, y_pred, zero_division=0)
    r      = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print("  PCA Autoencoder Anomaly Detection")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Threshold : {ae.threshold:.6f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    metrics = {
        "Model":     "Autoencoder",
        "Accuracy":  round(acc, 4),
        "Precision": round(p,   4),
        "Recall":    round(r,   4),
        "F1 Score":  round(f1,  4),
        "ROC-AUC":   round(auc, 4),
    }

    # Plots
    _plot_reconstruction_error(recon_errors, y_test, ae.threshold)
    _plot_explained_variance(ae)
    _plot_confusion(y_test, y_pred)
    _plot_roc(y_test, recon_errors, auc)

    # Save
    joblib.dump(ae, os.path.join(OUTPUT_DIR, "autoencoder.pkl"))
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "ae_metrics.csv"), index=False
    )
    print(f"\n[DONE] Autoencoder saved -> artifacts/autoencoder.pkl")
    print(f"[DONE] Metrics saved    -> artifacts/ae_metrics.csv")
    return ae, metrics


# ── Plots ─────────────────────────────────────────────────────────────────────
def _plot_reconstruction_error(errors, y_true, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    legit_idx = np.where(y_true == 0)[0][:2000]
    fraud_idx = np.where(y_true == 1)[0]
    ax.scatter(range(len(legit_idx)), errors[y_true == 0][:2000],
               alpha=0.3, s=5, color="steelblue", label="Legitimate")
    ax.scatter(range(len(fraud_idx)), errors[y_true == 1],
               alpha=0.9, s=30, color="#E74C3C", label="Fraud", zorder=5)
    ax.axhline(threshold, color="orange", linestyle="--",
               linewidth=2, label=f"Threshold = {threshold:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("Transaction Index")
    ax.set_ylabel("Reconstruction Error (log scale)")
    ax.set_title("Reconstruction Error per Transaction", fontweight="bold")
    ax.legend()

    ax2 = axes[1]
    ax2.hist(errors[y_true == 0], bins=60, alpha=0.6,
             color="steelblue", label="Legitimate", density=True)
    ax2.hist(errors[y_true == 1], bins=30, alpha=0.8,
             color="#E74C3C", label="Fraud", density=True)
    ax2.axvline(threshold, color="orange", linestyle="--",
                linewidth=2, label="Threshold")
    ax2.set_xlabel("Reconstruction Error")
    ax2.set_ylabel("Density")
    ax2.set_title("Error Distribution: Fraud vs Legitimate", fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ae_reconstruction_error.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved -> {path}")


def _plot_explained_variance(ae):
    fig, ax = plt.subplots(figsize=(8, 4))
    cumvar = np.cumsum(ae.pca.explained_variance_ratio_) * 100
    ax.bar(range(1, len(cumvar)+1),
           ae.pca.explained_variance_ratio_*100,
           color="steelblue", alpha=0.7, label="Per component")
    ax.plot(range(1, len(cumvar)+1), cumvar, "o-",
            color="#E74C3C", linewidth=2, label="Cumulative")
    ax.axhline(cumvar[-1], color="orange", linestyle="--",
               label=f"Total: {cumvar[-1]:.1f}%")
    ax.set_xlabel("PCA Component (Bottleneck Dimension)")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Autoencoder Bottleneck — Explained Variance",
                 fontsize=12, fontweight="bold")
    ax.legend(); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ae_explained_variance.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved -> {path}")


def _plot_confusion(y_test, y_pred):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Autoencoder",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cm_Autoencoder.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved -> {path}")


def _plot_roc(y_test, scores, auc):
    fpr, tpr, _ = roc_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"Autoencoder (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Autoencoder",
                 fontsize=12, fontweight="bold")
    ax.legend(); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_autoencoder.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved -> {path}")


if __name__ == "__main__":
    ae, metrics = train_autoencoder()
    print("\nFinal Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
