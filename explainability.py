"""
explainability.py — Fixed final version
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, shap

OUTPUT_DIR = "artifacts"


def load_artifacts():
    model         = joblib.load(os.path.join(OUTPUT_DIR, "best_model.pkl"))
    X_test        = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test        = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    feature_names = pd.read_csv(
        os.path.join(OUTPUT_DIR, "feature_names.csv")
    ).iloc[:, 0].tolist()
    return model, X_test, y_test, feature_names


def compute_shap(model, X_test, sample_size=2000):
    rng     = np.random.default_rng(42)
    indices = rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_sub   = X_test[indices]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)

    # Normalize shape: some versions return (n, features, classes) or list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]                        # RF: take class-1
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]                  # 3-D: take class-1 slice

    # shap_values must be 2-D (n_samples, n_features) from here on
    print(f"[DEBUG] shap_values shape : {shap_values.shape}")
    print(f"[DEBUG] X_sub shape       : {X_sub.shape}")
    print(f"[DEBUG] n_features in CSV : {X_sub.shape[1]}")

    return explainer, shap_values, X_sub, indices


def plot_summary(shap_values, X_sub, feature_names):
    plt.figure()
    shap.summary_plot(shap_values, X_sub,
                      feature_names=feature_names,
                      show=False, plot_type="dot")
    plt.title("SHAP Summary Plot (Beeswarm)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved SHAP summary -> {path}")


def plot_bar(shap_values, feature_names):
    # shap_values is guaranteed 2-D here; n_cols must equal len(feature_names)
    n_feat = shap_values.shape[1]
    names  = feature_names[:n_feat]          # trim if mismatch (safety)

    mean_abs = np.abs(shap_values).mean(axis=0)   # shape (n_feat,)

    # Sort with plain Python — no numpy fancy indexing
    paired   = sorted(zip(mean_abs.tolist(), names), key=lambda x: x[0])
    vals     = [p[0] for p in paired[-20:]]
    labels   = [p[1] for p in paired[-20:]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, vals, color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Feature Importances (SHAP)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved SHAP bar chart -> {path}")


def plot_waterfall(explainer, X_sub, shap_values, feature_names, y_sub):
    fraud_idx = np.where(y_sub == 1)[0]
    idx       = int(fraud_idx[0]) if len(fraud_idx) > 0 else 0

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.array(base_val).flatten()[-1])
    else:
        base_val = float(base_val)

    n_feat = shap_values.shape[1]
    sv = shap.Explanation(
        values        = shap_values[idx],
        base_values   = base_val,
        data          = X_sub[idx],
        feature_names = feature_names[:n_feat]
    )
    plt.figure()
    shap.plots.waterfall(sv, show=False, max_display=15)
    plt.title("SHAP Waterfall - Single Fraud Prediction", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_waterfall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved SHAP waterfall -> {path}")


def run_explainability():
    model, X_test, y_test, feature_names = load_artifacts()

    print("[INFO] Computing SHAP values (this may take ~30 sec) ...")
    explainer, shap_values, X_sub, indices = compute_shap(model, X_test)
    y_sub = y_test[indices]

    plot_summary(shap_values, X_sub, feature_names)
    plot_bar(shap_values, feature_names)
    plot_waterfall(explainer, X_sub, shap_values, feature_names, y_sub)
    print("[DONE] All SHAP plots saved.")


if __name__ == "__main__":
    run_explainability()
