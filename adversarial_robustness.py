"""
adversarial_robustness.py — Fast version
=========================================
Adversarial Robustness & Evasion Defense.
No new installs — only numpy, sklearn, xgboost.
Runs in under 3 minutes.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, json
from sklearn.metrics import (
    roc_auc_score, recall_score,
    precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from xgboost import XGBClassifier

OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── FGSM-style attack ─────────────────────────────────────────────────────────
class TabularFGSM:
    """
    Perturbs top-k important features in the direction
    that DECREASES fraud probability — making fraud look legit.
    Simulates a fraudster who probes and evades the model.
    """
    def __init__(self, model, feature_importances, epsilon=0.15, top_k=8):
        self.model    = model
        self.feat_imp = feature_importances / (feature_importances.max() + 1e-10)
        self.epsilon  = epsilon
        self.top_k    = top_k
        self.top_idx  = np.argsort(self.feat_imp)[::-1][:top_k]

    def attack(self, X_fraud):
        X_adv = X_fraud.copy().astype(np.float64)
        for i in range(len(X_adv)):
            for fi in self.top_idx:
                imp = self.feat_imp[fi]
                mag = self.epsilon * imp
                xp  = X_adv[i].copy(); xp[fi]  += mag
                xm  = X_adv[i].copy(); xm[fi]  -= mag
                pp  = self.model.predict_proba(xp.reshape(1,-1))[0][1]
                pm  = self.model.predict_proba(xm.reshape(1,-1))[0][1]
                if pp < pm:
                    X_adv[i, fi] += mag
                else:
                    X_adv[i, fi] -= mag
        return X_adv

    def evasion_rate(self, X_fraud, threshold=0.5):
        X_adv  = self.attack(X_fraud)
        probs  = self.model.predict_proba(X_adv)[:, 1]
        evaded = (probs < threshold).sum()
        return X_adv, float(evaded / len(X_fraud) * 100), probs


# ── Adversarial Trainer ───────────────────────────────────────────────────────
class AdversarialTrainer:
    def __init__(self, epsilon=0.15, augment_ratio=0.5):
        self.epsilon       = epsilon
        self.augment_ratio = augment_ratio

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        n_legit = (y_train == 0).sum()
        n_fraud = (y_train == 1).sum()

        # Step 1: Baseline model
        print("[INFO] Training baseline model...")
        self.baseline = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            scale_pos_weight=n_legit / max(n_fraud, 1),
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
        self.baseline.fit(X_train, y_train)
        print("[INFO] Baseline trained.")

        # Step 2: Generate adversarial examples
        print("[INFO] Generating adversarial examples...")
        fi          = self.baseline.feature_importances_
        attacker    = TabularFGSM(self.baseline, fi, self.epsilon)
        fraud_idx   = np.where(y_train == 1)[0]
        n_adv       = max(1, int(len(fraud_idx) * self.augment_ratio))
        # Use a small sample for speed
        n_adv       = min(n_adv, 300)
        adv_idx     = np.random.choice(fraud_idx, size=n_adv, replace=False)
        X_adv       = attacker.attack(X_train[adv_idx])
        print(f"[INFO] Generated {len(X_adv)} adversarial examples.")

        # Step 3: Augment + retrain
        print("[INFO] Training robust model...")
        X_aug = np.vstack([X_train, X_adv])
        y_aug = np.concatenate([y_train, np.ones(len(X_adv), dtype=int)])
        n_l2  = (y_aug == 0).sum(); n_f2 = (y_aug == 1).sum()

        self.robust = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            scale_pos_weight=n_l2 / max(n_f2, 1),
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
        self.robust.fit(X_aug, y_aug)
        print("[INFO] Robust model trained.")

        self.fi = fi / (fi.max() + 1e-10)
        return self

    def evaluate(self, X_test, y_test, threshold=0.5):
        X_test    = np.array(X_test)
        y_test    = np.array(y_test)
        fraud_idx = np.where(y_test == 1)[0]

        # Use small sample of fraud for attack speed
        n_attack  = min(len(fraud_idx), 100)
        atk_idx   = np.random.choice(fraud_idx, size=n_attack, replace=False)
        X_fr      = X_test[atk_idx]

        print(f"[INFO] Attacking {n_attack} fraud transactions...")

        att_base   = TabularFGSM(self.baseline, self.fi, self.epsilon)
        X_adv_base, er_base, _ = att_base.evasion_rate(X_fr)

        att_rob    = TabularFGSM(self.robust, self.fi, self.epsilon)
        X_adv_rob, er_rob, _   = att_rob.evasion_rate(X_fr)

        def metrics(model, X, y):
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)
            return {
                "AUC":       round(roc_auc_score(y, probs), 4),
                "Recall":    round(recall_score(y, preds, zero_division=0), 4),
                "Precision": round(precision_score(y, preds, zero_division=0), 4),
                "F1":        round(f1_score(y, preds, zero_division=0), 4),
            }

        # Build adversarial test sets
        X_adv_full_base = X_test.copy(); X_adv_full_base[atk_idx] = X_adv_base
        X_adv_full_rob  = X_test.copy(); X_adv_full_rob[atk_idx]  = X_adv_rob

        results = {
            "Baseline — Clean":        metrics(self.baseline, X_test, y_test),
            "Baseline — Under Attack":  metrics(self.baseline, X_adv_full_base, y_test),
            "Robust — Clean":          metrics(self.robust,   X_test, y_test),
            "Robust — Under Attack":    metrics(self.robust,   X_adv_full_rob,  y_test),
            "evasion_rate_baseline":    round(er_base, 2),
            "evasion_rate_robust":      round(er_rob,  2),
        }

        print(f"\n{'='*50}")
        print("  Adversarial Robustness Results")
        print(f"{'='*50}")
        print(f"  Evasion rate baseline : {er_base:.1f}%")
        print(f"  Evasion rate robust   : {er_rob:.1f}%")
        print(f"  Improvement           : {er_base - er_rob:.1f}%")
        for k, v in results.items():
            if isinstance(v, dict):
                print(f"\n  {k}:")
                for mk, mv in v.items():
                    print(f"    {mk}: {mv}")

        return results

    def save(self):
        joblib.dump(self.baseline, os.path.join(OUTPUT_DIR, "adv_baseline.pkl"))
        joblib.dump(self.robust,   os.path.join(OUTPUT_DIR, "adv_robust.pkl"))
        print("[INFO] Saved adv_baseline.pkl & adv_robust.pkl")


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_comparison(results):
    scenarios = ["Baseline — Clean","Baseline — Under Attack",
                 "Robust — Clean","Robust — Under Attack"]
    metrics   = ["AUC","Recall","F1"]
    colors    = ["#00BCD4","#E74C3C","#2ECC71","#F39C12"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Adversarial Robustness: Baseline vs Robust",
                 fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        vals = [results[s][metric] for s in scenarios if s in results]
        bars = ax.bar(range(len(vals)), vals, color=colors, width=0.6, edgecolor="white")
        ax.set_title(metric, fontweight="bold"); ax.set_ylim(0, 1.1)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(["Base\nClean","Base\nAttack","Robust\nClean","Robust\nAttack"],
                            fontsize=9)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "adversarial_robustness.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved → {path}")


def plot_evasion(results):
    base_r  = results["evasion_rate_baseline"]
    rob_r   = results["evasion_rate_robust"]
    improvement = base_r - rob_r

    fig, ax = plt.subplots(figsize=(7, 4))
    bars    = ax.bar(["Baseline\n(Vulnerable)","Robust\n(Defended)"],
                     [base_r, rob_r],
                     color=["#E74C3C","#2ECC71"], width=0.4, edgecolor="white")
    ax.set_ylabel("Evasion Rate (%)")
    ax.set_title(f"Fraud Evasion Rate — {improvement:.1f}% Reduction After Defense",
                 fontweight="bold")
    ax.set_ylim(0, max(base_r, rob_r) * 1.35)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bar.get_height():.1f}%", ha="center",
                fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evasion_rates.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_adversarial_training():
    from data_preprocessing import load_data, preprocess

    print("[INFO] Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess(df)

    trainer = AdversarialTrainer(epsilon=0.15, augment_ratio=0.5)
    trainer.fit(X_train, y_train)
    results = trainer.evaluate(X_test, y_test)

    plot_comparison(results)
    plot_evasion(results)
    trainer.save()

    summary = {
        "evasion_rate_baseline": results["evasion_rate_baseline"],
        "evasion_rate_robust":   results["evasion_rate_robust"],
        "improvement":           round(results["evasion_rate_baseline"] -
                                        results["evasion_rate_robust"], 2),
        "baseline_clean_auc":    results["Baseline — Clean"]["AUC"],
        "robust_clean_auc":      results["Robust — Clean"]["AUC"],
        "baseline_attack_auc":   results["Baseline — Under Attack"]["AUC"],
        "robust_attack_auc":     results["Robust — Under Attack"]["AUC"],
    }
    with open(os.path.join(OUTPUT_DIR, "adversarial_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE] Adversarial robustness training complete.")
    return trainer, results


if __name__ == "__main__":
    run_adversarial_training()
