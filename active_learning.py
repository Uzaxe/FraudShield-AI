"""
active_learning.py — Fast version
==================================
Human-in-the-Loop Active Learning.
Finds uncertain transactions for human review.
Retrains model with human labels.
No new installs needed.
Runs in under 30 seconds.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, json
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from xgboost import XGBClassifier

OUTPUT_DIR     = "artifacts"
AL_QUEUE_FILE  = os.path.join(OUTPUT_DIR, "al_review_queue.json")
AL_LABELS_FILE = os.path.join(OUTPUT_DIR, "al_human_labels.json")
AL_HISTORY_FILE= os.path.join(OUTPUT_DIR, "al_history.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Uncertainty Sampling ──────────────────────────────────────────────────────
class UncertaintySampler:
    """
    Selects transactions the model is LEAST confident about.
    These are the most valuable ones for a human to label.

    Strategies:
      entropy          — highest prediction entropy (default, best)
      margin           — smallest gap between class probabilities
      least_confidence — lowest max class probability
    """
    def __init__(self, strategy="entropy"):
        self.strategy = strategy

    def score(self, probs):
        if self.strategy == "least_confidence":
            return 1 - probs.max(axis=1)
        elif self.strategy == "margin":
            s = np.sort(probs, axis=1)[:, ::-1]
            return 1 - (s[:, 0] - s[:, 1])
        else:  # entropy
            eps = 1e-10
            p   = np.clip(probs, eps, 1-eps)
            return -np.sum(p * np.log(p), axis=1)

    def select(self, X, model, n=20, exclude=None):
        probs  = model.predict_proba(X)
        scores = self.score(probs)
        if exclude:
            scores[list(exclude)] = -np.inf
        idx = np.argsort(scores)[::-1][:n]
        return idx, scores


# ── Active Learning Manager ───────────────────────────────────────────────────
class ActiveLearningManager:
    def __init__(self, strategy="entropy", batch_size=20):
        self.strategy   = strategy
        self.batch_size = batch_size
        self.sampler    = UncertaintySampler(strategy)
        self.history    = self._load(AL_HISTORY_FILE,
                                     {"iterations":[],"auc":[],"f1":[],"recall":[],"n_labeled":[]})
        self.labels     = self._load(AL_LABELS_FILE, {})
        self.queue      = self._load(AL_QUEUE_FILE,  [])

    def _load(self, path, default):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default

    def _save(self):
        for path, data in [(AL_HISTORY_FILE, self.history),
                            (AL_LABELS_FILE,  self.labels),
                            (AL_QUEUE_FILE,   self.queue)]:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def generate_queue(self, X_pool, model, feature_names, n=20):
        reviewed = set(int(k) for k in self.labels)
        sel_idx, scores = self.sampler.select(X_pool, model, n=n, exclude=reviewed)
        probs = model.predict_proba(X_pool)[:, 1]

        self.queue = []
        for idx in sel_idx:
            self.queue.append({
                "idx":           int(idx),
                "fraud_prob":    round(float(probs[idx]), 4),
                "uncertainty":   round(float(scores[idx]), 4),
                "model_verdict": "FRAUD" if probs[idx] >= 0.5 else "LEGIT",
                "reviewed":      str(idx) in self.labels,
                "human_label":   self.labels.get(str(idx), {}).get("label", None),
                "features":      {str(name): round(float(val), 4)
                                  for name, val in zip(feature_names, X_pool[idx])},
            })
        self._save()
        print(f"[INFO] Queue generated: {len(self.queue)} uncertain transactions")
        return self.queue

    def submit_label(self, idx, label, analyst="Analyst"):
        self.labels[str(idx)] = {
            "label":     label,
            "analyst":   analyst,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        for item in self.queue:
            if item["idx"] == idx:
                item["reviewed"]   = True
                item["human_label"]= label
        self._save()
        print(f"[INFO] Labeled: #{idx} → {'FRAUD' if label else 'LEGIT'}")

    def retrain(self, X_train, y_train, X_test, y_test):
        if not self.labels:
            print("[WARN] No labels yet.")
            return {}

        labeled_idx   = [int(k) for k in self.labels]
        human_labels  = [self.labels[str(k)]["label"] for k in labeled_idx]

        # Load pool data
        X_pool = np.load(os.path.join(OUTPUT_DIR, "al_X_pool.npy"))
        X_lbl  = X_pool[labeled_idx]
        y_lbl  = np.array(human_labels)

        X_comb = np.vstack([X_train, X_lbl])
        y_comb = np.concatenate([y_train, y_lbl])

        n_l = (y_comb == 0).sum(); n_f = (y_comb == 1).sum()

        # Original metrics
        orig  = joblib.load(os.path.join(OUTPUT_DIR, "best_model.pkl"))
        op    = orig.predict_proba(X_test)[:, 1]
        orig_metrics = {
            "auc":    roc_auc_score(y_test, op),
            "f1":     f1_score(y_test, (op>=0.5).astype(int), zero_division=0),
            "recall": recall_score(y_test, (op>=0.5).astype(int), zero_division=0),
        }

        # Retrain
        print(f"[INFO] Retraining with {len(y_lbl)} human labels...")
        new_model = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            scale_pos_weight=n_l/max(n_f,1),
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
        new_model.fit(X_comb, y_comb)

        np_    = new_model.predict_proba(X_test)[:, 1]
        new_metrics = {
            "auc":    roc_auc_score(y_test, np_),
            "f1":     f1_score(y_test, (np_>=0.5).astype(int), zero_division=0),
            "recall": recall_score(y_test, (np_>=0.5).astype(int), zero_division=0),
        }

        # Update history
        it = len(self.history["iterations"]) + 1
        self.history["iterations"].append(it)
        self.history["auc"].append(round(new_metrics["auc"], 4))
        self.history["f1"].append(round(new_metrics["f1"], 4))
        self.history["recall"].append(round(new_metrics["recall"], 4))
        self.history["n_labeled"].append(len(labeled_idx))
        self._save()

        joblib.dump(new_model, os.path.join(OUTPUT_DIR, "al_retrained_model.pkl"))

        result = {
            "original_auc":    round(orig_metrics["auc"],    4),
            "new_auc":         round(new_metrics["auc"],     4),
            "auc_improvement": round(new_metrics["auc"] - orig_metrics["auc"], 4),
            "original_f1":     round(orig_metrics["f1"],     4),
            "new_f1":          round(new_metrics["f1"],      4),
            "original_recall": round(orig_metrics["recall"], 4),
            "new_recall":      round(new_metrics["recall"],  4),
            "n_human_labels":  len(labeled_idx),
        }

        print(f"  AUC:    {result['original_auc']:.4f} → {result['new_auc']:.4f} ({result['auc_improvement']:+.4f})")
        print(f"  F1:     {result['original_f1']:.4f} → {result['new_f1']:.4f}")
        print(f"  Recall: {result['original_recall']:.4f} → {result['new_recall']:.4f}")
        return result

    def get_stats(self):
        reviewed = len(self.labels)
        fraud_l  = sum(1 for v in self.labels.values() if v["label"]==1)
        return {
            "queued":        len(self.queue),
            "reviewed":      reviewed,
            "pending":       len(self.queue) - reviewed,
            "fraud_labeled": fraud_l,
            "legit_labeled": reviewed - fraud_l,
            "history":       self.history,
        }


# ── Initialize ─────────────────────────────────────────────────────────────────
def initialize_active_learning():
    from data_preprocessing import load_data, preprocess

    print("[INFO] Initializing Active Learning...")
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess(df)
    X_test  = np.array(X_test)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    model = joblib.load(os.path.join(OUTPUT_DIR, "best_model.pkl"))
    feats = pd.read_csv(
        os.path.join(OUTPUT_DIR, "feature_names.csv")
    ).iloc[:, 0].tolist()

    mgr   = ActiveLearningManager(strategy="entropy")
    queue = mgr.generate_queue(X_test, model, feats, n=20)

    # Save pool + train for retraining
    np.save(os.path.join(OUTPUT_DIR, "al_X_pool.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "al_y_pool.npy"),  y_test)
    np.save(os.path.join(OUTPUT_DIR, "al_X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "al_y_train.npy"), y_train)

    print(f"[DONE] Active Learning initialized. {len(queue)} transactions queued.")
    return mgr


if __name__ == "__main__":
    mgr   = initialize_active_learning()
    stats = mgr.get_stats()
    print(f"\nStats:")
    print(f"  Queued  : {stats['queued']}")
    print(f"  Reviewed: {stats['reviewed']}")
    print(f"  Pending : {stats['pending']}")
