"""
gnn_model.py
============
Graph Neural Network for fraud detection using PyTorch Geometric.

How it works:
  - Each transaction = a NODE in a graph
  - Two transactions are CONNECTED if they share similar feature patterns
  - GNN learns fraud patterns by looking at a transaction AND its neighbours
    simultaneously — catching coordinated fraud rings

Architecture: GraphSAGE (2 layers) -> MLP classifier

Install:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install torch-geometric
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, joblib
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)

OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample size for graph — must be larger than number of fraud cases in TEST set
# We sample from the ORIGINAL (pre-SMOTE) data to avoid inflated fraud counts
SAMPLE_SIZE = 5000


# ── Graph Construction ────────────────────────────────────────────────────────
def build_transaction_graph(X, y, k_neighbours: int = 8):
    import torch
    from torch_geometric.data import Data
    from sklearn.neighbors import kneighbors_graph
    import scipy.sparse as sp

    print(f"[INFO] Building transaction graph for {len(X)} nodes...")
    X_arr = np.array(X, dtype=np.float32)

    # Use first 6 features for graph connectivity (most informative)
    connect_feats = X_arr[:, :6]
    A     = kneighbors_graph(connect_feats, n_neighbors=k_neighbours,
                              mode="connectivity", include_self=False)
    A_coo = sp.coo_matrix(A)

    edge_index = torch.tensor(
        np.vstack([A_coo.row, A_coo.col]), dtype=torch.long
    )
    x      = torch.tensor(X_arr, dtype=torch.float)
    labels = torch.tensor(np.array(y, dtype=np.int64), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=labels)
    print(f"[INFO] Graph: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{int(labels.sum())} fraud nodes")
    return data


# ── GNN Model ─────────────────────────────────────────────────────────────────
def build_gnn(input_dim: int, hidden_dim: int = 64):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    class FraudGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1  = SAGEConv(input_dim, hidden_dim)
            self.conv2  = SAGEConv(hidden_dim, hidden_dim // 2)
            self.bn1    = nn.BatchNorm1d(hidden_dim)
            self.bn2    = nn.BatchNorm1d(hidden_dim // 2)
            self.drop   = nn.Dropout(0.3)
            self.linear = nn.Linear(hidden_dim // 2, 2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.drop(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            return self.linear(x)

    return FraudGNN()


# ── Train ─────────────────────────────────────────────────────────────────────
def train_gnn():
    import torch
    import torch.nn.functional as F

    # ── Load RAW data (before SMOTE) to avoid inflated fraud counts ───────────
    print("[INFO] Loading RAW data (before SMOTE) for graph construction...")
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("creditcard.csv")
    df = df.drop(columns=["Time"])
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    # ── Sample CAREFULLY: all fraud + some legit ──────────────────────────────
    fraud_idx = np.where(y == 1)[0]   # 492 fraud cases
    legit_idx = np.where(y == 0)[0]   # 284315 legit cases

    n_fraud   = len(fraud_idx)         # use ALL 492 fraud cases
    n_legit   = min(SAMPLE_SIZE - n_fraud, len(legit_idx))

    if n_legit <= 0:
        # SAMPLE_SIZE too small — increase it
        n_legit = min(2000, len(legit_idx))

    print(f"[INFO] Sampling {n_fraud} fraud + {n_legit} legit = {n_fraud+n_legit} total")

    np.random.seed(42)
    sel_legit = np.random.choice(legit_idx, size=n_legit, replace=False)
    sel_idx   = np.concatenate([fraud_idx, sel_legit])
    np.random.shuffle(sel_idx)

    X_sample = X[sel_idx]
    y_sample = y[sel_idx]

    # ── Build graph ───────────────────────────────────────────────────────────
    data = build_transaction_graph(X_sample, y_sample, k_neighbours=8)

    # ── Train / val / test masks ──────────────────────────────────────────────
    n    = data.num_nodes
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:int(0.70*n)]]             = True
    val_mask  [perm[int(0.70*n):int(0.85*n)]] = True
    test_mask [perm[int(0.85*n):]]             = True
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    # ── Class weights ─────────────────────────────────────────────────────────
    n_legit_train = (data.y[train_mask] == 0).sum().item()
    n_fraud_train = (data.y[train_mask] == 1).sum().item()
    ratio  = n_legit_train / max(n_fraud_train, 1)
    weight = torch.tensor([1.0, ratio], dtype=torch.float)
    print(f"[INFO] Class weight for fraud: {ratio:.1f}x")

    # ── Model + optimizer ─────────────────────────────────────────────────────
    device = torch.device("cpu")
    model  = build_gnn(input_dim=data.num_node_features).to(device)
    data   = data.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss   = float("inf")
    patience_count  = 0
    train_losses    = []
    val_losses      = []

    print("[INFO] Training GNN (this takes 1-3 minutes)...")
    for epoch in range(1, 201):
        model.train()
        opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask],
                               data.y[data.train_mask], weight=weight)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(
                val_out[data.val_mask], data.y[data.val_mask]
            ).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Train: {loss.item():.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, "gnn_best.pt"))
        else:
            patience_count += 1
            if patience_count >= 25:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "gnn_best.pt"),
                   map_location="cpu")
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out   = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        true  = data.y.cpu().numpy()

    mask_np     = test_mask.numpy()
    test_probs  = probs[mask_np]
    test_preds  = preds[mask_np]
    test_true   = true[mask_np]

    auc = roc_auc_score(test_true, test_probs)
    acc = accuracy_score(test_true, test_preds)
    p   = precision_score(test_true, test_preds, zero_division=0)
    r   = recall_score(test_true, test_preds, zero_division=0)
    f1  = f1_score(test_true, test_preds, zero_division=0)

    print(f"\n{'='*50}")
    print("  GNN (GraphSAGE) Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(test_true, test_preds,
                                target_names=["Legit", "Fraud"]))

    metrics = {
        "Model":     "GNN (GraphSAGE)",
        "Accuracy":  round(acc, 4),
        "Precision": round(p,   4),
        "Recall":    round(r,   4),
        "F1 Score":  round(f1,  4),
        "ROC-AUC":   round(auc, 4),
    }

    # Plots
    _plot_gnn_loss(train_losses, val_losses)
    _plot_gnn_confusion(test_true, test_preds)
    _plot_gnn_roc(test_true, test_probs, auc)

    # Save
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "gnn_metrics.csv"), index=False
    )
    print(f"\n[DONE] GNN metrics saved -> artifacts/gnn_metrics.csv")
    return model, metrics


# ── Plots ─────────────────────────────────────────────────────────────────────
def _plot_gnn_loss(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss", color="steelblue")
    ax.plot(val_losses,   label="Val Loss",   color="#E74C3C")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("GNN Training Loss", fontsize=12, fontweight="bold")
    ax.legend(); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gnn_training_loss.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved -> {path}")


def _plot_gnn_confusion(y_true, y_pred):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — GNN", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cm_GNN.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved -> {path}")


def _plot_gnn_roc(y_true, probs, auc):
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="purple", lw=2,
            label=f"GNN (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — GNN", fontsize=12, fontweight="bold")
    ax.legend(); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_gnn.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Saved -> {path}")


if __name__ == "__main__":
    model, metrics = train_gnn()
    print("\nFinal GNN Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
