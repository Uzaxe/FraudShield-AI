"""
graph_viz.py
============
Generates an interactive HTML transaction graph visualization.

- Nodes = transactions (blue = legit, red = fraud)
- Edges = similarity connections between transactions
- Node size = fraud probability from best_model

Used in Streamlit app for the "Graph View" tab.

Install: pip install networkx pyvis
"""

import numpy as np
import pandas as pd
import networkx as nx
import os, joblib
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp

OUTPUT_DIR = "artifacts"
GRAPH_SAMPLE = 300   # Keep small for readable visualization


def build_visual_graph(n_samples: int = GRAPH_SAMPLE) -> str:
    """
    Builds an interactive HTML graph of transactions.
    Returns path to saved HTML file.
    """
    try:
        from pyvis.network import Network
        USE_PYVIS = True
    except ImportError:
        USE_PYVIS = False
        print("[WARN] pyvis not installed — falling back to matplotlib graph")

    # Load artifacts
    model         = joblib.load(os.path.join(OUTPUT_DIR, "best_model.pkl"))
    scaler        = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    X_test        = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test        = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

    # Sample — make sure we include fraud cases
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    n_fraud   = min(len(fraud_idx), 30)
    n_legit   = min(n_samples - n_fraud, len(legit_idx))

    rng        = np.random.default_rng(42)
    sel_fraud  = rng.choice(fraud_idx, size=n_fraud, replace=False)
    sel_legit  = rng.choice(legit_idx, size=n_legit, replace=False)
    sel_idx    = np.concatenate([sel_fraud, sel_legit])
    rng.shuffle(sel_idx)

    X_sub  = X_test[sel_idx]
    y_sub  = y_test[sel_idx]
    probs  = model.predict_proba(X_sub)[:, 1]

    # Build k-NN graph
    A      = kneighbors_graph(X_sub[:, :6], n_neighbors=5,
                               mode="connectivity", include_self=False)
    A_coo  = sp.coo_matrix(A)

    # NetworkX graph
    G = nx.Graph()
    for i in range(len(X_sub)):
        fraud_prob = float(probs[i])
        G.add_node(i,
                   label=f"T{sel_idx[i]}",
                   fraud_prob=round(fraud_prob, 3),
                   is_fraud=int(y_sub[i]),
                   amount=round(float(X_sub[i, -1]), 3))

    edges = list(zip(A_coo.row.tolist(), A_coo.col.tolist()))
    G.add_edges_from(edges)

    html_path = os.path.join(OUTPUT_DIR, "transaction_graph.html")

    if USE_PYVIS:
        net = Network(height="600px", width="100%",
                      bgcolor="#0A1628", font_color="#FFFFFF")
        net.set_options("""
        {
          "nodes": {"borderWidth": 2, "shadow": true},
          "edges": {"color": {"color": "#2E5D9E"}, "smooth": false, "width": 0.5},
          "physics": {"stabilization": {"iterations": 150},
                      "barnesHut": {"gravitationalConstant": -8000}}
        }
        """)

        for node, data in G.nodes(data=True):
            prob      = data["fraud_prob"]
            is_fraud  = data["is_fraud"]
            # Color: red gradient for fraud probability
            if is_fraud == 1:
                color = "#E74C3C"
                size  = 20
                title = f"⚠️ FRAUD\nProb: {prob:.1%}\nAmount: {data['amount']}"
            elif prob > 0.3:
                color = "#F39C12"
                size  = 14
                title = f"⚠️ Suspicious\nProb: {prob:.1%}\nAmount: {data['amount']}"
            else:
                color = "#2980B9"
                size  = 10
                title = f"✅ Legit\nProb: {prob:.1%}\nAmount: {data['amount']}"

            net.add_node(node, label=f"T{node}",
                         color=color, size=size, title=title)

        for u, v in G.edges():
            net.add_edge(u, v)

        net.save_graph(html_path)
        print(f"[INFO] Interactive graph saved → {html_path}")

    else:
        # Fallback: matplotlib static graph
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        pos    = nx.spring_layout(G, seed=42, k=0.3)
        colors = ["#E74C3C" if d["is_fraud"] else
                  "#F39C12" if d["fraud_prob"] > 0.3 else "#2980B9"
                  for _, d in G.nodes(data=True)]
        sizes  = [300 if d["is_fraud"] else 100 for _, d in G.nodes(data=True)]

        fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0A1628")
        ax.set_facecolor("#0A1628")
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color="#2E5D9E", ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes,
                               alpha=0.85, ax=ax)

        patches = [
            mpatches.Patch(color="#E74C3C", label="Confirmed Fraud"),
            mpatches.Patch(color="#F39C12", label="Suspicious (>30%)"),
            mpatches.Patch(color="#2980B9", label="Legitimate"),
        ]
        ax.legend(handles=patches, loc="upper left",
                  facecolor="#142850", labelcolor="white")
        ax.set_title("Transaction Similarity Graph\n(nodes = transactions, edges = similar behaviour)",
                     color="white", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        png_path = os.path.join(OUTPUT_DIR, "transaction_graph.png")
        plt.savefig(png_path, dpi=150, facecolor="#0A1628")
        plt.close()
        print(f"[INFO] Static graph saved → {png_path}")

        # Write minimal HTML that embeds the image
        with open(html_path, "w") as f:
            f.write(f"""<html><body style='background:#0A1628;margin:0'>
            <img src='transaction_graph.png' style='width:100%'>
            </body></html>""")

    return html_path


def get_graph_stats() -> dict:
    """Returns summary statistics about the transaction graph."""
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    model  = joblib.load(os.path.join(OUTPUT_DIR, "best_model.pkl"))
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    probs  = model.predict_proba(X_test)[:, 1]

    return {
        "total_transactions": len(y_test),
        "confirmed_fraud":    int(y_test.sum()),
        "high_risk_flagged":  int((probs > 0.5).sum()),
        "medium_risk":        int(((probs > 0.3) & (probs <= 0.5)).sum()),
        "low_risk":           int((probs <= 0.3).sum()),
        "avg_fraud_prob":     float(probs.mean()),
    }


if __name__ == "__main__":
    html = build_visual_graph()
    stats = get_graph_stats()
    print("\nGraph Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
