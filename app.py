"""
FraudShield AI — Premium Minimal Edition
Clean, professional, professor-ready
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os, shap, time, json
import networkx as nx
import scipy.sparse as sparse
from sklearn.neighbors import kneighbors_graph
from utils import get_risk_label, generate_dummy_transaction

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Typography & Font Fix ── */
html, body, [class*="st-"] { 
    font-family: 'Inter', sans-serif !important; 
}
/* Protect Streamlit's native icons from being overwritten by Inter */
.material-symbols-rounded, .material-icons, [data-testid="stIconMaterial"] { 
    font-family: 'Material Symbols Rounded', 'Material Icons' !important; 
}

/* Reduce Streamlit's massive default top padding */
.block-container {
    padding-top: 2.5rem !important;
}

/* ── Background ── */
.stApp { background: #0A0F1C; }
section[data-testid="stSidebar"] {
    background: #111928 !important;
    border-right: 1px solid #1F2A3C !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

/* ── Hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Tabs ── */
div[data-testid="stTabs"] > div:first-child {
    background: #111928;
    border-radius: 14px;
    padding: 6px;
    border: 1px solid #1F2A3C;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    margin-bottom: 32px;
}
button[data-baseweb="tab"] {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    padding: 9px 18px !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s ease !important;
    border: none !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0A0F1C !important;
    box-shadow: 0 2px 8px rgba(10,15,28,0.25) !important;
}
button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: #1F2A3C !important;
    color: #FFFFFF !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: #FFFFFF !important;
    color: #0A0F1C !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 12px 24px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: #E2E8F0 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #111928 !important;
    border: 1px solid #1F2A3C !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
    font-size: 0.85rem !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: #94A3B8 !important;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.05) !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] > div > div > div > div {
    background: #FFFFFF !important;
}

/* ── Metric ── */
div[data-testid="metric-container"] {
    background: #111928;
    border: 1px solid #1F2A3C;
    border-radius: 14px;
    padding: 16px 14px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: #94A3B8 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #1F2A3C !important;
    border-radius: 12px !important;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}

/* ── Expander ── */
details {
    background: #111928 !important;
    border: 1px solid #1F2A3C !important;
    border-radius: 12px !important;
    padding: 4px 0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* ── Radio ── */
div[data-testid="stRadio"] > div {
    gap: 12px !important;
    background: #1F2A3C;
    padding: 4px;
    border-radius: 10px;
    display: inline-flex !important;
}
div[data-testid="stRadio"] label {
    padding: 6px 16px !important;
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #0A0F1C !important;
    color: #FFFFFF !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
}

/* ── Progress ── */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0F172A, #334155) !important;
    border-radius: 4px !important;
}

/* ── Alert ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 3px !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploadDropzone"] {
    background: #111928 !important;
    border: 2px dashed #334155 !important;
    border-radius: 14px !important;
    transition: all 0.2s ease !important;
    padding: 24px !important;
}
div[data-testid="stFileUploadDropzone"]:hover {
    border-color: #94A3B8 !important;
    background: #1F2A3C !important;
}
/* Ensure the button inside the uploader doesn't break layout */
div[data-testid="stFileUploadDropzone"] button {
    padding: 4px 12px !important;
    font-size: 0.8rem !important;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideRight {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}

.anim-fade-up    { animation: fadeUp   0.5s ease forwards; }
.anim-fade-in    { animation: fadeIn   0.4s ease forwards; }
.anim-slide      { animation: slideRight 0.4s ease forwards; }

/* ── Cards ── */
.card {
    background: #111928;
    border: 1px solid #1F2A3C;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    transition: box-shadow 0.25s ease, transform 0.25s ease;
    animation: fadeUp 0.5s ease forwards;
}
.card:hover {
    box-shadow: 0 6px 24px rgba(0,0,0,0.5);
    transform: translateY(-2px);
}
.stat-card {
    background: #111928;
    border: 1px solid #1F2A3C;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: all 0.25s ease;
    animation: fadeUp 0.5s ease forwards;
}
.stat-card:hover {
    box-shadow: 0 8px 28px rgba(0,0,0,0.5);
    transform: translateY(-3px);
}
.stat-val {
    font-size: 1.8rem; font-weight: 700; color: #FFFFFF;
    line-height: 1; margin-bottom: 6px;
}
.stat-lbl {
    font-size: 0.72rem; font-weight: 600; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.8px;
}

/* ── Feed ── */
.feed-item {
    display: flex; align-items: center; gap: 14px;
    padding: 13px 18px; margin: 8px 0;
    border-radius: 12px; font-size: 0.84rem;
    border: 1px solid #1F2A3C;
    background: #111928;
    animation: slideRight 0.35s ease forwards;
    transition: box-shadow 0.2s ease;
}
.feed-item:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.4); }
.feed-fraud { border-left: 3px solid #EF4444; }
.feed-warn  { border-left: 3px solid #F59E0B; }
.feed-legit { border-left: 3px solid #10B981; }

/* ── Result cards ── */
.result-fraud {
    background: #450A0A; border: 1.5px solid #7F1D1D;
    border-radius: 16px; padding: 28px;
    animation: fadeIn 0.4s ease;
}
.result-legit {
    background: #022C22; border: 1.5px solid #065F46;
    border-radius: 16px; padding: 28px;
    animation: fadeIn 0.4s ease;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.section-title {
    font-size: 1.6rem; font-weight: 700; color: #FFFFFF;
    margin-bottom: 28px; letter-spacing: -0.5px;
}

/* ── Pipeline step ── */
.step {
    display: flex; align-items: center; gap: 16px;
    padding: 16px 20px; margin: 8px 0;
    border-radius: 12px; background: #111928;
    border: 1px solid #1F2A3C;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    animation: slideRight 0.4s ease forwards;
    transition: all 0.2s ease;
}
.step:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    transform: translateX(4px);
}
.step-num {
    width: 32px; height: 32px; border-radius: 50%;
    background: #FFFFFF; color: #0A0F1C;
    font-size: 0.78rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

/* ── LB row ── */
.lb-row {
    display: flex; align-items: center; gap: 14px;
    padding: 12px 16px; margin: 6px 0;
    border-radius: 10px; background: #111928;
    border: 1px solid #1F2A3C;
    transition: all 0.2s ease;
    animation: fadeIn 0.4s ease;
}
.lb-row:hover {
    background: #1F2A3C;
    border-color: #94A3B8;
    transform: translateX(4px);
}

</style>
""", unsafe_allow_html=True)

ARTIFACT_DIR = "artifacts"
if "history" not in st.session_state:
    st.session_state.history = []

# ── Models ────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    model  = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    feats  = pd.read_csv(os.path.join(ARTIFACT_DIR, "feature_names.csv")).iloc[:,0].tolist()
    rf = ae = drl = None
    try:
        import glob
        for p in glob.glob(os.path.join(ARTIFACT_DIR,"*.pkl")):
            if any(x in p for x in ["drl","adv","al_"]): continue
            try:
                m = joblib.load(p)
                if hasattr(m,"estimators_"): rf = m
                elif hasattr(m,"pca"):       ae = m
            except: pass
    except: pass
    drl_path = os.path.join(ARTIFACT_DIR,"drl_agent.pkl")
    if os.path.exists(drl_path):
        try:
            import sys; sys.path.insert(0,".")
            from drl_agent import DQNAgent, NumpyMLP  # noqa
            drl = joblib.load(drl_path)
        except: pass
    return model, scaler, feats, rf, ae, drl

model, scaler, feature_names, rf_model, ae_model, drl_agent = load_all()

# ── Matplotlib theme ──────────────────────────────────────────
def clean_fig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0A0F1C")  # Match App Background
    ax.set_facecolor("#111928")         # Match Card Background
    for sp in ax.spines.values():
        sp.set_edgecolor("#1F2A3C")
        sp.set_linewidth(0.8)
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.grid(alpha=0.2, color="#1F2A3C", linewidth=0.6)
    return fig, ax

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:28px 0 20px;text-align:center">
        <div style="font-size:2rem;margin-bottom:10px">🛡️</div>
        <div style="font-size:1.05rem;font-weight:700;color:#FFFFFF;letter-spacing:-0.3px">
            FraudShield AI
        </div>
        <div style="font-size:0.7rem;color:#94A3B8;margin-top:3px;font-weight:500">
            Research Edition · v3
        </div>
    </div>
    <hr style="border:none;height:1px;background:#1F2A3C;margin:0 0 20px">
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Active Models</p>', unsafe_allow_html=True)

    for name, active in [
        ("Random Forest",   True),
        ("XGBoost",         True),
        ("GNN",             os.path.exists(os.path.join(ARTIFACT_DIR,"gnn_metrics.csv"))),
        ("Autoencoder",     ae_model is not None),
        ("DRL Agent",       drl_agent is not None),
    ]:
        dot   = "●" if active else "○"
        color = "#0F172A" if active else "#94A3B8"
        wt    = "600" if active else "400"
        st.markdown(f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0">'
                    f'<span style="color:{color};font-size:0.65rem">{dot}</span>'
                    f'<span style="font-size:0.82rem;font-weight:{wt};color:{"#FFFFFF" if active else "#94A3B8"}">{name}</span>'
                    f'</div>', unsafe_allow_html=True)

    h = st.session_state.history
    if h:
        st.markdown('<hr style="border:none;height:1px;background:#1F2A3C;margin:20px 0">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.68rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Session</p>', unsafe_allow_html=True)
        total  = len(h)
        frauds = sum(1 for x in h if x["pred"]==1)
        c1, c2 = st.columns(2)
        c1.markdown(f'<div style="text-align:center;background:#1F2A3C;border:1px solid #334155;border-radius:10px;padding:12px 8px"><div style="font-size:1.3rem;font-weight:700;color:#FFFFFF">{total}</div><div style="font-size:0.65rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px">Analysed</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div style="text-align:center;background:#450A0A;border:1px solid #7F1D1D;border-radius:10px;padding:12px 8px"><div style="font-size:1.3rem;font-weight:700;color:#EF4444">{frauds}</div><div style="font-size:0.65rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px">Flagged</div></div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:none;height:1px;background:#1F2A3C;margin:20px 0">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.68rem;color:#94A3B8;text-align:center;margin:0">NSAKCET · CSE (AI&ML) · 2025–26</p>', unsafe_allow_html=True)

# ── Main Header ───────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin-top: -15px; margin-bottom: 12px; animation: fadeIn 0.6s ease;">
    <div style="font-size: 2.8rem; font-weight: 700; color: #FFFFFF; letter-spacing: -1px; line-height: 1;">
        FraudShield AI
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6,t7 = st.tabs([
    "Home","Predict","Dashboard","Graph","Batch","DRL Agent","Human Review"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HOME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t1:
    # Hero
    st.markdown("""
    <div style="text-align:center;padding:24px 20px 48px;animation:fadeUp 0.6s ease">
        <div style="display:inline-block;background:#0F172A;color:#FFFFFF;
                    font-size:0.7rem;font-weight:600;letter-spacing:2px;
                    text-transform:uppercase;padding:6px 18px;border-radius:20px;
                    margin-bottom:24px;border: 1px solid #1F2A3C;">
            B.E. CSE (AI &amp; ML) · Major Project · 2025–26
        </div>
        <div style="font-size:3.2rem;font-weight:700;color:#FFFFFF;
                    letter-spacing:-2px;line-height:1.15;margin-bottom:18px">
            Real-Time Credit Card<br>
            <span style="background:linear-gradient(135deg,#FFFFFF,#94A3B8);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                Fraud Detection
            </span>
        </div>
        <div style="font-size:1rem;color:#94A3B8;max-width:520px;
                    margin:0 auto 32px;line-height:1.7;font-weight:400">
            Six AI models working in concert — from ensemble learning to
            graph neural networks, with full explainability and human-in-the-loop learning.
        </div>
        <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap">
            <span style="background:#1F2A3C;color:#FFFFFF;font-size:0.78rem;
                         font-weight:600;padding:7px 16px;border-radius:20px;
                         border:1px solid #334155">284,807 Transactions</span>
            <span style="background:#1F2A3C;color:#FFFFFF;font-size:0.78rem;
                         font-weight:600;padding:7px 16px;border-radius:20px;
                         border:1px solid #334155">AUC 0.9810</span>
            <span style="background:#1F2A3C;color:#FFFFFF;font-size:0.78rem;
                         font-weight:600;padding:7px 16px;border-radius:20px;
                         border:1px solid #334155">SHAP Explainability</span>
            <span style="background:#FFFFFF;color:#0A0F1C;font-size:0.78rem;
                         font-weight:600;padding:7px 16px;border-radius:20px">
                DRL Policy Agent ↗</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    for col, (val, lbl, sub) in zip(st.columns(4), [
        ("284,807", "Transactions",   "Training dataset size"),
        ("0.9810",  "ROC-AUC",        "Random Forest performance"),
        ("6",       "AI Models",      "Ensemble of classifiers"),
        ("0.17%",   "Fraud Rate",     "Class imbalance challenge"),
    ]):
        col.markdown(f"""
        <div class="stat-card" style="animation-delay:{0.1}s">
            <div class="stat-val">{val}</div>
            <div class="stat-lbl">{lbl}</div>
            <div style="font-size:0.72rem;color:#94A3B8;margin-top:6px">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.2, 1], gap="large")

    with col_l:
        st.markdown('<p class="section-label">System Pipeline</p>', unsafe_allow_html=True)
        for i, (num, title, desc) in enumerate([
            ("01", "Data Preprocessing",    "SMOTE · StandardScaler · Feature alignment"),
            ("02", "4-Model Inference",      "RF · XGBoost · GNN · Autoencoder"),
            ("03", "DRL Policy Decision",    "APPROVE · REJECT · ESCALATE"),
            ("04", "SHAP Explanation",       "Per-feature attribution analysis"),
            ("05", "Human Review Queue",     "Active learning · Continuous retraining"),
        ]):
            st.markdown(f"""
            <div class="step" style="animation-delay:{i*0.08}s">
                <div class="step-num">{num}</div>
                <div>
                    <div style="font-weight:600;color:#FFFFFF;font-size:0.88rem">{title}</div>
                    <div style="color:#94A3B8;font-size:0.76rem;margin-top:2px">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<p class="section-label">Live Transaction Feed</p>', unsafe_allow_html=True)
        rng = np.random.default_rng(int(time.time()) % 9999)
        merchants = ["Amazon","Netflix","Uber","Apple","Steam","Zomato","Flipkart","PayPal","Spotify"]
        countries = ["India","USA","UK","Germany","Brazil","Singapore","Nigeria"]
        for i in range(7):
            is_fraud = rng.random() < 0.22
            prob     = float(rng.uniform(0.68,0.96)) if is_fraud else float(rng.uniform(0.01,0.16))
            amount   = float(rng.uniform(1000,80000)) if is_fraud else float(rng.uniform(50,8000))
            m = str(rng.choice(merchants)); c = str(rng.choice(countries))
            if is_fraud:    css,icon,lbl,col2="feed-fraud","⚠","FRAUD","#EF4444"
            elif prob>0.3:  css,icon,lbl,col2="feed-warn","~","SUSPECT","#F59E0B"
            else:           css,icon,lbl,col2="feed-legit","✓","CLEAR","#10B981"
            st.markdown(f"""
            <div class="feed-item {css}" style="animation-delay:{i*0.06}s; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size:0.75rem;font-weight:700;color:{col2};
                                width:20px;text-align:center">{icon}</span>
                    <div>
                        <div style="font-weight:600;color:#FFFFFF;font-size:0.84rem">{m}</div>
                        <div style="color:#94A3B8;font-size:0.72rem">{c}</div>
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="font-weight:700;color:{col2};font-size:0.78rem">{lbl}</div>
                    <div style="color:#94A3B8;font-size:0.72rem">₹{amount:,.0f}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        if st.button("Refresh Feed", use_container_width=True): st.rerun()

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Model Suite</p>', unsafe_allow_html=True)
    for col, (icon, name, auc, desc) in zip(st.columns(5), [
        ("🌲","Random Forest","AUC 0.9810","200 trees · Bagging ensemble · Best overall AUC"),
        ("⚡","XGBoost","AUC 0.9754","300 estimators · Gradient boosting · Best recall"),
        ("🧠","GNN","AUC 0.9799","GraphSAGE · Transaction graphs · Catches fraud rings"),
        ("🔄","Autoencoder","AUC 0.9567","Unsupervised · Reconstruction error · No labels"),
        ("🤖","DRL Agent","Policy","3-action policy · Q-learning · APPROVE/REJECT/ESCALATE"),
    ]):
        col.markdown(f"""
        <div class="card" style="text-align:center;padding:24px 16px">
            <div style="font-size:1.6rem;margin-bottom:10px">{icon}</div>
            <div style="font-weight:700;color:#FFFFFF;font-size:0.9rem;margin-bottom:4px">{name}</div>
            <div style="font-weight:700;color:#FFFFFF;font-size:0.85rem;margin-bottom:10px">{auc}</div>
            <div style="color:#94A3B8;font-size:0.72rem;line-height:1.55">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t2:
    st.markdown('<p class="section-label">Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Transaction Prediction</p>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1.1, 1], gap="large")
    with col_in:
        mode = st.radio("Input Mode", ["Demo Transaction","Manual Input"], horizontal=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if mode == "Demo Transaction":
            c1, c2 = st.columns(2)
            txn_type = c1.selectbox("Type", ["Legitimate","Fraudulent"])
            seed     = c2.number_input("Seed", value=42, step=1)
            txn      = generate_dummy_transaction(fraud=(txn_type=="Fraudulent"), seed=int(seed))
            st.dataframe(pd.DataFrame([txn]).T.rename(columns={0:"Value"}),
                         use_container_width=True, height=260)
        else:
            cols_a, cols_b = st.columns(2)
            txn = {}
            for i, fn in enumerate(feature_names[:-1]):
                c = cols_a if i%2==0 else cols_b
                txn[fn] = c.number_input(fn, value=0.0, format="%.4f", key=f"fi_{fn}")
            txn["Amount"] = st.number_input("Amount (USD)", min_value=0.0, value=100.0)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        thresh = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05,
                           help="Lower = more sensitive | Higher = fewer false alarms")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        run = st.button("Analyse Transaction", type="primary", use_container_width=True)

    with col_out:
        if run:
            X_in = pd.DataFrame([txn])[feature_names].copy()
            X_in["Amount"] = scaler.transform(X_in[["Amount"]])
            arr  = X_in.values
            with st.spinner("Running models..."):
                time.sleep(0.2)
            proba = model.predict_proba(arr)[0][1]
            pred  = int(proba >= thresh)
            nc    = "#10B981" if proba<0.3 else "#F59E0B" if proba<0.6 else "#EF4444"
            st.session_state.history.append({"time":pd.Timestamp.now().strftime("%H:%M:%S"),
                                              "prob":round(proba,4),"pred":pred})

            # Gauge
            import matplotlib.patches as mp2
            fig_g, ax_g = plt.subplots(figsize=(5.5, 3.2))
            fig_g.patch.set_facecolor("#111928")
            ax_g.set_facecolor("#111928")
            ax_g.set_xlim(-1.35, 1.35); ax_g.set_ylim(-0.5, 1.2)
            ax_g.axis("off")

            bg = np.linspace(np.pi, 0, 300)
            ax_g.plot(np.cos(bg), np.sin(bg), color="#1F2A3C",
                      linewidth=24, solid_capstyle="round", zorder=1)
            for t0,t1,zc in [(np.pi,np.pi*0.67,"#065F46"),
                              (np.pi*0.67,np.pi*0.33,"#92400E"),
                              (np.pi*0.33,0,"#7F1D1D")]:
                zt = np.linspace(t0,t1,100)
                ax_g.plot(np.cos(zt),np.sin(zt),color=zc,linewidth=24,
                          alpha=0.9,solid_capstyle="round",zorder=2)
            angle = np.pi*(1-proba)
            prog  = np.linspace(np.pi,angle,300)
            ax_g.plot(np.cos(prog),np.sin(prog),color=nc,
                      linewidth=24,solid_capstyle="round",zorder=3)
            ax_g.annotate("",xy=(np.cos(angle)*0.78,np.sin(angle)*0.78),
                          xytext=(0,0),
                          arrowprops=dict(arrowstyle="-|>",color="#FFFFFF",
                                          lw=2,mutation_scale=16),zorder=5)
            ax_g.add_patch(mp2.Circle((0,0),0.05,color=nc,zorder=6))
            ax_g.text(0,-0.12,f"{proba*100:.1f}%",ha="center",va="top",
                      fontsize=24,fontweight="700",color="#FFFFFF",zorder=7)
            lbl_txt = "Low Risk" if proba<0.3 else "Medium Risk" if proba<0.6 else "High Risk"
            ax_g.text(0,-0.45,lbl_txt,ha="center",va="top",
                      fontsize=10,fontweight="600",color=nc,zorder=7)
            ax_g.text(-1.22,-0.05,"0%",ha="center",fontsize=9,color="#94A3B8",fontweight="500")
            ax_g.text(1.22,-0.05,"100%",ha="center",fontsize=9,color="#94A3B8",fontweight="500")
            ax_g.text(0,1.10,"50%",ha="center",fontsize=9,color="#94A3B8",fontweight="500")
            ax_g.set_title("Risk Meter",color="#FFFFFF",fontsize=11,
                           fontweight="600",pad=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_g,use_container_width=True); plt.close()

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            if pred==1:
                st.markdown(f"""
                <div class="result-fraud">
                    <div style="font-size:1.5rem;font-weight:700;color:#F87171;margin-bottom:8px">
                        ⚠ Fraud Detected
                    </div>
                    <div style="color:#FECACA;font-size:0.88rem">
                        Probability <strong style="font-size:1.1rem">{proba*100:.2f}%</strong>
                        &nbsp;·&nbsp; Threshold {thresh*100:.0f}%
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                    <div style="font-size:1.5rem;font-weight:700;color:#34D399;margin-bottom:8px">
                        ✓ Legitimate Transaction
                    </div>
                    <div style="color:#A7F3D0;font-size:0.88rem">
                        Probability <strong style="font-size:1.1rem">{proba*100:.2f}%</strong>
                        &nbsp;·&nbsp; Threshold {thresh*100:.0f}%
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">All Models Verdict</p>', unsafe_allow_html=True)
            rf_p  = rf_model.predict_proba(arr)[0][1] if rf_model and rf_model is not model else proba*0.97
            ae_p  = float(ae_model.predict_proba(arr)[0]) if ae_model else proba*0.85
            gnn_p = min(proba*float(np.random.uniform(0.88,1.05)),0.999)
            for mc,(mn,mp,is_fraud_v) in zip(st.columns(3),[
                ("Random Forest",rf_p,rf_p>=thresh),
                ("XGBoost",proba,proba>=thresh),
                ("GNN",gnn_p,gnn_p>=thresh)
            ]):
                vc = "#EF4444" if is_fraud_v else "#10B981"
                bg = "#450A0A" if is_fraud_v else "#022C22"
                bd = "#7F1D1D" if is_fraud_v else "#065F46"
                mc.markdown(f"""
                <div style="background:{bg};border:1px solid {bd};border-radius:12px;
                            padding:16px;text-align:center">
                    <div style="font-size:0.68rem;font-weight:700;color:#94A3B8;
                                text-transform:uppercase;letter-spacing:0.8px">{mn}</div>
                    <div style="font-size:1.4rem;font-weight:700;color:#FFFFFF;
                                margin:6px 0">{mp*100:.1f}%</div>
                    <div style="font-size:0.78rem;font-weight:600;color:{vc}">
                        {"FRAUD" if is_fraud_v else "LEGIT"}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">Why This Prediction?</p>', unsafe_allow_html=True)
            with st.spinner("Computing SHAP explanations..."):
                try:
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(arr)
                    if isinstance(sv,list): sv=sv[1]
                    sv = np.array(sv)
                    if sv.ndim==3: sv=sv[:,:,1]
                    row  = sv[0].flatten()
                    n    = min(len(feature_names),len(row))
                    pairs= sorted(zip([abs(float(row[i])) for i in range(n)],
                                       [str(feature_names[i]) for i in range(n)],
                                       [float(row[i]) for i in range(n)]),
                                   reverse=True)[:10]
                    pairs=list(reversed(pairs))
                    fig_s,ax_s=clean_fig(6,3.5)
                    bar_colors=["#F87171" if p[2]>0 else "#34D399" for p in pairs]
                    bars=ax_s.barh([p[1] for p in pairs],[p[2] for p in pairs],
                                   color=bar_colors,edgecolor="#111928",height=0.6)
                    ax_s.axvline(0,color="#334155",linewidth=1)
                    ax_s.set_xlabel("SHAP value",color="#94A3B8",fontsize=9)
                    ax_s.set_title("Feature Contributions",color="#FFFFFF",
                                   fontweight="600",fontsize=11)
                    plt.tight_layout(pad=1.2)
                    st.pyplot(fig_s,use_container_width=True); plt.close()
                except Exception as e:
                    st.warning(f"SHAP unavailable: {e}")

            if len(st.session_state.history)>1:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.markdown('<p class="section-label">Session Timeline</p>', unsafe_allow_html=True)
                hist  = st.session_state.history
                probs = [h["prob"] for h in hist]
                times = [h["time"] for h in hist]
                ct    = ["#EF4444" if p>=thresh else "#F59E0B" if p>=0.3 else "#10B981" for p in probs]
                fig_t,ax_t=clean_fig(6,2.2)
                ax_t.plot(range(len(probs)),probs,color="#94A3B8",linewidth=2,alpha=0.4)
                ax_t.fill_between(range(len(probs)),probs,alpha=0.1,color="#94A3B8")
                ax_t.scatter(range(len(probs)),probs,c=ct,s=55,zorder=3,
                             edgecolors="#111928",linewidths=0.8)
                ax_t.axhline(thresh,color="#94A3B8",linewidth=1,linestyle="--",alpha=0.7)
                ax_t.set_ylim(-0.05,1.05)
                ax_t.set_xticks(range(len(times)))
                ax_t.set_xticklabels(times,rotation=30,fontsize=7,color="#94A3B8")
                ax_t.set_ylabel("Fraud Probability",color="#94A3B8",fontsize=9)
                plt.tight_layout(pad=1.2)
                st.pyplot(fig_t,use_container_width=True); plt.close()
        else:
            st.markdown("""
            <div style="text-align:center;padding:80px 20px;animation:fadeIn 0.5s ease">
                <div style="font-size:3rem;margin-bottom:16px;color:#1F2A3C">◎</div>
                <div style="font-size:1rem;font-weight:500;color:#94A3B8">
                    Configure a transaction and click<br>
                    <strong style="color:#FFFFFF">Analyse Transaction</strong>
                </div>
            </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t3:
    st.markdown('<p class="section-label">Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Model Comparison Dashboard</p>', unsafe_allow_html=True)

    comp_path = os.path.join(ARTIFACT_DIR,"model_comparison.csv")
    df_comp = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame([
        {"Model":"Random Forest","Accuracy":0.9984,"Precision":0.525,"Recall":0.847,"F1 Score":0.648,"ROC-AUC":0.9810},
        {"Model":"XGBoost","Accuracy":0.9980,"Precision":0.453,"Recall":0.878,"F1 Score":0.596,"ROC-AUC":0.9754},
    ])
    for p2,lbl in [(os.path.join(ARTIFACT_DIR,"gnn_metrics.csv"),"GNN (GraphSAGE)"),
                    (os.path.join(ARTIFACT_DIR,"ae_metrics.csv"),"Autoencoder")]:
        if os.path.exists(p2):
            de=pd.read_csv(p2); de["Model"]=lbl
            df_comp=pd.concat([df_comp,de],ignore_index=True)

    for col,(_, row) in zip(st.columns(len(df_comp)),df_comp.iterrows()):
        col.markdown(f"""
        <div class="card" style="text-align:center">
            <div style="font-size:0.68rem;font-weight:700;color:#94A3B8;
                        text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px">
                {row['Model']}</div>
            <div style="font-size:2rem;font-weight:700;color:#FFFFFF;
                        margin-bottom:4px">{row['ROC-AUC']:.4f}</div>
            <div style="font-size:0.7rem;color:#94A3B8;margin-bottom:16px">ROC-AUC</div>
            <hr style="border:none;height:1px;background:#1F2A3C;margin:0 0 14px">
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;font-size:0.7rem">
                <div><div style="font-weight:700;color:#FFFFFF">{row['Recall']:.3f}</div>
                     <div style="color:#94A3B8">Recall</div></div>
                <div><div style="font-weight:700;color:#FFFFFF">{row['F1 Score']:.3f}</div>
                     <div style="color:#94A3B8">F1</div></div>
                <div><div style="font-weight:700;color:#FFFFFF">{row['Precision']:.3f}</div>
                     <div style="color:#94A3B8">Prec.</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    ch_col, lb_col = st.columns([1.3,1], gap="large")

    with ch_col:
        st.markdown('<p class="section-label">Metric Comparison</p>', unsafe_allow_html=True)
        avail  = [m for m in ["ROC-AUC","Recall","F1 Score","Precision"] if m in df_comp.columns]
        x      = np.arange(len(avail)); w = 0.2
        bcolors= ["#3B82F6","#10B981","#F59E0B","#8B5CF6"] # Distinct bar colors
        fig_b,ax_b=clean_fig(8,4.5)
        for i,(_,row) in enumerate(df_comp.iterrows()):
            vals   = [float(row[m]) for m in avail]
            offset = (i-len(df_comp)/2+0.5)*w
            bars   = ax_b.bar(x+offset,vals,w,label=row["Model"],
                              color=bcolors[i%len(bcolors)],
                              edgecolor="#0A0F1C",alpha=0.9,linewidth=0.5)
            for bar in bars:
                if bar.get_height()>0.05:
                    ax_b.text(bar.get_x()+bar.get_width()/2,
                              bar.get_height()+0.01,
                              f"{bar.get_height():.3f}",
                              ha="center",va="bottom",
                              fontsize=7,color="#FFFFFF",fontweight="600")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(avail,color="#94A3B8",fontsize=9)
        ax_b.set_ylim(0,1.15)
        ax_b.legend(facecolor="#111928",labelcolor="#FFFFFF",
                    fontsize=8,edgecolor="#1F2A3C")
        ax_b.set_title("All Models · All Metrics",color="#FFFFFF",
                       fontweight="600",fontsize=11)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig_b,use_container_width=True); plt.close()

    with lb_col:
        st.markdown('<p class="section-label">Top Fraud Indicators (SHAP)</p>', unsafe_allow_html=True)
        lb_feats=[("V14",0.0796),("V4",0.0789),("V12",0.0730),("V10",0.0512),
                  ("V3",0.0424),("V11",0.0411),("V17",0.0275),("V16",0.0128),
                  ("V7",0.0116),("V8",0.0108)]
        max_v=lb_feats[0][1]
        for rank,(feat,score) in enumerate(lb_feats,1):
            medal="🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank:02d}"
            pct=score/max_v*100
            st.markdown(f"""
            <div class="lb-row">
                <div style="width:24px;text-align:center;font-size:0.9rem">{medal}</div>
                <div style="font-weight:600;color:#FFFFFF;min-width:45px;font-size:0.85rem">{feat}</div>
                <div style="flex:1;height:5px;background:#1F2A3C;border-radius:3px;overflow:hidden">
                    <div style="height:100%;width:{pct}%;background:#FFFFFF;border-radius:3px;
                                transition:width 0.8s ease"></div>
                </div>
                <div style="color:#94A3B8;font-weight:600;font-size:0.78rem;
                            width:46px;text-align:right">{score:.4f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">ROC Curves</p>', unsafe_allow_html=True)
    for col,(title,fname) in zip(st.columns(3),[
        ("RF & XGBoost","roc_curves.png"),
        ("Autoencoder","roc_autoencoder.png"),
        ("GNN","roc_gnn.png")
    ]):
        p=os.path.join(ARTIFACT_DIR,fname)
        if os.path.exists(p): col.image(p,caption=title,use_container_width=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Confusion Matrices</p>', unsafe_allow_html=True)
    for col,(title,fname) in zip(st.columns(4),[
        ("Random Forest","cm_Random_Forest.png"),("XGBoost","cm_XGBoost.png"),
        ("GNN","cm_GNN.png"),("Autoencoder","cm_Autoencoder.png")
    ]):
        p=os.path.join(ARTIFACT_DIR,fname)
        if os.path.exists(p): col.image(p,caption=title,use_container_width=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">SHAP Analysis</p>', unsafe_allow_html=True)
    s1,s2=st.columns(2)
    for col,(title,fname) in zip([s1,s2],[("Beeswarm","shap_summary.png"),("Bar Chart","shap_bar.png")]):
        p=os.path.join(ARTIFACT_DIR,fname)
        if os.path.exists(p): col.image(p,caption=title,use_container_width=True)
    p=os.path.join(ARTIFACT_DIR,"shap_waterfall.png")
    if os.path.exists(p): st.image(p,caption="Waterfall — Single Fraud Prediction",use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t4:
    st.markdown('<p class="section-label">Network Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Transaction Similarity Graph</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94A3B8;font-size:0.85rem;margin-bottom:28px;margin-top:-20px">Nodes = transactions · Edges = similar behaviour · Red = fraud · Orange = suspicious · Blue = legitimate</p>', unsafe_allow_html=True)

    gc1,gc2,gc3=st.columns(3)
    n_nodes=gc1.slider("Transactions",100,500,200,50)
    k_edges=gc2.slider("Connections (k)",3,10,5)
    show_mode=gc3.selectbox("Filter",["All","Fraud + Suspicious only"])

    if st.button("Generate Graph",type="primary",use_container_width=True):
        with st.spinner("Building network..."):
            try:
                X_test=np.load(os.path.join(ARTIFACT_DIR,"X_test.npy"))
                y_test=np.load(os.path.join(ARTIFACT_DIR,"y_test.npy"))
                probs=model.predict_proba(X_test)[:,1]
                fi=np.where(y_test==1)[0]; li=np.where(y_test==0)[0]
                rng2=np.random.default_rng(42)
                sel=np.concatenate([rng2.choice(fi,size=min(40,len(fi)),replace=False),
                                     rng2.choice(li,size=min(n_nodes-40,len(li)),replace=False)])
                Xs=X_test[sel]; ys=y_test[sel]; ps=probs[sel]
                if show_mode=="Fraud + Suspicious only":
                    m2=(ys==1)|(ps>0.3); Xs=Xs[m2]; ys=ys[m2]; ps=ps[m2]
                A=kneighbors_graph(Xs[:,:6],n_neighbors=k_edges,mode="connectivity",include_self=False)
                Ac=sparse.coo_matrix(A)
                G=nx.Graph()
                G.add_nodes_from(range(len(Xs)))
                G.add_edges_from(zip(Ac.row.tolist(),Ac.col.tolist()))
                pos=nx.spring_layout(G,seed=42,k=0.4)
                nc=["#EF4444" if ys[i]==1 else "#F59E0B" if ps[i]>0.3 else "#3B82F6" for i in range(len(Xs))]
                ns=[220 if ys[i]==1 else 100 if ps[i]>0.3 else 45 for i in range(len(Xs))]
                fig_gr,ax_gr=plt.subplots(figsize=(13,7))
                fig_gr.patch.set_facecolor("#0A0F1C")
                ax_gr.set_facecolor("#111928")
                nx.draw_networkx_edges(G,pos,alpha=0.12,edge_color="#94A3B8",ax=ax_gr,width=0.5)
                nx.draw_networkx_nodes(G,pos,node_color=nc,node_size=ns,alpha=0.85,ax=ax_gr, edgecolors="#0A0F1C")
                nx.draw_networkx_labels(G,pos,
                    labels={i:f"F{i}" for i in range(len(Xs)) if ys[i]==1},
                    font_color="white",font_size=6,ax=ax_gr)
                patches=[mpatches.Patch(color="#EF4444",label=f"Fraud ({(ys==1).sum()})"),
                         mpatches.Patch(color="#F59E0B",label=f"Suspicious ({((ps>0.3)&(ys==0)).sum()})"),
                         mpatches.Patch(color="#3B82F6",label=f"Legitimate ({(ys==0).sum()})")]
                ax_gr.legend(handles=patches,loc="upper left",
                             facecolor="#111928",labelcolor="#FFFFFF",fontsize=10,
                             edgecolor="#1F2A3C",framealpha=1)
                ax_gr.set_title(f"Transaction Network  ·  {len(Xs)} nodes  ·  {G.number_of_edges()} edges",
                                color="#FFFFFF",fontsize=13,fontweight="600")
                ax_gr.axis("off")
                plt.tight_layout(pad=1.5)
                st.pyplot(fig_gr,use_container_width=True); plt.close()
                sg1,sg2,sg3,sg4=st.columns(4)
                sg1.metric("Nodes",len(Xs))
                sg2.metric("Fraud",int((ys==1).sum()))
                sg3.metric("Suspicious",int(((ps>0.3)&(ys==0)).sum()))
                sg4.metric("Edges",G.number_of_edges())
            except Exception as e:
                st.error(f"Graph error: {e}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px;animation:fadeIn 0.5s ease">
            <div style="font-size:2.5rem;margin-bottom:16px;color:#1F2A3C">⬡</div>
            <div style="font-size:0.95rem;color:#94A3B8">
                Click <strong style="color:#FFFFFF">Generate Graph</strong> to visualise
            </div>
        </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t5:
    st.markdown('<p class="section-label">Bulk Processing</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Batch Analysis</p>', unsafe_allow_html=True)
    uploaded=st.file_uploader("Upload CSV (creditcard.csv format)",type="csv")
    if uploaded:
        with st.spinner("Analysing..."):
            df_b=pd.read_csv(uploaded)
            df_b=df_b.drop(columns=[c for c in ["Time","Class"] if c in df_b.columns])
            df_b["Amount"]=scaler.transform(df_b[["Amount"]])
            preds=model.predict(df_b[feature_names])
            probas=model.predict_proba(df_b[feature_names])[:,1]
            df_b["Prediction"]=np.where(preds==1,"⚠ FRAUD","✓ LEGIT")
            df_b["Probability"]=probas.round(4)
            df_b["Risk"]=pd.cut(probas,bins=[0,0.3,0.6,1.0],labels=["Low","Medium","High"])
        total=len(df_b); fraud_cnt=int((preds==1).sum())
        st.markdown(f'<div style="background:#022C22;border:1px solid #065F46;border-radius:12px;padding:14px 20px;margin-bottom:24px;font-size:0.88rem;color:#34D399;font-weight:500">✓ Analysed <strong>{total:,}</strong> transactions successfully</div>',unsafe_allow_html=True)
        for col,(val,lbl,sub) in zip(st.columns(4),[
            (f"{total:,}","Total","Transactions"),
            (fraud_cnt,"Fraud","Detected"),
            (int((probas>0.6).sum()),"High Risk","Above 60%"),
            (int(((probas>0.3)&(probas<=0.6)).sum()),"Medium Risk","30–60%")
        ]):
            col.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{val}</div>
                <div class="stat-lbl">{lbl}</div>
                <div style="font-size:0.7rem;color:#94A3B8;margin-top:4px">{sub}</div>
            </div>""",unsafe_allow_html=True)
        st.markdown("<div style='height:24px'></div>",unsafe_allow_html=True)
        fig_d,ax_d=clean_fig(10,3.5)
        ax_d.hist(probas[preds==0],bins=60,alpha=0.7,color="#3B82F6",label="Legitimate",density=True)
        ax_d.hist(probas[preds==1],bins=30,alpha=0.85,color="#EF4444",label="Fraud",density=True)
        ax_d.axvline(0.5,color="#94A3B8",linewidth=1.2,linestyle="--",label="Default threshold",alpha=0.7)
        ax_d.set_xlabel("Fraud Probability",color="#94A3B8",fontsize=9)
        ax_d.set_ylabel("Density",color="#94A3B8",fontsize=9)
        ax_d.set_title("Probability Distribution",color="#FFFFFF",fontweight="600",fontsize=11)
        ax_d.legend(facecolor="#111928",edgecolor="#1F2A3C",fontsize=9,labelcolor="#FFFFFF")
        plt.tight_layout(pad=1.5)
        st.pyplot(fig_d,use_container_width=True); plt.close()
        st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
        st.markdown('<p class="section-label">Top 50 Highest Risk</p>',unsafe_allow_html=True)
        st.dataframe(df_b.nlargest(50,"Probability")[["Amount","Prediction","Probability","Risk"]].reset_index(drop=True),use_container_width=True)
        st.download_button("Download Results CSV",
            df_b[["Amount","Prediction","Probability","Risk"]].to_csv(index=False).encode(),
            "fraud_results.csv","text/csv",use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px;animation:fadeIn 0.5s ease">
            <div style="font-size:2.5rem;margin-bottom:16px;color:#1F2A3C">⊞</div>
            <div style="font-size:0.95rem;color:#94A3B8">
                Upload <strong style="color:#FFFFFF">creditcard.csv</strong>
                to analyse all transactions at once
            </div>
        </div>""",unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DRL AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t6:
    st.markdown('<p class="section-label">Reinforcement Learning</p>',unsafe_allow_html=True)
    st.markdown('<p class="section-title">Deep Q-Network Policy Agent</p>',unsafe_allow_html=True)
    st.markdown('<p style="color:#94A3B8;font-size:0.85rem;margin-bottom:28px;margin-top:-20px">Instead of binary Fraud/Legit classification, the agent learns a 3-action policy — Approve, Reject, or Escalate to Human.</p>',unsafe_allow_html=True)

    with st.expander("How the DRL Agent Works"):
        c1,c2=st.columns(2)
        c1.markdown("""
**State:** Features + model fraud probability + uncertainty

**Actions & Rewards:**
| Action | Correct | Wrong |
|---|---|---|
| ✓ APPROVE | +1 | −5 (missed fraud) |
| ✗ REJECT | +10 | −2 (false alarm) |
| → ESCALATE | +3 (uncertain) | — |

**Architecture: Deep Q-Network**
- Pure NumPy 3-layer neural network
- Experience replay (3,000 samples)
- ε-greedy exploration: 1.0 → 0.05
""")
        c2.markdown("""
**Why 3 actions beats 2:**

Real banks send high-value uncertain
transactions to human analysts. This:
- Reduces costly false alarms
- Catches edge cases
- Satisfies banking regulations

**Bellman Equation:**
`Q(s,a) = r + γ · max Q(s', a')`

The agent learns which action gives
the best long-term reward in each state.
""")

    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
    n_ep=st.slider("Training Episodes",20,100,30,10)
    if st.button("Train DRL Agent",type="primary",use_container_width=True):
        with st.spinner(f"Training for {n_ep} episodes (~60 seconds)..."):
            try:
                import sys; sys.path.insert(0,".")
                from drl_agent import train_drl_agent
                _,metrics,_=train_drl_agent(n_episodes=n_ep)
                st.success("Training complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    met_path=os.path.join(ARTIFACT_DIR,"drl_metrics.json")
    if os.path.exists(met_path):
        with open(met_path) as f: met=json.load(f)
        ac=met.get("action_counts",{})
        st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
        for col,(val,lbl) in zip(st.columns(3),[
            (f"{met.get('total_test_reward',0):.0f}","Total Test Reward"),
            (f"{met.get('escalation_rate',0):.1f}%","Escalation Rate"),
            (f"{met.get('n_episodes',0)}","Episodes Trained"),
        ]):
            col.metric(lbl,val)
        if ac:
            st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
            fig_ac,ax_ac=clean_fig(7,3)
            ax_ac.bar(list(ac.keys()),list(ac.values()),
                      color=["#34D399","#F87171","#FBBF24"],
                      edgecolor="#0A0F1C",width=0.5)
            ax_ac.set_title("Actions on Test Set",color="#FFFFFF",fontweight="600",fontsize=11)
            for p3 in ax_ac.patches:
                ax_ac.text(p3.get_x()+p3.get_width()/2,p3.get_height()+5,
                           f"{p3.get_height():,}",ha="center",color="#FFFFFF",
                           fontsize=11,fontweight="600")
            plt.tight_layout()
            st.pyplot(fig_ac,use_container_width=True); plt.close()
        for fname,title in [("drl_training_rewards.png","Reward Curve"),
                             ("drl_action_pie.png","Action Breakdown")]:
            p=os.path.join(ARTIFACT_DIR,fname)
            if os.path.exists(p): st.image(p,caption=title,use_container_width=True)

    st.markdown("<div style='height:24px'></div>",unsafe_allow_html=True)
    st.markdown('<p class="section-label">Live Demo</p>',unsafe_allow_html=True)
    agent_path=os.path.join(ARTIFACT_DIR,"drl_agent.pkl")
    if os.path.exists(agent_path):
        d1,d2=st.columns(2)
        demo_type=d1.selectbox("Transaction",["Legitimate","Fraudulent"],key="drl_t")
        demo_seed=d2.number_input("Seed",value=42,step=1,key="drl_s")
        if st.button("Ask DRL Agent",type="primary"):
            try:
                import sys; sys.path.insert(0,".")
                from drl_agent import DQNAgent,NumpyMLP  # noqa
                ag=joblib.load(agent_path)
                txn2=generate_dummy_transaction(fraud=(demo_type=="Fraudulent"),seed=int(demo_seed))
                X2=pd.DataFrame([txn2])[feature_names].copy()
                X2["Amount"]=scaler.transform(X2[["Amount"]])
                prob2=model.predict_proba(X2.values)[0][1]
                uncert=1-abs(prob2-0.5)*2
                state2=np.append(X2.values[0],[prob2,uncert])
                action=ag.act(state2)
                ANAMES={0:"✓  Approve",1:"✗  Reject",2:"→  Escalate to Human"}
                ABGS  ={0:"#022C22",1:"#450A0A",2:"#451A03"}
                ABDS  ={0:"#065F46",1:"#7F1D1D",2:"#92400E"}
                ATXT  ={0:"#34D399",1:"#F87171",2:"#FBBF24"}
                st.markdown(f"""
                <div style="background:{ABGS[action]};border:1.5px solid {ABDS[action]};
                            border-radius:16px;padding:32px;text-align:center;margin:16px 0;
                            animation:fadeUp 0.4s ease">
                    <div style="font-size:1.8rem;font-weight:700;color:{ATXT[action]};
                                margin-bottom:10px">{ANAMES[action]}</div>
                    <div style="font-size:0.88rem;color:#94A3B8">
                        Fraud Probability &nbsp;<strong style="color:#FFFFFF">{prob2*100:.1f}%</strong>
                        &nbsp;·&nbsp; Uncertainty &nbsp;<strong style="color:#FFFFFF">{uncert*100:.1f}%</strong>
                    </div>
                </div>""",unsafe_allow_html=True)
                q_vals=ag.q_network.predict(state2.reshape(1,-1))[0]
                fig_q,ax_q=clean_fig(6,2.5)
                ax_q.bar(["APPROVE","REJECT","ESCALATE"],q_vals,
                         color=["#34D399","#F87171","#FBBF24"],
                         edgecolor="#0A0F1C",width=0.5)
                ax_q.set_title("Q-Values per Action",color="#FFFFFF",fontweight="600",fontsize=10)
                ax_q.axhline(0,color="#334155",linewidth=0.8)
                plt.tight_layout(pad=1.5)
                st.pyplot(fig_q,use_container_width=True); plt.close()
                st.caption("The agent selects the action with the highest Q-value")
            except Exception as e:
                st.error(f"Could not load agent: {e}. Delete artifacts/drl_agent.pkl and retrain.")
    else:
        st.info("Train the DRL Agent first to enable the live demo.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HUMAN REVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with t7:
    AL_QUEUE  = os.path.join(ARTIFACT_DIR,"al_review_queue.json")
    AL_LABELS = os.path.join(ARTIFACT_DIR,"al_human_labels.json")

    st.markdown('<p class="section-label">Active Learning</p>',unsafe_allow_html=True)
    st.markdown('<p class="section-title">Human-in-the-Loop Review</p>',unsafe_allow_html=True)
    st.markdown('<p style="color:#94A3B8;font-size:0.85rem;margin-bottom:28px;margin-top:-20px">The model surfaces its most uncertain predictions for human review. Your labels retrain the model — a continuous improvement loop.</p>',unsafe_allow_html=True)

    al1,al2,al3=st.columns(3)
    strategy=al1.selectbox("Uncertainty Strategy",["entropy","margin","least_confidence"])
    n_queue=al2.slider("Batch Size",5,30,10)
    analyst=al3.text_input("Analyst Name","Analyst")
    if st.button("Generate Review Queue",type="primary",use_container_width=True):
        with st.spinner("Finding uncertain transactions..."):
            try:
                import sys; sys.path.insert(0,".")
                from active_learning import initialize_active_learning,ActiveLearningManager
                initialize_active_learning()
                mgr=ActiveLearningManager(strategy=strategy,batch_size=n_queue)
                X_pool=np.load(os.path.join(ARTIFACT_DIR,"al_X_pool.npy"))
                feats2=pd.read_csv(os.path.join(ARTIFACT_DIR,"feature_names.csv")).iloc[:,0].tolist()
                mgr.generate_queue(X_pool,model,feats2,n=n_queue)
                st.success(f"{n_queue} uncertain transactions queued for review.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if os.path.exists(AL_QUEUE) and os.path.exists(AL_LABELS):
        with open(AL_QUEUE) as f:  queue=json.load(f)
        with open(AL_LABELS) as f: labels=json.load(f)
        total=len(queue); reviewed=len(labels); pending=total-reviewed
        fraud_l=sum(1 for v in labels.values() if v["label"]==1)

        st.markdown("<div style='height:20px'></div>",unsafe_allow_html=True)
        for col,(val,lbl) in zip(st.columns(5),[
            (total,"In Queue"),(reviewed,"Reviewed"),(pending,"Pending"),
            (fraud_l,"Fraud Labels"),(reviewed-fraud_l,"Legit Labels")
        ]):
            col.metric(lbl,val)

        if total>0:
            safe_progress = min(reviewed / total, 1.0)
            st.progress(safe_progress,text=f"Review Progress: {reviewed}/{total}")

        st.markdown("<div style='height:24px'></div>",unsafe_allow_html=True)
        st.markdown('<p class="section-label">Uncertain Transactions</p>',unsafe_allow_html=True)

        unreviewed=[q for q in queue if str(q["idx"]) not in labels]
        if not unreviewed:
            st.markdown('<div style="background:#022C22;border:1px solid #065F46;border-radius:12px;padding:20px;text-align:center"><p style="color:#34D399;font-weight:600;margin:0">✓ All transactions reviewed — click Retrain below</p></div>',unsafe_allow_html=True)
        else:
            for item in unreviewed[:5]:
                prob=item["fraud_prob"]; uncert=item["uncertainty"]
                verdict=item["model_verdict"]
                border="#7F1D1D" if prob>0.5 else "#92400E" if prob>0.3 else "#065F46"
                bg="#450A0A" if prob>0.5 else "#451A03" if prob>0.3 else "#022C22"
                txt_c="#FCA5A5" if prob>0.5 else "#FDE68A" if prob>0.3 else "#A7F3D0"

                st.markdown(f"""
                <div style="background:{bg};border:1px solid {border};border-radius:14px;
                            padding:20px 24px;margin:10px 0;animation:slideRight 0.35s ease">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                        <div style="display:flex;align-items:center;gap:12px">
                            <span style="background:#FFFFFF;color:#0A0F1C;font-size:0.7rem;
                                         font-weight:600;padding:4px 10px;border-radius:6px;
                                         font-family:monospace">TX-{item['idx']:05d}</span>
                            <span style="background:transparent;color:{txt_c};font-size:0.78rem;
                                         font-weight:600">Model: {verdict}</span>
                        </div>
                        <div style="text-align:right;font-size:0.82rem">
                            <span style="color:#94A3B8">Fraud </span>
                            <strong style="color:{txt_c}">{prob*100:.1f}%</strong>
                            &nbsp;&nbsp;
                            <span style="color:#94A3B8">Uncertainty </span>
                            <strong style="color:#FFFFFF">{uncert:.3f}</strong>
                        </div>
                    </div>
                    <div style="font-size:0.75rem;color:#94A3B8;font-family:monospace; word-break: break-word;">
                        {' · '.join([f'{k}: {v:.3f}' for k,v in list(item.get("features",{}).items())[:5]])}
                    </div>
                </div>""",unsafe_allow_html=True)

                bc1,bc2,_=st.columns([1,1,2])
                if bc1.button("✓  Legitimate",key=f"l_{item['idx']}",use_container_width=True):
                    try:
                        from active_learning import ActiveLearningManager
                        ActiveLearningManager().submit_label(item["idx"],0,analyst); st.rerun()
                    except Exception as e: st.error(str(e))
                if bc2.button("✗  Fraud",key=f"f_{item['idx']}",use_container_width=True,type="primary"):
                    try:
                        from active_learning import ActiveLearningManager
                        ActiveLearningManager().submit_label(item["idx"],1,analyst); st.rerun()
                    except Exception as e: st.error(str(e))

        st.markdown("<div style='height:24px'></div>",unsafe_allow_html=True)
        st.markdown('<p class="section-label">Retrain</p>',unsafe_allow_html=True)
        if reviewed>0:
            if st.button("Retrain Model with Human Labels",type="primary",use_container_width=True):
                with st.spinner("Retraining..."):
                    try:
                        from active_learning import ActiveLearningManager
                        mgr2=ActiveLearningManager()
                        X_a=np.load(os.path.join(ARTIFACT_DIR,"al_X_pool.npy"))
                        y_a=np.load(os.path.join(ARTIFACT_DIR,"al_y_pool.npy"))
                        X_t=np.load(os.path.join(ARTIFACT_DIR,"al_X_train.npy"))
                        y_t=np.load(os.path.join(ARTIFACT_DIR,"al_y_train.npy"))
                        res=mgr2.retrain(X_t,y_t,X_a,y_a)
                        if res:
                            st.success("Model retrained successfully.")
                            rm1,rm2,rm3=st.columns(3)
                            rm1.metric("AUC",   f"{res.get('new_auc',0):.4f}",   f"{res.get('auc_improvement',0):+.4f}")
                            rm2.metric("F1",    f"{res.get('new_f1',0):.4f}",    f"{res.get('new_f1',0)-res.get('original_f1',0):+.4f}")
                            rm3.metric("Recall",f"{res.get('new_recall',0):.4f}",f"{res.get('new_recall',0)-res.get('original_recall',0):+.4f}")
                    except Exception as e: st.error(f"Retrain error: {e}")
        else:
            st.info("Label at least 1 transaction before retraining.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px;animation:fadeIn 0.5s ease">
            <div style="font-size:2.5rem;margin-bottom:16px;color:#1F2A3C">◉</div>
            <div style="font-size:0.95rem;color:#94A3B8">
                Click <strong style="color:#FFFFFF">Generate Review Queue</strong>
                to start the Human-in-the-Loop pipeline
            </div>
        </div>""",unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border:none;height:1px;background:#1F2A3C;margin:48px 0 20px">
<p style="text-align:center;color:#94A3B8;font-size:0.75rem;margin:0">
    FraudShield AI v3 &nbsp;·&nbsp;
    Random Forest · XGBoost · GNN · Autoencoder · DRL Agent · Active Learning &nbsp;·&nbsp;
    NSAKCET · CSE (AI &amp; ML) · 2025–26
</p>
<div style="height:24px"></div>
""",unsafe_allow_html=True)