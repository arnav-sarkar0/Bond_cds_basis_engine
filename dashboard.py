"""
Bond–CDS Basis Engine  |  Streamlit Dashboard
===============================================
Run with:
    streamlit run dashboard.py

Requires:
    pip install streamlit plotly pandas scipy numpy
"""

import io
import csv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import brentq, minimize

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Bond–CDS Basis Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark terminal theme */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1e2d40;
}
section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label {
    color: #64748b !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Mono', monospace !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #161b27 !important;
    border: 1px solid #1e2d40 !important;
    color: #e2e8f0 !important;
}

/* Header */
.dash-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #334155;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.dash-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.dash-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    color: #475569;
    margin-top: 4px;
}

/* Metric cards */
.metric-card {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 6px;
    padding: 16px 20px;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-label {
    font-size: 10px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 22px;
    font-weight: 600;
    color: #e2e8f0;
}
.metric-sub {
    font-size: 11px;
    color: #334155;
    margin-top: 2px;
}

/* Signal badges */
.badge-buy   { background:#0f3d2e; color:#34d399; border:1px solid #065f46;
               padding:3px 10px; border-radius:4px; font-size:11px;
               font-family:'IBM Plex Mono',monospace; }
.badge-sell  { background:#3d1515; color:#f87171; border:1px solid #7f1d1d;
               padding:3px 10px; border-radius:4px; font-size:11px;
               font-family:'IBM Plex Mono',monospace; }
.badge-fair  { background:#1e2a3a; color:#64748b; border:1px solid #1e3a5f;
               padding:3px 10px; border-radius:4px; font-size:11px;
               font-family:'IBM Plex Mono',monospace; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0d1117;
    border-bottom: 1px solid #1e2d40;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #475569 !important;
    padding: 12px 24px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #1e2d40;
    margin: 20px 0;
}

/* Table */
.basis-table {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    width: 100%;
    border-collapse: collapse;
}
.basis-table th {
    color: #475569;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 8px 16px;
    border-bottom: 1px solid #1e2d40;
    text-align: left;
    font-weight: 500;
}
.basis-table td {
    padding: 10px 16px;
    border-bottom: 1px solid #0f172a;
    color: #cbd5e1;
}
.basis-table tr:hover td { background: #0f1823; }
.pos-basis { color: #f87171; }
.neg-basis { color: #34d399; }

/* Run button */
div[data-testid="stButton"] > button {
    background: #0ea5e9 !important;
    color: #0a0e1a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 28px !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    background: #38bdf8 !important;
}

/* Upload widget */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px dashed #1e2d40 !important;
    border-radius: 6px !important;
}

/* Info box */
.info-box {
    background: #0c1929;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #0ea5e9;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #64748b;
    line-height: 1.6;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #38bdf8 !important;
    border: 1px solid #1e3a5f !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# GENERATOR PARAMETERS  (from generate_market_data.py)
# ══════════════════════════════════════════════════════════════

RATING_HAZARD = {
    "AAA": 0.0002, "AA": 0.0005, "A": 0.0010,
    "BBB": 0.0035, "BB": 0.0120, "B": 0.0300, "CCC": 0.0800,
}
RATING_TERM_SHAPE = {
    "AAA": [1.00,1.05,1.10,1.12,1.13],
    "AA":  [1.00,1.08,1.15,1.20,1.22],
    "A":   [1.00,1.10,1.20,1.28,1.33],
    "BBB": [1.00,1.15,1.30,1.42,1.50],
    "BB":  [1.00,1.20,1.40,1.55,1.65],
    "B":   [1.00,1.25,1.50,1.70,1.85],
    "CCC": [1.00,1.30,1.60,1.85,2.05],
}
SECTOR_MULTIPLIER  = {"financials":1.20,"energy":1.35,"tech":0.85,"industrials":1.00,"utilities":0.75}
SCENARIO_MULTIPLIER= {"normal":1.00,"stress":2.50,"crisis":5.00}
SCENARIO_CDS_BASIS = {
    "normal": {"mean": 0.0, "std": 3.0},
    "stress": {"mean":15.0, "std":10.0},
    "crisis": {"mean":40.0, "std":25.0},
}
MATURITIES   = [1, 2, 3, 4, 5]
RISK_FREE    = 0.05
RECOVERY     = 0.40
FACE         = 100.0
CDS_STEPS    = 500   # lower for dashboard speed
SMOOTH_PEN   = 10.0


# ══════════════════════════════════════════════════════════════
# CORE MATH
# ══════════════════════════════════════════════════════════════

def hazard_integral(hazards, t):
    result, prev = 0.0, 0.0
    for i, t_i in enumerate(range(1, len(hazards)+1)):
        dt = min(t, float(t_i)) - prev
        if dt > 0:
            result += hazards[i] * dt
        prev = float(t_i)
        if t_i >= t:
            break
    return result

def py_bond_price(hazards, coupon=5.0, face=100.0, r=RISK_FREE, recovery=RECOVERY):
    L, T, price = 1.0-recovery, len(hazards), 0.0
    for i in range(T):
        t   = float(i+1)
        H_t = hazard_integral(hazards, t)
        d   = np.exp(-(r*t + L*H_t))
        cf  = coupon + (face if i==T-1 else 0.0)
        price += cf * d
    return price

def py_cds_spread(hazards, maturity, r=RISK_FREE, recovery=RECOVERY, steps=CDS_STEPS):
    L, dt = 1.0-recovery, maturity/steps
    prem, prot = 0.0, 0.0
    for i in range(1, steps+1):
        t        = i*dt
        H_t      = hazard_integral(hazards, t)
        discount = np.exp(-r*t)
        survival = np.exp(-H_t)
        idx      = min(int(np.ceil(t))-1, len(hazards)-1)
        h_t      = hazards[idx]
        prem += discount * survival * dt
        prot += discount * h_t * survival * L * dt
    return 0.0 if prem==0 else prot/prem

def implied_coupon(hazards, maturity):
    L = 1.0 - RECOVERY
    discounts = []
    for i in range(maturity):
        t   = float(i+1)
        H_t = hazard_integral(hazards, t)
        discounts.append(np.exp(-(RISK_FREE*t + L*H_t)))
    return FACE * (1.0 - discounts[-1]) / sum(discounts)

def build_hazard_curve(rating, sector, scenario):
    base  = RATING_HAZARD[rating]
    sm    = SECTOR_MULTIPLIER[sector]
    scm   = SCENARIO_MULTIPLIER[scenario]
    shape = RATING_TERM_SHAPE[rating]
    h     = np.array([base*sm*scm*shape[i] for i in range(len(MATURITIES))])
    return np.clip(h, 1e-6, 0.95)

def generate_data(rating, sector, scenario, seed=42):
    rng     = np.random.default_rng(seed)
    hazards = build_hazard_curve(rating, sector, scenario)
    basis_cfg = SCENARIO_CDS_BASIS[scenario]
    bonds_data, cds_data = [], []
    for m in MATURITIES:
        coupon       = implied_coupon(hazards[:m], m)
        clean_price  = py_bond_price(hazards[:m], coupon=coupon)
        noise_std    = 0.05 if rating in ("AAA","AA","A") else 0.15 if rating in ("BBB","BB") else 0.30
        market_price = max(clean_price + rng.normal(0, noise_std), 1.0)
        bonds_data.append({"maturity":m,"coupon":round(coupon,4),"face":FACE,"market_price":round(market_price,4)})
        model_s   = py_cds_spread(hazards[:m], m)
        basis_n   = rng.normal(basis_cfg["mean"], basis_cfg["std"]) / 10_000
        mkt_spread= max(model_s + basis_n, 1e-5)
        cds_data.append({"maturity":m,"spread_bps":round(mkt_spread*10000,2)})
    return bonds_data, cds_data

def bootstrap_hazard(bonds):
    hazards = []
    for bond in bonds:
        mp, coupon, face = bond["price"], bond.get("coupon",5.0), bond.get("face",100.0)
        def residual(h):
            return py_bond_price(np.array(hazards+[h]), coupon=coupon, face=face) - mp
        try:
            h_i = brentq(residual, 1e-6, 0.9999, xtol=1e-10, maxiter=300)
        except ValueError:
            h_i = 0.05
        hazards.append(h_i)
    return np.array(hazards)

def smooth_hazards(hazard_init, bonds):
    def objective(h):
        h   = np.asarray(h)
        err = sum((py_bond_price(h[:b["maturity"]], coupon=b.get("coupon",5.0), face=b.get("face",100.0)) - b["price"])**2 for b in bonds)
        smo = sum((h[i+1]-h[i])**2 for i in range(len(h)-1))
        return err + SMOOTH_PEN * smo
    res = minimize(objective, hazard_init, method="L-BFGS-B",
                   bounds=[(1e-6,0.9999)]*len(hazard_init),
                   options={"maxiter":500,"ftol":1e-12})
    return res.x

def survival_and_default(hazards):
    H  = np.array([hazard_integral(hazards, float(t)) for t in range(1, len(hazards)+1)])
    S  = np.exp(-H)
    return S, 1.0-S

def compute_basis(hazards, bonds, market_cds):
    results = []
    for bond in bonds:
        m       = bond["maturity"]
        mkt_cds = market_cds.get(m)
        if mkt_cds is None:
            continue
        model_cds = py_cds_spread(hazards[:m], float(m))
        basis_bps = (mkt_cds - model_cds) * 10_000
        if basis_bps > 5:
            signal, signal_type = "Buy Bond / Sell CDS", "buy"
        elif basis_bps < -5:
            signal, signal_type = "Sell Bond / Buy CDS", "sell"
        else:
            signal, signal_type = "Fairly Priced", "fair"
        results.append({
            "maturity": m, "model_cds": model_cds,
            "market_cds": mkt_cds, "basis_bps": basis_bps,
            "signal": signal, "signal_type": signal_type,
        })
    return results

def check_bond_arb(hazards, bonds):
    rows = []
    for bond in bonds:
        m       = bond["maturity"]
        P_mkt   = bond["price"]
        P_model = py_bond_price(hazards[:m], coupon=bond.get("coupon",5.0), face=bond.get("face",100.0))
        diff    = P_mkt - P_model
        if diff > 0.10:
            sig, sig_type = "Overpriced → Sell/Short", "sell"
        elif diff < -0.10:
            sig, sig_type = "Underpriced → Buy", "buy"
        else:
            sig, sig_type = "Fairly Priced", "fair"
        rows.append({"maturity":m,"market_price":P_mkt,"model_price":round(P_model,4),
                     "diff":round(diff,4),"signal":sig,"signal_type":sig_type})
    return rows


# ══════════════════════════════════════════════════════════════
# PLOTLY CHART HELPERS
# ══════════════════════════════════════════════════════════════

CHART_BG   = "#0a0e1a"
GRID_COLOR = "#1e2d40"
FONT_COLOR = "#94a3b8"
FONT_MONO  = "IBM Plex Mono"

def base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(family=FONT_MONO, size=12, color="#64748b"), x=0.01),
        paper_bgcolor=CHART_BG, plot_bgcolor="#0d1117",
        font=dict(family=FONT_MONO, color=FONT_COLOR, size=11),
        margin=dict(l=48, r=16, t=40, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified",
    )

def chart_hazard(mats, hazard_raw, hazard_final):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mats, y=hazard_raw,   mode="lines+markers",
        name="Bootstrapped", line=dict(color="#334155", dash="dash", width=1.5),
        marker=dict(color="#475569", size=6)))
    fig.add_trace(go.Scatter(x=mats, y=hazard_final, mode="lines+markers",
        name="Smoothed", line=dict(color="#0ea5e9", width=2),
        marker=dict(color="#0ea5e9", size=7),
        fill="tozeroy", fillcolor="rgba(14,165,233,0.06)"))
    fig.update_layout(**base_layout("HAZARD RATE CURVE  h(t)"),
                      xaxis_title="Maturity (years)", yaxis_title="Hazard Rate",
                      yaxis_tickformat=".4f")
    return fig

def chart_survival(mats, S, PD):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mats, y=S,  mode="lines+markers", name="Survival  S(t)",
        line=dict(color="#34d399", width=2), marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.05)"))
    fig.add_trace(go.Scatter(x=mats, y=PD, mode="lines+markers", name="Cumulative PD",
        line=dict(color="#f87171", width=2), marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.05)"))
    fig.update_layout(**base_layout("SURVIVAL & DEFAULT PROBABILITY"),
                      xaxis_title="Maturity (years)", yaxis_title="Probability",
                      yaxis_range=[0,1])
    return fig

def chart_cds(basis_results):
    b_mats  = [r["maturity"]          for r in basis_results]
    model_s = [r["model_cds"]*10000   for r in basis_results]
    mkt_s   = [r["market_cds"]*10000  for r in basis_results]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b_mats, y=model_s, mode="lines+markers", name="Model CDS",
        line=dict(color="#0ea5e9", width=2), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=b_mats, y=mkt_s,   mode="lines+markers", name="Market CDS",
        line=dict(color="#f59e0b", width=2, dash="dash"), marker=dict(size=7)))
    fig.update_layout(**base_layout("CDS SPREAD: MODEL vs MARKET"),
                      xaxis_title="Maturity (years)", yaxis_title="Spread (bps)")
    return fig

def chart_basis(basis_results):
    b_mats     = [r["maturity"]  for r in basis_results]
    basis_vals = [r["basis_bps"] for r in basis_results]
    colors     = ["#f87171" if b > 0 else "#34d399" for b in basis_vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=b_mats, y=basis_vals, marker_color=colors,
                         marker_line_color=CHART_BG, marker_line_width=2,
                         name="Basis"))
    fig.add_hline(y=0, line_color="#334155", line_width=1)
    fig.add_hline(y=5,  line_color="#1e3a2a", line_width=1, line_dash="dot")
    fig.add_hline(y=-5, line_color="#3d1515", line_width=1, line_dash="dot")
    fig.update_layout(**base_layout("BASIS  (Market − Model CDS)  bps"),
                      xaxis_title="Maturity (years)", yaxis_title="Basis (bps)")
    return fig

def chart_bond_fit(mats, hazard_final, bonds):
    mkt_prices   = [b["price"] for b in bonds]
    model_prices = [py_bond_price(hazard_final[:b["maturity"]],
                    coupon=b.get("coupon",5.0), face=b.get("face",100.0)) for b in bonds]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mats, y=mkt_prices,   mode="lines+markers", name="Market Price",
        line=dict(color="#f59e0b", width=2, dash="dash"), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=mats, y=model_prices, mode="lines+markers", name="Model Price",
        line=dict(color="#0ea5e9", width=2), marker=dict(size=7),
        fill="tonexty", fillcolor="rgba(14,165,233,0.04)"))
    fig.update_layout(**base_layout("BOND PRICE: MARKET vs MODEL"),
                      xaxis_title="Maturity (years)", yaxis_title="Price")
    return fig


# ══════════════════════════════════════════════════════════════
# CSV HELPERS
# ══════════════════════════════════════════════════════════════

def bonds_to_csv(bonds_data):
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=["maturity","coupon","face","market_price"])
    w.writeheader(); w.writerows(bonds_data)
    return buf.getvalue()

def cds_to_csv(cds_data):
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=["maturity","spread_bps"])
    w.writeheader(); w.writerows(cds_data)
    return buf.getvalue()

def parse_uploaded_bonds(file):
    df    = pd.read_csv(file)
    bonds = []
    for _, row in df.iterrows():
        bonds.append({
            "maturity": int(row["maturity"]),
            "coupon":   float(row.get("coupon", 5.0)),
            "face":     float(row.get("face",   100.0)),
            "price":    float(row["market_price"]),
        })
    return sorted(bonds, key=lambda b: b["maturity"])

def parse_uploaded_cds(file):
    df  = pd.read_csv(file)
    cds = {}
    for _, row in df.iterrows():
        cds[int(row["maturity"])] = float(row["spread_bps"]) / 10_000.0
    return cds


# ══════════════════════════════════════════════════════════════
# SIGNAL BADGE HTML
# ══════════════════════════════════════════════════════════════

def badge(signal_type, text):
    cls = {"buy":"badge-buy","sell":"badge-sell","fair":"badge-fair"}.get(signal_type,"badge-fair")
    return f'<span class="{cls}">{text}</span>'


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="dash-header">Duffie–Singleton Credit Model</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-title">Bond–CDS Basis Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-subtitle">Hazard rate calibration · CDS pricing · Basis detection · Arbitrage signals</div>', unsafe_allow_html=True)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### DATA SOURCE")
    data_mode = st.radio("", ["Generate synthetic data", "Upload CSV files"], label_visibility="collapsed")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if data_mode == "Generate synthetic data":
        st.markdown("### ISSUER PROFILE")
        rating   = st.selectbox("Credit Rating",   list(RATING_HAZARD.keys()),     index=3)
        sector   = st.selectbox("Sector",          list(SECTOR_MULTIPLIER.keys()),  index=3)
        scenario = st.selectbox("Market Scenario", list(SCENARIO_MULTIPLIER.keys()),index=0)
        seed     = st.slider("Random Seed", 1, 100, 42)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### MODEL PARAMS")
        use_smooth = st.toggle("Smooth hazard curve", value=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        run = st.button("▶  RUN ENGINE")

    else:
        st.markdown("### UPLOAD FILES")
        bonds_file = st.file_uploader("bonds.csv", type="csv")
        cds_file   = st.file_uploader("cds_market.csv", type="csv")
        st.markdown("""<div class="info-box">
bonds.csv columns:<br>maturity, coupon, face, market_price<br><br>
cds_market.csv columns:<br>maturity, spread_bps
</div>""", unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### MODEL PARAMS")
        use_smooth = st.toggle("Smooth hazard curve", value=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        run = st.button("▶  RUN ENGINE")


# ══════════════════════════════════════════════════════════════
# ENGINE RUN
# ══════════════════════════════════════════════════════════════

if run:
    with st.spinner("Calibrating hazard curve…"):

        # ── Load / generate data ─────────────────────────────
        if data_mode == "Generate synthetic data":
            bonds_raw, cds_raw = generate_data(rating, sector, scenario, seed)
            bonds      = [{"maturity":b["maturity"],"coupon":b["coupon"],
                           "face":b["face"],"price":b["market_price"]} for b in bonds_raw]
            market_cds = {c["maturity"]: c["spread_bps"]/10000.0 for c in cds_raw}
            profile    = f"{rating} · {sector.title()} · {scenario.upper()}"
        else:
            if bonds_file is None or cds_file is None:
                st.error("Please upload both bonds.csv and cds_market.csv")
                st.stop()
            bonds      = parse_uploaded_bonds(bonds_file)
            market_cds = parse_uploaded_cds(cds_file)
            bonds_raw  = [{"maturity":b["maturity"],"coupon":b["coupon"],
                           "face":b["face"],"market_price":b["price"]} for b in bonds]
            cds_raw    = [{"maturity":m,"spread_bps":round(s*10000,2)} for m,s in market_cds.items()]
            profile    = "Custom Upload"

        mats = np.array([b["maturity"] for b in bonds], dtype=float)

        # ── Bootstrap ────────────────────────────────────────
        hazard_raw   = bootstrap_hazard(bonds)
        hazard_final = smooth_hazards(hazard_raw, bonds) if use_smooth else hazard_raw.copy()

        # ── Survival / default ───────────────────────────────
        S, PD = survival_and_default(hazard_final)

        # ── Basis ────────────────────────────────────────────
        basis_results = compute_basis(hazard_final, bonds, market_cds)
        arb_results   = check_bond_arb(hazard_final, bonds)

    # ── Store in session state ────────────────────────────
    st.session_state["results"] = {
        "mats": mats, "hazard_raw": hazard_raw, "hazard_final": hazard_final,
        "S": S, "PD": PD, "basis_results": basis_results, "arb_results": arb_results,
        "bonds": bonds, "bonds_raw": bonds_raw, "cds_raw": cds_raw,
        "market_cds": market_cds, "profile": profile,
    }


# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.markdown("""<div class="info-box" style="margin-top:40px; font-size:12px; line-height:2;">
Configure an issuer profile in the sidebar and click <strong style="color:#0ea5e9">▶ RUN ENGINE</strong> to calibrate the hazard curve and compute the Bond–CDS basis.<br><br>
<strong style="color:#64748b">Workflow:</strong><br>
1 · Bootstrap piecewise-constant h(t) from bond prices<br>
2 · Optionally smooth the hazard curve<br>
3 · Price CDS using the same h(t) via Duffie–Singleton<br>
4 · Compare with market CDS → detect basis & generate signals
</div>""", unsafe_allow_html=True)
    st.stop()

R = st.session_state["results"]

# ── Profile banner ────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Issuer Profile</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#0ea5e9;font-weight:600;">{R['profile']}</div>
</div>
""", unsafe_allow_html=True)

# ── Top metric cards ──────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, m in zip([c1,c2,c3,c4,c5], R["mats"].astype(int)):
    r = next((x for x in R["basis_results"] if x["maturity"]==m), None)
    if r:
        color = "#f87171" if r["basis_bps"]>5 else "#34d399" if r["basis_bps"]<-5 else "#64748b"
        col.markdown(f"""<div class="metric-card">
<div class="metric-label">{m}Y Basis</div>
<div class="metric-value" style="color:{color}">{r['basis_bps']:+.1f}<span style="font-size:13px">bp</span></div>
<div class="metric-sub">Model {r['model_cds']*10000:.0f}bp · Mkt {r['market_cds']*10000:.0f}bp</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab_inputs, tab_results, tab_charts = st.tabs(["  INPUTS  ", "  RESULTS  ", "  CHARTS  "])


# ── TAB 1: INPUTS ─────────────────────────────────────────────
with tab_inputs:
    col_b, col_c = st.columns(2)

    with col_b:
        st.markdown("##### Bond Market Data")
        bond_df = pd.DataFrame([{
            "Maturity": f"{b['maturity']}Y",
            "Coupon":   f"{b['coupon']:.4f}",
            "Face":     f"{b['face']:.0f}",
            "Mkt Price":f"{b['price']:.4f}",
        } for b in R["bonds"]])
        st.dataframe(bond_df, use_container_width=True, hide_index=True)

        # Download bonds CSV
        st.download_button(
            "⬇  Download bonds.csv",
            data=bonds_to_csv(R["bonds_raw"]),
            file_name="bonds.csv", mime="text/csv",
        )

    with col_c:
        st.markdown("##### CDS Market Spreads")
        cds_df = pd.DataFrame([{
            "Maturity":    f"{m}Y",
            "Market CDS":  f"{s*10000:.2f} bps",
        } for m, s in R["market_cds"].items()])
        st.dataframe(cds_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇  Download cds_market.csv",
            data=cds_to_csv(R["cds_raw"]),
            file_name="cds_market.csv", mime="text/csv",
        )

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    st.markdown("##### Calibrated Hazard Rates")
    haz_df = pd.DataFrame({
        "Maturity":         [f"{int(m)}Y" for m in R["mats"]],
        "h(t) Bootstrap":   [f"{h:.6f}" for h in R["hazard_raw"]],
        "h(t) Smoothed":    [f"{h:.6f}" for h in R["hazard_final"]],
        "H(t) Cumulative":  [f"{hazard_integral(R['hazard_final'], float(m)):.6f}" for m in R["mats"]],
        "S(t)":             [f"{s:.4f}" for s in R["S"]],
        "PD(t)":            [f"{p:.4f}" for p in R["PD"]],
    })
    st.dataframe(haz_df, use_container_width=True, hide_index=True)


# ── TAB 2: RESULTS ────────────────────────────────────────────
with tab_results:

    st.markdown("##### Bond–CDS Basis & Trade Signals")
    rows_html = ""
    for r in R["basis_results"]:
        color = "#f87171" if r["basis_bps"]>0 else "#34d399"
        b_cls = "pos-basis" if r["basis_bps"]>0 else "neg-basis"
        sig   = badge(r["signal_type"], r["signal"])
        rows_html += f"""<tr>
<td>{r['maturity']}Y</td>
<td>{r['model_cds']*10000:.1f} bp</td>
<td>{r['market_cds']*10000:.1f} bp</td>
<td class="{b_cls}">{r['basis_bps']:+.1f} bp</td>
<td>{sig}</td>
</tr>"""

    st.markdown(f"""<table class="basis-table">
<thead><tr>
<th>Mat</th><th>Model CDS</th><th>Market CDS</th><th>Basis</th><th>Signal</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)
    st.markdown("##### Bond Arbitrage Check")
    rows_html2 = ""
    for r in R["arb_results"]:
        sig = badge(r["signal_type"], r["signal"])
        d_cls = "pos-basis" if r["diff"]>0 else "neg-basis" if r["diff"]<0 else ""
        rows_html2 += f"""<tr>
<td>{r['maturity']}Y</td>
<td>{r['market_price']:.4f}</td>
<td>{r['model_price']:.4f}</td>
<td class="{d_cls}">{r['diff']:+.4f}</td>
<td>{sig}</td>
</tr>"""

    st.markdown(f"""<table class="basis-table">
<thead><tr>
<th>Mat</th><th>Market Price</th><th>Model Price</th><th>Diff</th><th>Signal</th>
</tr></thead>
<tbody>{rows_html2}</tbody>
</table>""", unsafe_allow_html=True)


# ── TAB 3: CHARTS ─────────────────────────────────────────────
with tab_charts:
    col_l, col_r = st.columns(2)

    with col_l:
        st.plotly_chart(chart_hazard(R["mats"], R["hazard_raw"], R["hazard_final"]),
                        use_container_width=True)
        st.plotly_chart(chart_cds(R["basis_results"]),
                        use_container_width=True)
        st.plotly_chart(chart_bond_fit(R["mats"], R["hazard_final"], R["bonds"]),
                        use_container_width=True)

    with col_r:
        st.plotly_chart(chart_survival(R["mats"], R["S"], R["PD"]),
                        use_container_width=True)
        st.plotly_chart(chart_basis(R["basis_results"]),
                        use_container_width=True)
