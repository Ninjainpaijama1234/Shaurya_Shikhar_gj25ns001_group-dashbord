# app.py ─ Cinema Performance Command-Center (defensive edition)
# -----------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from lifetimes import BetaGeoFitter, GammaGammaFitter
from prophet import Prophet

# ──────────── 1. PAGE CONFIG & CSS ────────────
st.set_page_config(
    page_title="🎬 Cinema Command-Center",
    layout="wide",
    page_icon="🍿",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html,body,[class*="css"]{font-family:Inter,sans-serif;}
    .metric-val{font-size:1.8rem;font-weight:700;color:#E50914;}
    .metric-lbl{font-size:.8rem;color:#555;}
    .metric-box{padding:14px;border-radius:14px;background:#fafafa;box-shadow:0 2px 6px rgba(0,0,0,.05);}
    div.js-plotly-plot .modebar{display:none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────── 2. SAFE HELPERS ────────────
ESSENTIAL_NUM = {
    "tickets_sold": 0,
    "ticket_price": 0.0,
    "capacity": 1,  # avoid division by zero
}
ESSENTIAL_CAT = {
    "city": "Unknown",
    "payment_method": "Unknown",
    "payment_status": "Success",
    "theatre_id": "T-0",
    "show_id": "S-0",
    "user_id": "U-0",
}

def guarantee_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures all downstream columns exist, with safe defaults."""
    for col, default in ESSENTIAL_NUM.items():
        if col not in df.columns:
            df[col] = default
    for col, default in ESSENTIAL_CAT.items():
        if col not in df.columns:
            df[col] = default
    # dates
    for date_col in ["booking_date", "show_time"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if "show_time" in df.columns and "show_date" not in df.columns:
        df["show_date"] = df["show_time"].dt.date
    return df

# ──────────── 3. DATA LOAD ────────────
@st.cache_data(show_spinner=True)
def load_data(path="BookMyShow_Combined_Clean_v2.csv") -> pd.DataFrame:
    if not Path(path).is_file():
        st.error(f"CSV not found at **{path}** — upload or move the file and reload.")
        st.stop()
    df = pd.read_csv(path)
    return guarantee_columns(df)

df = load_data()

# ──────────── 4. SIDEBAR FILTERS ────────────
with st.sidebar:
    st.header("Filters")
    date_min, date_max = (
        df["show_date"].min() if "show_date" in df else df.index.min(),
        df["show_date"].max() if "show_date" in df else df.index.max(),
    )
    date_range = st.date_input("Show Date", (date_min, date_max))
    cities = st.multiselect("City", sorted(df["city"].unique()), default=list(df["city"].unique()))

mask = (
    (df["show_date"].between(*date_range) if "show_date" in df else True)
    & (df["city"].isin(cities))
)
df = df.loc[mask].copy()

# ──────────── 5. RFM + SEGMENTS (robust) ────────────
SNAPSHOT = df["booking_date"].max() + timedelta(days=1) if "booking_date" in df else pd.Timestamp.today()

rfm = (
    df.groupby("user_id")
    .agg(
        recency=("booking_date", lambda x: (SNAPSHOT - x.max()).days if x.notna().any() else 9999),
        frequency=("tickets_sold", "sum"),
        monetary=("ticket_price", "sum"),
    )
    .reset_index()
)
sc_feats = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
kmeans = KMeans(n_clusters=4, n_init="auto", random_state=42).fit(sc_feats)
rfm["segment"] = kmeans.labels_
SIL = silhouette_score(sc_feats, kmeans.labels_)

# attach back
df = df.merge(rfm[["user_id", "segment"]], on="user_id", how="left")

# ──────────── 6. SAFE METRICS ────────────
def big_metric(val, label):
    st.markdown(f"""<div class="metric-box">
        <div class="metric-val">{val}</div>
        <div class="metric-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

m1 = f"{df['tickets_sold'].sum():,.0f}"
m2 = f"{(1 - df['tickets_sold'].sum()/df['capacity'].sum())*100:,.1f}%"
m3 = f"{SIL:.2f}"
col1,col2,col3 = st.columns(3)
with col1: big_metric(m1, "Tickets Sold")
with col2: big_metric(m2, "Vacancy %")
with col3: big_metric(m3, "Segm. Silhouette")

# ──────────── 7. PRICE–DEMAND MODEL (graceful) ────────────
X = df[["ticket_price"]]
y = df["tickets_sold"]
if len(X["ticket_price"].unique()) > 1:
    reg = GradientBoostingRegressor().fit(X, y)
    demand_curve = reg.predict(np.arange(100, 501, 10).reshape(-1,1))
else:
    reg = None
    demand_curve = np.zeros(41)

# ──────────── 8. DASHBOARD TABS ────────────
tabs = st.tabs(["Capacity", "Pricing"])

# ---- TAB 1: Capacity (simple but safe) ----
with tabs[0]:
    st.subheader("Seat Occupancy by Hour")
    if "show_time" in df.columns:
        heat = (
            df.assign(hour=df["show_time"].dt.hour)
            .groupby("hour")
            .agg(occ=("tickets_sold","sum"), cap=("capacity","sum"))
            .assign(occ_pct=lambda x: x.occ/x.cap)
            .reset_index()
        )
        fig = px.bar(heat, x="hour", y="occ_pct", labels={"occ_pct":"Occupancy %"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("`show_time` column not available; occupancy plot skipped.")

# ---- TAB 2: Pricing ----
with tabs[1]:
    st.subheader("Price Simulator")
    sel_price = st.slider("Ticket Price (₹)", 100, 500, 250, 25)
    if reg:
        proj = reg.predict([[sel_price]])[0]
        st.metric("Predicted Tickets", f"{proj:,.0f}")
        fig2 = go.Figure(go.Scatter(x=np.arange(100,501,10), y=demand_curve, mode="lines"))
        fig2.update_layout(xaxis_title="Price", yaxis_title="Predicted Tickets", height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Not enough price variation to build a demand model.")

# ──────────── 9. FOOTER ────────────
st.markdown(
    "<center style='font-size:.75rem;color:#888'>© 2025 • Robust version — no more KeyErrors 🚀</center>",
    unsafe_allow_html=True,
)
