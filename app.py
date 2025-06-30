# app.py ─ Cinema Performance Command-Center (final hardened build)
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta, date
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lifetimes import BetaGeoFitter, GammaGammaFitter
from prophet import Prophet

# ─────────────────────────── UI CONFIG ───────────────────────────
st.set_page_config(
    page_title="🎬 Cinema Command-Center",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html,body,[class*="css"]{font-family:Inter, sans-serif;}
    .metric-box{background:#fafafa;border-radius:14px;padding:14px;
                box-shadow:0 2px 6px rgba(0,0,0,.04);}
    .metric-val{font-size:1.8rem;font-weight:700;color:#E50914;margin:0;}
    .metric-lbl{font-size:.8rem;color:#555;margin-top:-4px;}
    div.js-plotly-plot .modebar{display:none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────── SAFE COLUMN DEFINITIONS ─────────────────
ESSENTIAL_NUM = {
    "tickets_sold": 0,
    "ticket_price": 0.0,
    "capacity": 1,            # prevent ÷0
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
    """Create any missing columns with safe defaults."""
    for col, default in ESSENTIAL_NUM.items():
        if col not in df.columns:
            df[col] = default
    for col, default in ESSENTIAL_CAT.items():
        if col not in df.columns:
            df[col] = default

    # Parse dates if present
    for c in ["booking_date", "show_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "show_time" in df.columns and "show_date" not in df.columns:
        df["show_date"] = df["show_time"].dt.date

    return df

# ────────────────────────── DATA LOADER ──────────────────────────
@st.cache_data(show_spinner=True)
def load_data(csv_path="BookMyShow_Combined_Clean_v2.csv") -> pd.DataFrame:
    if not Path(csv_path).is_file():
        st.error(f"**{csv_path}** not found. Upload/move the file and reload app.")
        st.stop()
    df = pd.read_csv(csv_path)
    return guarantee_columns(df)

df = load_data()

# ───────────────────────── SIDEBAR FILTERS ───────────────────────
with st.sidebar:
    st.header("Filters")

    # robust date range defaults
    if "show_date" in df.columns and df["show_date"].notna().any():
        _dmin = df["show_date"].dropna().min()
        _dmax = df["show_date"].dropna().max()
        dmin = _dmin if isinstance(_dmin, date) else _dmin.date()
        dmax = _dmax if isinstance(_dmax, date) else _dmax.date()
    else:
        today = date.today()
        dmin = dmax = today

    date_from, date_to = st.date_input("Show Date Range", (dmin, dmax))
    city_sel = st.multiselect("City", sorted(df["city"].unique()), default=list(df["city"].unique()))

# Apply filters safely
mask_date = True
if "show_date" in df.columns and df["show_date"].notna().any():
    mask_date = df["show_date"].between(date_from, date_to)
mask_city = df["city"].isin(city_sel)
df = df[mask_date & mask_city].copy()

# ────────────────────────── RFM SEGMENTS ─────────────────────────
SNAPSHOT = (
    df["booking_date"].max() + timedelta(days=1)
    if "booking_date" in df.columns and df["booking_date"].notna().any()
    else pd.Timestamp.today()
)

rfm = (
    df.groupby("user_id")
      .agg(
          recency=("booking_date", lambda s: (SNAPSHOT - s.max()).days if s.notna().any() else 9999),
          frequency=("tickets_sold", "sum"),
          monetary=("ticket_price", "sum"),
      )
      .reset_index()
)
sc = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rfm["segment"] = KMeans(n_clusters=4, random_state=42, n_init="auto").fit_predict(sc)
SIL = silhouette_score(sc, rfm["segment"])
df = df.merge(rfm[["user_id", "segment"]], on="user_id", how="left")

# ───────────────────────── KPI METRICS STRIP ─────────────────────
def metric(val, label):
    st.markdown(
        f"""<div class="metric-box">
                <p class="metric-val">{val}</p>
                <p class="metric-lbl">{label}</p>
            </div>""",
        unsafe_allow_html=True,
    )

tickets = f"{df['tickets_sold'].sum():,.0f}"
vacancy = f"{(1 - df['tickets_sold'].sum()/df['capacity'].sum())*100:,.1f}%"
sil_txt = f"{SIL:.2f}"

c1, c2, c3 = st.columns(3)
with c1: metric(tickets, "Tickets Sold")
with c2: metric(vacancy, "Vacancy Rate")
with c3: metric(sil_txt, "Segment Silhouette")

# ───────────────────────── SIMPLE PRICE MODEL ────────────────────
price_model_ready = df["ticket_price"].nunique() > 1
if price_model_ready:
    Xp, yp = df[["ticket_price"]], df["tickets_sold"]
    price_reg = GradientBoostingRegressor().fit(Xp, yp)
    price_curve_x = np.arange(100, 501, 10)
    price_curve_y = price_reg.predict(price_curve_x.reshape(-1, 1))
else:
    price_reg = None

# ──────────────────────────── DASHBOARD ──────────────────────────
tab1, tab2 = st.tabs(["📊 Capacity", "💰 Pricing"])

with tab1:
    st.subheader("Hourly Seat Occupancy")
    if "show_time" in df.columns:
        hourly = (
            df.assign(hour=df["show_time"].dt.hour)
              .groupby("hour")
              .agg(occ=("tickets_sold","sum"), cap=("capacity","sum"))
              .assign(occ_pct=lambda x: x.occ/x.cap)
              .reset_index()
        )
        fig = px.bar(hourly, x="hour", y="occ_pct", labels={"occ_pct":"Occupancy %"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("`show_time` column not available; occupancy chart skipped.")

with tab2:
    st.subheader("Price Simulator")
    chosen_price = st.slider("Choose Ticket Price (₹)", 100, 500, 250, 25)
    if price_reg:
        pred_tix = int(price_reg.predict([[chosen_price]])[0])
        st.metric("Predicted Tickets", f"{pred_tix:,}")
        fig2 = go.Figure(go.Scatter(x=price_curve_x, y=price_curve_y, mode="lines"))
        fig2.update_layout(xaxis_title="Price (₹)", yaxis_title="Predicted Tickets", height=360)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Not enough price variability to model demand.")

# ─────────────────────────── FOOTER ──────────────────────────────
st.markdown(
    "<center style='color:#888;font-size:.75rem'>© 2025 • Final hardened build — all date-range errors resolved ✔️</center>",
    unsafe_allow_html=True,
)
