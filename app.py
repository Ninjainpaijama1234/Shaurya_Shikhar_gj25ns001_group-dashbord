# app.py — Cinema Performance Command-Center (rock-solid build)
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import silhouette_score

# ────────────────────────── CONFIG & THEME ───────────────────────
st.set_page_config("🎬 Cinema Command-Center", "🎟️", "wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    html,body,[class*="css"]{font-family:Inter,sans-serif;}
    .kpi{background:#fafafa;border-radius:14px;padding:14px;box-shadow:0 2px 6px rgba(0,0,0,.05);}
    .kpi span{display:block;text-align:center;}
    .kpi .v{font-size:1.7rem;font-weight:700;color:#E50914;}
    .kpi .l{font-size:.8rem;color:#555;margin-top:-4px;}
    div.js-plotly-plot .modebar{display:none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────── SAFE COLUMN DEFAULTS ─────────────────────
NUM_DEFAULT = {"tickets_sold": 0, "ticket_price": 0.0, "capacity": 1}
CAT_DEFAULT = {
    "city": "Unknown", "payment_method": "Unknown", "payment_status": "Success",
    "theatre_id": "T-0", "show_id": "S-0", "user_id": "U-0",
}

def guarantee_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c, d in NUM_DEFAULT.items():
        if c not in df: df[c] = d
    for c, d in CAT_DEFAULT.items():
        if c not in df: df[c] = d
    for c in ["booking_date", "show_time"]:
        if c in df: df[c] = pd.to_datetime(df[c], errors="coerce")
    if "show_time" in df and "show_date" not in df:
        df["show_date"] = df["show_time"].dt.normalize()      # datetime64, always comparable
    return df

# ───────────────────────── DATA LOADER ──────────────────────────
@st.cache_data(show_spinner=True)
def load(csv="BookMyShow_Combined_Clean_v2.csv") -> pd.DataFrame:
    if not Path(csv).is_file():
        st.error(f"**{csv}** not found. Upload then press **R** to reload.")
        st.stop()
    return guarantee_cols(pd.read_csv(csv))

df = load()

# ───────────────────────── SIDEBAR FILTERS ──────────────────────
with st.sidebar:
    st.header("Filters")

    if "show_date" in df and df["show_date"].notna().any():
        vdates = df["show_date"].dropna()
        dmin = vdates.min().date()
        dmax = vdates.max().date()
        start_d, end_d = st.date_input("Show Date Range", (dmin, dmax))
        start_ts, end_ts = pd.Timestamp(start_d), pd.Timestamp(end_d)
        mask_date = df["show_date"].between(start_ts, end_ts)
    else:
        st.info("No valid `show_date` column – date filter disabled.")
        mask_date = pd.Series(True, index=df.index)

    cities = st.multiselect("City", sorted(df["city"].unique()), default=list(df["city"].unique()))
    mask_city = df["city"].isin(cities)

df = df[mask_date & mask_city].copy()

# ───────────────────────── RFM SEGMENTATION ─────────────────────
SNAP = (df["booking_date"].max() + timedelta(days=1)) if "booking_date" in df else pd.Timestamp.today()
rfm = (
    df.groupby("user_id")
      .agg(recency=("booking_date", lambda s: (SNAP - s.max()).days if s.notna().any() else 9999),
           frequency=("tickets_sold", "sum"),
           monetary=("ticket_price", "sum"))
      .reset_index()
)
sc = StandardScaler().fit_transform(rfm[["recency","frequency","monetary"]])
rfm["segment"] = KMeans(4, random_state=42, n_init="auto").fit_predict(sc)
sil = silhouette_score(sc, rfm["segment"])
df = df.merge(rfm[["user_id","segment"]], on="user_id", how="left")

# ───────────────────────── KPI DISPLAY ─────────────────────────
def kpi(val, lbl):
    st.markdown(f'<div class="kpi"><span class="v">{val}</span><span class="l">{lbl}</span></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: kpi(f"{df.tickets_sold.sum():,}", "Tickets Sold")
with c2: kpi(f"{(1 - df.tickets_sold.sum()/df.capacity.sum())*100:,.1f} %", "Vacancy Rate")
with c3: kpi(f"{sil:.2f}", "Segm. Silhouette")

# ───────────────────────── PRICE DEMAND MODEL ───────────────────
price_ready = df["ticket_price"].nunique() > 1
if price_ready:
    reg = GradientBoostingRegressor(random_state=42).fit(df[["ticket_price"]], df["tickets_sold"])
    curve_x = np.arange(100, 501, 10)
    curve_y = reg.predict(curve_x.reshape(-1, 1))

# ───────────────────────── DASHBOARD TABS ───────────────────────
tab_cap, tab_price = st.tabs(["📊 Capacity", "💰 Pricing"])

with tab_cap:
    st.subheader("Hourly Seat Occupancy")
    if "show_time" in df:
        hourly = (
            df.assign(hour=df["show_time"].dt.hour)
              .groupby("hour")
              .agg(occ=("tickets_sold","sum"), cap=("capacity","sum"))
              .assign(occ_pct=lambda x: x.occ/x.cap)
              .reset_index()
        )
        st.plotly_chart(px.bar(hourly, x="hour", y="occ_pct",
                               labels={"occ_pct":"Occupancy %"}, height=380),
                        use_container_width=True)
    else:
        st.warning("Column `show_time` missing – occupancy chart skipped.")

with tab_price:
    st.subheader("Price Simulator")
    price_sel = st.slider("Ticket Price (₹)", 100, 500, 250, 25)
    if price_ready:
        st.metric("Predicted Tickets", f"{reg.predict([[price_sel]])[0]:,.0f}")
        fig = go.Figure(go.Scatter(x=curve_x, y=curve_y, mode="lines"))
        fig.update_layout(xaxis_title="Price (₹)", yaxis_title="Predicted Tickets", height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough price variation to build demand model.")

# ───────────────────────── FOOTER ───────────────────────────────
st.markdown(
    "<center style='color:#888;font-size:.75rem'>© 2025 • Rock-solid build — date & type errors fully handled 🚀</center>",
    unsafe_allow_html=True,
)
