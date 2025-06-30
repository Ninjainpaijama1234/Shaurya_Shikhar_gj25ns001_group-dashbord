# app.py — Cinema Performance Command-Center (type-safe build)
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import silhouette_score

# ───── 0.  PAGE CONFIG & THEME ───────────────────────────────────────────────
st.set_page_config("🎬 Cinema Command-Center", "🎟️", "wide")
st.markdown(
    """
    <style>
    html,body,[class*="css"]{font-family:Inter,sans-serif;}
    .kpi{background:#fafafa;border-radius:14px;padding:14px;box-shadow:0 2px 6px rgba(0,0,0,.05);}
    .kpi .v{font-size:1.7rem;font-weight:700;color:#E50914;text-align:center;}
    .kpi .l{font-size:.8rem;color:#555;text-align:center;margin-top:-4px;}
    div.js-plotly-plot .modebar{display:none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───── 1.  COLUMN DEFAULTS & COERCION HELPERS ───────────────────────────────
NUM_DEFAULT = {"tickets_sold": 0, "ticket_price": 0.0, "capacity": 1}
CAT_DEFAULT = {
    "city": "Unknown", "payment_method": "Unknown", "payment_status": "Success",
    "theatre_id": "T-0", "show_id": "S-0", "user_id": "U-0",
}

def guarantee_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure mandatory cols + canonical dtypes (esp. show_date as datetime64[ns])."""
    for col, default in NUM_DEFAULT.items():
        if col not in df: df[col] = default
    for col, default in CAT_DEFAULT.items():
        if col not in df: df[col] = default

    # Robust datetime parsing
    for dcol in ["booking_date", "show_time", "show_date"]:
        if dcol in df:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Derive show_date if absent
    if "show_date" not in df and "show_time" in df:
        df["show_date"] = df["show_time"]

    # Final: unify + normalise
    if "show_date" in df:
        df["show_date"] = pd.to_datetime(df["show_date"], errors="coerce").dt.normalize()

    return df

# ───── 2.  DATA LOADER ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_csv(path="BookMyShow_Combined_Clean_v2.csv") -> pd.DataFrame:
    if not Path(path).is_file():
        st.error(f"❌ **{path}** not found. Upload it, then click *Rerun*.")
        st.stop()
    return guarantee_cols(pd.read_csv(path))

df = load_csv()

# ───── 3.  SIDEBAR FILTERS (all datetime64-safe) ─────────────────────────────
with st.sidebar:
    st.header("Filters")

    if "show_date" in df and df["show_date"].notna().any():
        dmin = df["show_date"].min().date()
        dmax = df["show_date"].max().date()
        start_d, end_d = st.date_input("Show Date Range", (dmin, dmax))
        start_ts, end_ts = pd.Timestamp(start_d), pd.Timestamp(end_d)
        mask_date = df["show_date"].between(start_ts, end_ts)
    else:
        st.info("No valid *show_date* column — date filter disabled.")
        mask_date = pd.Series(True, index=df.index)

    city_pick = st.multiselect("City", sorted(df["city"].unique()), default=list(df["city"].unique()))
    mask_city = df["city"].isin(city_pick)

df = df[mask_date & mask_city].copy()

# ───── 4.  RFM SEGMENTATION ─────────────────────────────────────────────────
SNAP = (df["booking_date"].max() + timedelta(days=1)) if "booking_date" in df else pd.Timestamp.today()
rfm = (
    df.groupby("user_id")
      .agg(recency=("booking_date", lambda s: (SNAP - s.max()).days if s.notna().any() else 9999),
           frequency=("tickets_sold", "sum"),
           monetary=("ticket_price", "sum"))
      .reset_index()
)
X_scaled = StandardScaler().fit_transform(rfm[["recency","frequency","monetary"]])
rfm["segment"] = KMeans(4, random_state=42, n_init="auto").fit_predict(X_scaled)
sil = silhouette_score(X_scaled, rfm["segment"])
df = df.merge(rfm[["user_id","segment"]], on="user_id", how="left")

# ───── 5.  KPI STRIP ────────────────────────────────────────────────────────
def kpi(val, lbl):
    st.markdown(f'<div class="kpi"><p class="v">{val}</p><p class="l">{lbl}</p></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: kpi(f"{df.tickets_sold.sum():,}", "Tickets Sold")
with c2: kpi(f"{(1 - df.tickets_sold.sum()/df.capacity.sum())*100:,.1f} %", "Vacancy Rate")
with c3: kpi(f"{sil:.2f}", "Segm. Silhouette")

# ───── 6.  SIMPLE PRICE-DEMAND MODEL ────────────────────────────────────────
price_ready = df["ticket_price"].nunique() > 1
if price_ready:
    reg = GradientBoostingRegressor(random_state=42).fit(df[["ticket_price"]], df["tickets_sold"])
    curve_x = np.arange(100, 501, 10)
    curve_y = reg.predict(curve_x.reshape(-1, 1))

# ───── 7.  DASHBOARD TABS ───────────────────────────────────────────────────
tab_cap, tab_price = st.tabs(["📊 Capacity", "💰 Pricing"])

with tab_cap:
    st.subheader("Hourly Seat Occupancy")
    if "show_time" in df:
        hourly = (
            df.assign(hour=df["show_time"].dt.hour)
              .groupby("hour").agg(occ=("tickets_sold","sum"), cap=("capacity","sum"))
              .assign(occ_pct=lambda x: x.occ/x.cap).reset_index()
        )
        st.plotly_chart(
            px.bar(hourly, x="hour", y="occ_pct", labels={"occ_pct":"Occupancy %"}, height=380),
            use_container_width=True,
        )
    else:
        st.warning("*show_time* column missing — occupancy chart skipped.")

with tab_price:
    st.subheader("Price Simulator")
    sel_price = st.slider("Ticket Price (₹)", 100, 500, 250, 25)
    if price_ready:
        st.metric("Predicted Tickets", f"{reg.predict([[sel_price]])[0]:,.0f}")
        fig = go.Figure(go.Scatter(x=curve_x, y=curve_y, mode="lines"))
        fig.update_layout(xaxis_title="Price (₹)", yaxis_title="Predicted Tickets", height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient price variation — demand model skipped.")

# ───── 8.  FOOTER ──────────────────────────────────────────────────────────
st.markdown(
    "<center style='color:#888;font-size:.75rem'>© 2025 • Type-safe build — date/compare errors eliminated ✅</center>",
    unsafe_allow_html=True,
)
