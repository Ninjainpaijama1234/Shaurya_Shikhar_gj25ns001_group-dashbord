# app.py — Cinema Performance Command-Center (aligned to your column names)
# =========================================================================
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import silhouette_score

# ─────────────────────────── 0. THEME ────────────────────────────
st.set_page_config("🎬 Cinema Command-Center", "🎟️", "wide")
st.markdown("""
<style>
html,body,[class*="css"]{font-family:Inter,sans-serif;}
.kpi{background:#fafafa;border-radius:14px;padding:14px;box-shadow:0 2px 6px rgba(0,0,0,.05);}
.kpi .v{font-size:1.7rem;font-weight:700;color:#E50914;text-align:center;}
.kpi .l{font-size:.8rem;color:#555;text-align:center;margin-top:-4px;}
div.js-plotly-plot .modebar{display:none !important;}
</style>""", unsafe_allow_html=True)

# ───── 1. COLUMN MAPPING TO YOUR HEADERS ─────────────────────────
COL_MAP = {
    "tickets_sold":  ["total_tickets"],
    "ticket_price":  ["price_per_ticket"],
    "capacity":      ["available_seats"],
    # we’ll construct show_time from show_date + start_time, so no synonym needed
    "booking_date":  ["booking_date"],
    "payment_method":["payment_method"],
    "payment_status":["payment_status"],
    "theatre_id":    ["theater_id"],   # US spelling → canonical
}

NUM_DEFAULT = {"tickets_sold": 0, "ticket_price": 0.0, "capacity": 1}
CAT_DEFAULT = {"city": "Unknown"}      # dataset has no city; created on the fly

def auto_map(df: pd.DataFrame) -> pd.DataFrame:
    """Rename first matching synonym in COL_MAP to canonical name."""
    lower_lookup = {c.lower(): c for c in df.columns}
    for canon, aliases in COL_MAP.items():
        for a in aliases:
            if a.lower() in lower_lookup:
                df = df.rename(columns={lower_lookup[a.lower()]: canon})
                break
    return df

def prepare_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = auto_map(raw)

    # fill missing mandatory columns
    for c, d in NUM_DEFAULT.items():
        if c not in df: df[c] = d
    for c, d in CAT_DEFAULT.items():
        if c not in df: df[c] = d

    # robust datetime parsing
    for dt_col in ["booking_date", "show_date", "start_time"]:
        if dt_col in df: df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    # build show_time = show_date + start_time (if both present)
    if "show_date" in df and "start_time" in df:
        df["show_time"] = df["show_date"] + (df["start_time"] - df["start_time"].dt.normalize())
    elif "show_time" in df:
        df["show_time"] = pd.to_datetime(df["show_time"], errors="coerce")
    else:
        df["show_time"] = pd.NaT

    # normalised date for filters
    if "show_time" in df:
        df["show_date_norm"] = df["show_time"].dt.normalize()
    return df

# ───── 2. LOAD CSV ───────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_csv(path="BookMyShow_Combined_Clean_v2.csv"):
    if not Path(path).is_file():
        st.error(f"🛑  **{path}** not found. Upload it, then click *Rerun*.")
        st.stop()
    return prepare_df(pd.read_csv(path))

df = load_csv()

# ───── 3. SIDEBAR FILTERS ────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    if df["show_date_norm"].notna().any():
        dmin, dmax = df["show_date_norm"].min().date(), df["show_date_norm"].max().date()
        start_d, end_d = st.date_input("Show Date Range", (dmin, dmax))
        mask_date = df["show_date_norm"].between(pd.Timestamp(start_d), pd.Timestamp(end_d))
    else:
        st.info("No valid show dates — date filter disabled.")
        mask_date = pd.Series(True, index=df.index)

    if "city" in df:
        cities = st.multiselect("City", sorted(df["city"].unique()), default=list(df["city"].unique()))
        mask_city = df["city"].isin(cities)
    else:
        mask_city = pd.Series(True, index=df.index)

df = df[mask_date & mask_city].copy()

# ───── 4. SEGMENTATION (RFM) ─────────────────────────────────────
snap = (df["booking_date"].max() + timedelta(days=1)) if "booking_date" in df else pd.Timestamp.today()
rfm = (
    df.groupby("user_id")
      .agg(recency=("booking_date", lambda s:(snap-s.max()).days if s.notna().any() else 9999),
           frequency=("tickets_sold","sum"),
           monetary=("ticket_price","sum"))
      .reset_index()
)
if len(rfm):
    X = StandardScaler().fit_transform(rfm[["recency","frequency","monetary"]])
    rfm["segment"] = KMeans(4, random_state=42, n_init="auto").fit_predict(X)
    sil = silhouette_score(X, rfm["segment"])
else:
    rfm["segment"] = []; sil = 0
df = df.merge(rfm[["user_id","segment"]], on="user_id", how="left")

# ───── 5. KPI STRIP ──────────────────────────────────────────────
def kpi(v,l): st.markdown(f'<div class="kpi"><p class="v">{v}</p><p class="l">{l}</p></div>', unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)
with c1: kpi(f"{df.tickets_sold.sum():,}", "Tickets Sold")
with c2: kpi(f"{(1 - df.tickets_sold.sum()/df.capacity.sum())*100:,.1f} %", "Vacancy")
with c3: kpi(f"{sil:.2f}", "Segm. Silhouette")

# ───── 6. PRICE DEMAND MODEL ────────────────────────────────────
price_ready = df["ticket_price"].nunique() > 1
if price_ready:
    reg = GradientBoostingRegressor(random_state=42).fit(df[["ticket_price"]], df["tickets_sold"])
    px_vals = np.linspace(df["ticket_price"].min(), df["ticket_price"].max(), 30)
    py_vals = reg.predict(px_vals.reshape(-1,1))

# ───── 7. TABS ───────────────────────────────────────────────────
tab_cap, tab_price = st.tabs(["📊 Capacity", "💰 Pricing"])

with tab_cap:
    st.subheader("Hourly Seat Occupancy")
    if df["show_time"].notna().any():
        hourly = (df.assign(hour=df["show_time"].dt.hour)
                    .groupby("hour").agg(occ=("tickets_sold","sum"), cap=("capacity","sum"))
                    .assign(occ_pct=lambda x:x.occ/x.cap).reset_index())
        st.plotly_chart(px.bar(hourly, x="hour", y="occ_pct", labels={"occ_pct":"Occupancy %"}, height=380),
                        use_container_width=True)
    else:
        st.info("No *show_time* data to plot occupancy.")

with tab_price:
    st.subheader("Price Simulator")
    sel = st.slider("Ticket Price (₹)", int(df["ticket_price"].min()), int(df["ticket_price"].max()), step=5)
    if price_ready:
        st.metric("Predicted Tickets", f"{reg.predict([[sel]])[0]:,.0f}")
        fig = go.Figure(go.Scatter(x=px_vals, y=py_vals, mode="lines"))
        fig.update_layout(xaxis_title="Price (₹)", yaxis_title="Predicted Tickets", height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need >1 unique ticket price for demand model.")

# ───── 8. FOOTER ────────────────────────────────────────────────
st.markdown("<center style='color:#888;font-size:.75rem'>© 2025 • Column-mapped build now using your dataset 📈</center>", unsafe_allow_html=True)
