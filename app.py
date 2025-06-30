# app.py — Cinema Performance Command-Center  
# (city filter removed + price-slider protected against single value / NaNs)
# ============================================================================
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import silhouette_score

# ─────────────────── 0. PAGE CONFIG / THEME ────────────────────
st.set_page_config("🎬 Cinema Command-Center", "🎟️", "wide")
st.markdown("""
<style>
html,body,[class*="css"]{font-family:Inter,sans-serif;}
.kpi{background:#fafafa;border-radius:14px;padding:14px;box-shadow:0 2px 6px rgba(0,0,0,.05);}
.kpi .v{font-size:1.7rem;font-weight:700;color:#E50914;text-align:center;}
.kpi .l{font-size:.8rem;color:#555;text-align:center;margin-top:-4px;}
div.js-plotly-plot .modebar{display:none !important;}
</style>""", unsafe_allow_html=True)

# ─────────────────── 1. COLUMN MAP (your headers) ──────────────
MAP = {
    "tickets_sold":  ["total_tickets"],
    "ticket_price":  ["price_per_ticket"],
    "capacity":      ["available_seats"],
    "booking_date":  ["booking_date"],
    "payment_status":["payment_status"],
    "payment_method":["payment_method"],
    "theatre_id":    ["theater_id"],
}
NUM_DEFAULT = {"tickets_sold": 0, "ticket_price": 0.0, "capacity": 1}

def map_cols(df: pd.DataFrame) -> pd.DataFrame:
    lkp = {c.lower(): c for c in df.columns}
    for canon, variants in MAP.items():
        for v in variants:
            if v.lower() in lkp:
                df = df.rename(columns={lkp[v.lower()]: canon}); break
    for c,d in NUM_DEFAULT.items(): df[c] = pd.to_numeric(df.get(c, d), errors="coerce").fillna(d)
    # dates
    for dt in ["booking_date","show_date","start_time"]: 
        if dt in df: df[dt] = pd.to_datetime(df[dt], errors="coerce")
    if {"show_date","start_time"} <= set(df.columns):
        df["show_time"] = df["show_date"] + (df["start_time"] - df["start_time"].dt.normalize())
    df["show_date_norm"] = pd.to_datetime(df.get("show_time")).dt.normalize()
    return df

# ─────────────────── 2. LOAD CSV ───────────────────────────────
@st.cache_data(show_spinner=True)
def load(csv="BookMyShow_Combined_Clean_v2.csv"):
    if not Path(csv).is_file():
        st.error(f"❌ **{csv}** not found. Upload then rerun."); st.stop()
    return map_cols(pd.read_csv(csv))

df = load()

# ─────────────────── 3. SIDEBAR (only date filter) ─────────────
with st.sidebar:
    st.header("Filters")
    if df["show_date_norm"].notna().any():
        dmin,dmax = df["show_date_norm"].min().date(), df["show_date_norm"].max().date()
        start,end = st.date_input("Show Date Range",(dmin,dmax))
        mask_date = df["show_date_norm"].between(pd.Timestamp(start), pd.Timestamp(end))
    else:
        st.info("No valid show dates in file."); mask_date = pd.Series(True,index=df.index)
df = df[mask_date].copy()

# ─────────────────── 4. SEGMENTATION (RFM) ─────────────────────
snap = (df["booking_date"].max()+timedelta(days=1)) if "booking_date" in df else pd.Timestamp.today()
rfm = (df.groupby("user_id")
         .agg(recency=("booking_date",lambda s:(snap-s.max()).days if s.notna().any() else 9999),
              frequency=("tickets_sold","sum"),
              monetary=("ticket_price","sum"))
         .reset_index())
if len(rfm):
    X = StandardScaler().fit_transform(rfm[["recency","frequency","monetary"]])
    rfm["segment"] = KMeans(4,random_state=42,n_init="auto").fit_predict(X)
    sil = silhouette_score(X, rfm.segment)
else: sil = 0
df = df.merge(rfm[["user_id","segment"]],how="left",on="user_id")

# ─────────────────── 5. KPI STRIP ──────────────────────────────
def kpi(v,l): st.markdown(f'<div class="kpi"><p class="v">{v}</p><p class="l">{l}</p></div>',unsafe_allow_html=True)
a,b,c=st.columns(3)
with a:kpi(f"{df.tickets_sold.sum():,}","Tickets Sold")
with b:kpi(f"{(1-df.tickets_sold.sum()/df.capacity.sum())*100:,.1f}%","Vacancy")
with c:kpi(f"{sil:.2f}","Segm. Silhouette")

# ─────────────────── 6. PRICE MODEL (guarded) ──────────────────
price_vals = df["ticket_price"].dropna().unique()
price_ready = len(price_vals) > 1
if price_ready:
    reg = GradientBoostingRegressor(random_state=42).fit(df[["ticket_price"]], df["tickets_sold"])
    p_min,p_max = price_vals.min(), price_vals.max()
    x_curve = np.linspace(p_min, p_max, 30); y_curve = reg.predict(x_curve.reshape(-1,1))

# ─────────────────── 7. DASHBOARD ──────────────────────────────
tab1,tab2 = st.tabs(["📊 Capacity","💰 Pricing"])

with tab1:
    st.subheader("Hourly Seat Occupancy")
    if df["show_time"].notna().any():
        hourly=(df.assign(hour=df.show_time.dt.hour)
                  .groupby("hour").agg(occ=("tickets_sold","sum"),cap=("capacity","sum"))
                  .assign(occ_pct=lambda x:x.occ/x.cap).reset_index())
        st.plotly_chart(px.bar(hourly,x="hour",y="occ_pct",labels={"occ_pct":"Occupancy %"},height=380),
                        use_container_width=True)
    else:
        st.info("No show_time data to plot occupancy.")

with tab2:
    st.subheader("Price Simulator")
    if price_ready:
        sel = st.slider("Ticket Price (₹)", int(p_min), int(p_max), int((p_min+p_max)//2), step=5)
        st.metric("Predicted Tickets", f"{reg.predict([[sel]])[0]:,.0f}")
        fig = go.Figure(go.Scatter(x=x_curve, y=y_curve, mode="lines"))
        fig.update_layout(xaxis_title="Price (₹)", yaxis_title="Predicted Tickets", height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need >1 unique ticket price to fit demand model.")

# ─────────────────── 8. FOOTER ─────────────────────────────────
st.markdown("<center style='color:#888;font-size:.75rem'>© 2025 • Clean build — city removed & slider safeguarded ✅</center>",
            unsafe_allow_html=True)
