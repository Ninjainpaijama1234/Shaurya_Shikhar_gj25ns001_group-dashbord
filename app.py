"""
Cinema Performance Command‑Center 2.1  •  Investor‑grade Streamlit dashboard
Robust to schema variations in BookMyShow_Combined_Clean_v2.csv
Author: Senior Data‑Science Architect – 01 Jul 2025
"""

# ═══════════════════════════════════════════════════════════════════
# ░ Imports ░
# ────────────────────────────────────────────────────────────────────
import warnings, sys, os, typing as _t
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lifetimes import BetaGeoFitter, GammaGammaFitter
from mlxtend.frequent_patterns import apriori, association_rules
from prophet import Prophet
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# ░ Page settings & theme CSS ░
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Cinema Command‑Center",
    layout="wide",
    page_icon=":clapper:",
    initial_sidebar_state="expanded",
)
THEME="""
<style>
html, body, [class*="css"]  {font-family:'Inter',sans-serif;}
body {background:linear-gradient(120deg,#ffffff,#f8f9fb 30%,#f2f3f7);} 
.kpi-card{display:flex;flex-direction:column;align-items:center;justify-content:center;
          border-radius:18px;padding:1.1rem 1rem;background:#ffffffdd;box-shadow:0 8px 18px rgba(0,0,0,.05);} 
.kpi-value{font-size:1.85rem;font-weight:700;color:#E50914;margin:0;} 
.kpi-label{font-size:.85rem;color:#555;margin-top:-2px;} 
button[data-baseweb="tab"]{font-weight:600;} 
button[data-baseweb="tab"][aria-selected="true"]{color:#E50914;border-bottom:2px solid #E50914;} 
div.js-plotly-plot .modebar{display:none!important;}
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# ░ Data Loader – resilient to column‑name drift ░
# ────────────────────────────────────────────────────────────────────
BOOK_COL_ALIASES = ["booking_date", "booking_datetime", "booking_dt", "booking_time"]
SHOW_COL_ALIASES  = ["show_time", "show_datetime", "show_dt", "showtime"]

@st.cache_data(show_spinner=True)
def load_data(path:str="BookMyShow_Combined_Clean_v2.csv") -> pd.DataFrame:
    """Load CSV even if date columns differ; coerce canonical names."""
    if not os.path.exists(path):
        st.error(f"✖ Data file '{path}' not found in root directory.")
        st.stop()

    df = pd.read_csv(path)

    # ─ Canonicalise booking_date ─
    booking_col = next((c for c in BOOK_COL_ALIASES if c in df.columns), None)
    if booking_col:
        df["booking_date"] = pd.to_datetime(df[booking_col], errors="coerce")
    else:
        df["booking_date"] = pd.NaT  # placeholder to keep downstream code intact

    # ─ Canonicalise show_time ─
    show_col = next((c for c in SHOW_COL_ALIASES if c in df.columns), None)
    if show_col:
        df["show_time"] = pd.to_datetime(df[show_col], errors="coerce")
    else:
        df["show_time"] = pd.NaT

    # Ensure key numeric columns exist to avoid KeyErrors later
    for col, default in {
        "tickets_sold": 0,
        "capacity": 1,
        "ticket_price": df.get("ticket_price", pd.Series([0])).median() or 0,
    }.items():
        if col not in df.columns:
            df[col] = default

    # Fallback for city / show_id / user_id
    for col in ["city", "show_id", "user_id", "booking_id"]:
        if col not in df.columns:
            df[col] = "Unknown" if df[col].dtype==object else 0
    
    df["show_date"] = df["show_time"].dt.date
    return df

df_raw = load_data()

# ═══════════════════════════════════════════════════════════════════
# ░ Feature engineering & segmentation ░
# ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def make_features(base: pd.DataFrame):
    data = base.copy()

    # --- RFM ---
    snap = data["booking_date"].dropna().max() + timedelta(days=1)
    rfm = (
        data.dropna(subset=["booking_date"]).groupby("user_id").agg(
            recency=("booking_date", lambda x: (snap - x.max()).days),
            frequency=("booking_id", "count"),
            monetary=("ticket_amount" if "ticket_amount" in data.columns else "ticket_price", "sum"),
        ).reset_index()
    )
    if "city" in data.columns:
        rfm = rfm.merge(data[["user_id", "city"]].drop_duplicates(), on="user_id", how="left")

    scaler = StandardScaler()
    k_features = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    kmeans   = KMeans(n_clusters=4, random_state=42, n_init="auto").fit(k_features)
    rfm["segment"] = kmeans.labels_
    silhouette = silhouette_score(k_features, kmeans.labels_)

    data = data.merge(rfm[["user_id", "segment"]], on="user_id", how="left")
    return data, rfm, silhouette

df, rfm, sil_score = make_features(df_raw)

# ═══════════════════════════════════════════════════════════════════
# ░ Helper: safe KPI calculation ░
# ────────────────────────────────────────────────────────────────────

def safe_div(num, den):
    return num / den if den else np.nan

# ═══════════════════════════════════════════════════════════════════
# ░ Model builders – wrapped in try/except so dashboard remains live ░
# ────────────────────────────────────────────────────────────────────

from functools import lru_cache

@lru_cache(maxsize=None)
def prophet_forecast(df_: pd.DataFrame):
    try:
        daily = (
            df_.dropna(subset=["show_date"]).groupby(["show_id", "show_date"])
            .tickets_sold.sum().reset_index().rename(columns={"show_date": "ds", "tickets_sold": "y"})
        )
        fc_list = []
        for sid, grp in daily.groupby("show_id"):
            if len(grp) < 10:
                continue
            m = Prophet(weekly_seasonality=True, daily_seasonality=False)
            m.fit(grp[["ds", "y"]])
            future = m.make_future_dataframe(periods=7)
            pred = m.predict(future).tail(7)[["ds", "yhat"]]
            pred["show_id"] = sid
            fc_list.append(pred)
        return pd.concat(fc_list) if fc_list else pd.DataFrame()
    except Exception as e:
        st.warning(f"Forecast model skipped: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=None)
def seat_classifier(df_: pd.DataFrame):
    try:
        tmp = df_.dropna(subset=["booking_date", "show_time"]).copy()
        tmp["hrs_to_show"] = (tmp["show_time"] - tmp["booking_date"]).dt.total_seconds() / 3600
        tmp["empty48"] = ((tmp["hrs_to_show"] < 48) & (tmp["tickets_sold"] == 0)).astype(int)
        X = tmp[["ticket_price", "hrs_to_show"]]
        y = tmp["empty48"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
        clf = GradientBoostingClassifier().fit(Xtr, ytr)
        return clf, roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
    except Exception as e:
        st.warning(f"Seat classifier skipped: {e}")
        return None, np.nan

@lru_cache(maxsize=None)
def churn_and_clv(rfm: pd.DataFrame):
    try:
        X = rfm[["recency", "frequency", "monetary"]]
        y = (rfm.recency <= 90).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
        lg = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        rfm["repurchase_p"] = lg.predict_proba(X)[:,1]
        auc = roc_auc_score(yte, lg.predict_proba(Xte)[:,1])

        lt = (
            df.groupby("user_id").agg(
                freq=("booking_id", "count"),
                rec=("booking_date", lambda s: (s.max() - s.min()).days),
                T=("booking_date", lambda s: (df.booking_date.max() - s.min()).days),
                mon=("ticket_price", "mean"),
            )
        )
        bgf = BetaGeoFitter().fit(lt.freq, lt.rec, lt.T)
        ggf = GammaGammaFitter().fit(lt.freq, lt.mon)
        lt["clv"] = ggf.customer_lifetime_value(bgf, lt.freq, lt.rec, lt.T, lt.mon, time=6, freq="D")
        rfm = rfm.merge(lt.clv, on="user_id")
        return rfm, auc
    except Exception as e:
        st.warning(f"Churn/CLV models skipped: {e}")
        rfm["repurchase_p"] = 0.0
        rfm["clv"] = 0.0
        return rfm, np.nan

@lru_cache(maxsize=None)
def price_model(df_: pd.DataFrame):
    try:
        reg = GradientBoostingRegressor().fit(df_[["ticket_price"]], df_.tickets_sold)
        rmse = np.sqrt(((reg.predict(df_[["ticket_price"]]) - df_.tickets_sold) ** 2).mean())
        return reg, rmse
    except Exception as e:
        st.warning(f"Price model skipped: {e}")
        return None, np.nan

@lru_cache(maxsize=None)
def slot_clusters(df_: pd.DataFrame):
    try:
        df_ = df_.copy(); df_["slot"] = df_.show_time.dt.hour.fillna(-1).astype(int)
        agg = df_.groupby("slot").tickets_sold.sum().reset_index()
        km = KMeans(n_clusters=min(5,len(agg)), random_state=42, n_init="auto").fit(StandardScaler().fit_transform(agg[["tickets_sold"]]))
        agg["cluster"] = km.labels_
        agg["demand_index"] = agg.groupby("cluster").tickets_sold.transform("mean").rank(pct=True)
        return agg.sort_values("slot")
    except Exception as e:
        st.warning(f"Slot clustering skipped: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=None)
def payment_risk(df_: pd.DataFrame):
    if not {"payment_status","payment_method"}.issubset(df_.columns):
        return None, np.nan
    try:
        temp = df_.assign(fail=(df_.payment_status=="Failed").astype(int))
        X = pd.get_dummies(temp[["payment_method","city"]])
        y = temp.fail
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
        clf = GradientBoostingClassifier().fit(Xtr, ytr)
        return clf, roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
    except Exception as e:
        st.warning(f"Payment risk model skipped: {e}")
        return None, np.nan

# ═══════════════════════════════════════════════════════════════════
# ░ Execute models (lazy‑cached) ░
# ────────────────────────────────────────────────────────────────────
with st.spinner("🔄 Training models …"):
    fc_df        = prophet_forecast(df)
    seat_clf, seat_auc = seat_classifier(df)
    rfm, churn_auc     = churn_and_clv(rfm)
    price_reg, price_rmse = price_model(df)
    slot_df             = slot_clusters(df)
    pay_clf, pay_auc    = payment_risk(df)
    df = df.merge(rfm[["user_id","repurchase_p","clv"]], on="user_id", how="left")

# ═══════════════════════════════════════════════════════════════════
# ░ Sidebar filters ░
# ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/movie-projector.png", width=96)
    st.markdown("### 🔍 Filters")
    st.markdown("Filter the dashboards in real‑time.")
    date_min, date_max = df.show_date.min(), df.show_date.max()
    date_range = st.date_input("Show Date", (date_min, date_max))
    city_sel = st.multiselect("City", sorted(df.city.unique()), default=list(df.city.unique()))
    seg_sel  = st.multiselect("Segment", sorted(rfm.segment.unique()), default=list(rfm.segment.unique()))
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), file_name="cinema_clean.csv")

mask = (
    df.show_date.between(*date_range) &
    df.city.isin(city_sel) &
    df.segment.isin(seg_sel)
)
sub = df[mask]

# ═══════════════════════════════════════════════════════════════════
# ░ Header & KPI strip ░
# ────────────────────────────────────────────────────────────────────
st.markdown("""<h1 style='font-weight:800;color:#E50914;margin-bottom:0.2rem;'>Cinema Performance Command‑Center</h1>""", unsafe_allow_html=True)
st.write("One dashboard. Every lever.")

kpi_vals = [
    (f"{fc_df.yhat.mean():,.0f}" if not fc_df.empty else "–", "Avg 7‑Day Demand"),
    (f"{safe_div(sub.tickets_sold.sum(), sub.capacity.sum())*100:,.1f} %" if "capacity" in sub else "–", "Vacancy Rate"),
    (f"{rfm.repurchase_p.mean():.2f}", "Re‑purchase P"),
    (f"₹{rfm.clv.mean():,.0f}", "Mean CLV"),
    (f"{price_rmse:,.1f}", "Price RMSE"),
    (f"{sil_score:.2f}", "Segm. Silhouette"),
]
cols = st.columns(len(kpi_vals), gap="small")
for c,(val,label) in zip(cols,kpi_vals):
    c.markdown(f"<div class='kpi-card'><div class='kpi-value'>{val}</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# ░ Tabbed dashboards ░
# ────────────────────────────────────────────────────────────────────
capacity_t, lifecycle_t, pricing_t, schedule_t, payment_t = st.tabs(["📊 Capacity","🎟️ Lifecycle","💰 Pricing","🗓️ Schedule","🔒 Payment"])

with capacity_t:
    st.subheader("Demand Forecast (7‑Day)")
    if fc_df.empty:
        st.info("Not enough history to generate forecasts.")
    else:
        fig_fc = px.line(fc_df, x="ds", y="yhat", color="show_id", height=350, labels={"ds":"Date","yhat":"Tickets"})
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption("Prophet‑based forecast for each show – drives proactive capacity management.")

    if "capacity" in sub.columns:
        heat = (
            sub.assign(hour=sub.show_time.dt.hour.fillna(-1))
               .groupby(["theatre_id","hour"]).agg(obs=("tickets_sold","sum"), cap=("capacity","sum"))
               .assign(fill=lambda x: safe_div(x.obs, x.cap))
               .reset_index()
        )
        fig_h = px.density_heatmap(heat, x="hour", y="theatre_id", z="fill", color_continuous_scale="Reds", height=330)
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption("Red indicates under‑utilised pockets by theatre × hour.")

    st.expander("Model Metrics").write(f"Seat‑emptiness classifier AUC = {seat_auc:.3f}" if not np.isnan(seat_auc) else "Model unavailable")

with lifecycle_t:
    st.subheader("Segment Distribution")
    fig_seg = px.histogram(rfm, x="segment", color="segment", height=300, text_auto=True)
    st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("Repurchase Probability Deciles")
    rfm["decile"] = pd.qcut(rfm.repurchase_p, 10, labels=False)
    dec = rfm.groupby("decile").repurchase_p.mean().reset_index()
    fig_dec = px.bar(dec, x="decile", y="repurchase_p", height=280, text_auto=True)
    st.plotly_chart(fig_dec, use_container_width=True)

    st.expander("Model Metrics").markdown(f"- Churn AUC = **{churn_auc:.3f}**\n- CLV avg = **₹{rfm.clv.mean():,.0f}**")

with pricing_t:
    st.subheader("Demand Curve Simulator")
    price_sel = st.slider("Ticket Price (₹)", 100, 500, 250, 10)
    if price_reg:
        pred = price_reg.predict([[price_sel]])[0]
        st.metric("Predicted Tickets", f"{pred:,.0f}")
        xs = np.arange(100,501,10)
        ys = price_reg.predict(xs.reshape(-1,1))
        fig_curve = go.Figure(go.Scatter(x=xs, y=ys, mode="lines"))
        fig_curve.update_layout(height=320, xaxis_title="Price", yaxis_title="Tickets")
        st.plotly_chart(fig_curve, use_container_width=True)
    else:
        st.info("Price model unavailable.")

    st.expander("Model Metrics").write(f"Price RMSE = {price_rmse:,.1f}" if not np.isnan(price_rmse) else "–")

with schedule_t:
    st.subheader("Slot Demand Index")
    if slot_df.empty:
        st.info("Slot analysis unavailable.")
    else:
        fig_slot = px.bar(slot_df, x="slot", y="demand_index", color="cluster", height=320)
        st.plotly_chart(fig_slot, use_container_width=True)
        st.caption("Allocate more screens to high‑index hours.")

    uplift = st.slider("Hypothetical Occupancy Uplift (%)", 0, 100, 15, 5)
    rev_base = sub.tickets_sold.sum()*sub.ticket_price.mean()
    st.metric("Projected Monthly Revenue", f"₹{rev_base*(1+uplift/100):,.0f}", delta=f"{uplift}%")

with payment_t:
    st.subheader("Payment‑Failure Risk")
    if pay_clf is None:
        st.info("Payment columns not present in dataset.")
    else:
        risk = pay_clf.predict_proba(pd.get_dummies(df[["payment_method","city"]]))[:,1]
        risk_df = df.assign(risk=risk).groupby(["payment_method","city"]).risk.mean().reset_index()
        fig_risk = px.density_heatmap(risk_df, x="payment_method", y="city", z="risk", color_continuous_scale="Blues", height=330)
        st.plotly_chart(fig_risk, use_container_width=True)
        st.caption("Pinpoint high‑risk rails × geographies.")
        st.expander("Model Metrics").write(f"AUC = {pay_auc:.3f}")

# ═══════════════════════════════════════════════════════════════════
# ░ Footer ░
# ────────────────────────────────────────────────────────────────────
st.markdown("""<center style='color:#999;font-size:.8rem'>© 2025 Cinema Performance • Built with Streamlit • Accent #E50914</center>""", unsafe_allow_html=True)
