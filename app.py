# app.py ─ Cinema Performance Command-Center 2.0
# A polished, investor-grade Streamlit dashboard
# ════════════════════════════════════════════════════════════════════
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

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL THEME CSS
st.set_page_config(
    page_title="🎬 Cinema Command-Center",
    layout="wide",
    page_icon=":clapper:",
    initial_sidebar_state="expanded",
)

THEME_CSS = """
<style>
/* --- GLOBAL --- */
html, body, [class*="css"]  {font-family:'Inter',sans-serif;}
body {background:linear-gradient(120deg,#ffffff,#f7f7f9 30%,#f2f2f4);}

/* --- HEADER --- */
h1.title    {font-size:2.2rem;font-weight:800;color:#E50914;margin-bottom:.5rem;}
h2, h3, h4  {color:#222;}

/* --- KPI BADGES --- */
.kpi-card   {display:flex;flex-direction:column;align-items:center;
             justify-content:center;border-radius:18px;padding:18px;
             background:linear-gradient(135deg,#ffffff 0%,#f0f0f3 100%);
             box-shadow:0 8px 16px rgba(0,0,0,.05);}
.kpi-value  {font-size:1.9rem;font-weight:700;color:#E50914;margin:0;}
.kpi-label  {font-size:.85rem;color:#555;margin-top:-4px;}

/* --- TABS DECORATION --- */
button[data-baseweb="tab"] {font-weight:600;color:#333;}
button[data-baseweb="tab"]:hover {color:#E50914;}

/* --- PLOTLY TOOLBAR OFF --- */
div.js-plotly-plot .modebar {display:none !important;}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# DATA LOAD (+ CACHE)
@st.cache_data(show_spinner=True)
def load_data(csv_path="BookMyShow_Combined_Clean_v2.csv"):
    return pd.read_csv(csv_path, parse_dates=["booking_date", "show_time"])

df_raw = load_data()

# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
def make_features(data: pd.DataFrame):
    d = data.copy()
    d["show_date"] = d["show_time"].dt.date
    snap = d["booking_date"].max() + timedelta(days=1)

    rfm = (
        d.groupby("user_id")
        .agg(
            recency=("booking_date", lambda x: (snap - x.max()).days),
            frequency=("booking_id", "count"),
            monetary=("ticket_amount", "sum"),
        )
        .reset_index()
    ).merge(d[["user_id", "city"]].drop_duplicates(), on="user_id")

    # Segment via K-Means
    sc = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto").fit(sc)
    rfm["segment"] = kmeans.labels_
    sil = silhouette_score(sc, kmeans.labels_)

    return d.merge(rfm[["user_id", "segment"]], on="user_id"), rfm, sil

df, rfm, silhouette = make_features(df_raw)

# ─────────────────────────────────────────────────────────────────────
# MODEL BUILDERS (ALL CACHED)
@st.cache_resource(show_spinner=False)
def prophet_forecast(data: pd.DataFrame):
    daily = (
        data.groupby(["show_id", "show_date"])
        .tickets_sold.sum()
        .reset_index()
        .rename(columns={"show_date": "ds", "tickets_sold": "y"})
    )
    forecasts = []
    for sid in daily.show_id.unique()[:40]:             # cap for speed
        sub = daily[daily.show_id == sid]
        if len(sub) < 15:
            continue
        m = Prophet(weekly_seasonality=True, daily_seasonality=False)
        m.fit(sub[["ds", "y"]])
        fut = m.make_future_dataframe(periods=7)
        fc = m.predict(fut).tail(7)[["ds", "yhat"]]
        fc["show_id"] = sid
        forecasts.append(fc)
    return pd.concat(forecasts) if forecasts else pd.DataFrame()

@st.cache_resource(show_spinner=False)
def train_classifier(df: pd.DataFrame):
    df = df.assign(
        hrs_to_show=lambda x: (x.show_time - x.booking_date).dt.total_seconds() / 3600,
        empty48=lambda x: ((x.hrs_to_show < 48) & (x.tickets_sold == 0)).astype(int),
    )
    X = df[["ticket_price", "hrs_to_show"]]
    y = df["empty48"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    clf = GradientBoostingClassifier().fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    return clf, auc

@st.cache_resource(show_spinner=False)
def churn_and_clv(rfm_df: pd.DataFrame):
    X = rfm_df[["recency", "frequency", "monetary"]]
    y = (rfm_df.recency <= 90).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    lg = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    rfm_df["repurchase_p"] = lg.predict_proba(X)[:, 1]
    auc = roc_auc_score(yte, lg.predict_proba(Xte)[:, 1])

    lt = (
        df.groupby("user_id")
        .agg(
            freq=("booking_id", "count"),
            rec=("booking_date", lambda s: (s.max() - s.min()).days),
            T=("booking_date", lambda s: (df.booking_date.max() - s.min()).days),
            mon=("ticket_amount", "mean"),
        )
    )
    bgf = BetaGeoFitter().fit(lt.freq, lt.rec, lt.T)
    ggf = GammaGammaFitter().fit(lt.freq, lt.mon)
    lt["clv"] = ggf.customer_lifetime_value(bgf, lt.freq, lt.rec, lt.T, lt.mon, time=6, freq="D")
    return rfm_df.merge(lt.clv, on="user_id"), auc

@st.cache_resource(show_spinner=False)
def price_model(data: pd.DataFrame):
    X = data[["ticket_price"]]
    y = data.tickets_sold
    reg = GradientBoostingRegressor().fit(X, y)
    rmse = np.sqrt(((reg.predict(X) - y) ** 2).mean())
    return reg, rmse

@st.cache_resource(show_spinner=False)
def slot_clusters(data: pd.DataFrame):
    data["slot"] = data.show_time.dt.hour
    agg = data.groupby("slot").tickets_sold.sum().reset_index()
    km = KMeans(n_clusters=5, random_state=42, n_init="auto")
    agg["cluster"] = km.fit_predict(StandardScaler().fit_transform(agg[["tickets_sold"]]))
    agg["demand_index"] = agg.groupby("cluster").tickets_sold.transform("mean")
    agg["demand_index"] = agg.demand_index.rank(pct=True)
    return agg.sort_values("slot")

@st.cache_resource(show_spinner=False)
def pay_fail(df: pd.DataFrame):
    if not {"payment_status", "payment_method"}.issubset(df.columns):
        return None, None
    df2 = df.assign(fail=(df.payment_status == "Failed").astype(int))
    X = pd.get_dummies(df2[["payment_method", "city"]])
    y = df2.fail
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    clf = GradientBoostingClassifier().fit(Xtr, ytr)
    return clf, roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

# ─────────────────────────────────────────────────────────────────────
# MODEL EXECUTION
with st.spinner("Crunching numbers …"):
    df_fc      = prophet_forecast(df)
    seat_clf, seat_auc = train_classifier(df)
    rfm, churn_auc     = churn_and_clv(rfm)
    price_reg, price_rmse = price_model(df)
    slots = slot_clusters(df)
    pay_clf, pay_auc = pay_fail(df)

df = df.merge(rfm[["user_id", "repurchase_p", "clv"]], on="user_id")

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/movie-projector.png", width=96)
    st.markdown("### Filter Audience")
    date_range = st.date_input("Show Date", (df.show_date.min(), df.show_date.max()))
    city_opt   = st.multiselect("City", df.city.unique(), default=list(df.city.unique()))
    seg_opt    = st.multiselect("Segment", rfm.segment.unique(), default=list(rfm.segment.unique()))
    st.markdown("---")
    st.markdown("Made with ❤️ & Streamlit")

mask = (
    (df.show_date.between(*date_range))
    & (df.city.isin(city_opt))
    & (df.segment.isin(seg_opt))
)
df_filt = df[mask]

# ─────────────────────────────────────────────────────────────────────
# PAGE HEADER
st.markdown('<h1 class="title">Cinema Performance Command-Center</h1>', unsafe_allow_html=True)
st.write("A 360-degree cockpit for capacity, customer, pricing, schedule and payment health—powered by machine learning.")

# ─────────────────────────────────────────────────────────────────────
# KPI STRIP
def kpi(val, label, icon="🎯"):
    col = st.columns(1)[0]
    col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{icon} {val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

k1 = f"{df_fc.yhat.mean():,.0f}" if not df_fc.empty else "–"
k2 = f"{(1 - df_filt.tickets_sold.sum() / df_filt.capacity.sum())*100:,.1f} %" if "capacity" in df_filt else "–"
k3 = f"{rfm.repurchase_p.mean():.2f}"
k4 = f"₹{rfm.clv.mean():,.0f}"
k5 = f"{price_rmse:,.1f}"
k6 = f"{silhouette:.2f}"

col1, col2, col3, col4, col5, col6 = st.columns(6, gap="small")
with col1: kpi(k1, "Avg 7-Day Demand")
with col2: kpi(k2, "Vacancy Rate %")
with col3: kpi(k3, "Re-purchase P")
with col4: kpi(k4, "Mean CLV")
with col5: kpi(k5, "Price RMSE")
with col6: kpi(k6, "Segm. Silhouette")

# ─────────────────────────────────────────────────────────────────────
# TAB NAVIGATION
tabs = st.tabs(
    ["📊 Capacity", "💟 Lifecycle", "💰 Pricing", "🗓️ Schedule", "🔒 Payment"]
)
# ── TAB 1 CAPACITY ────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Show-Level Demand Forecast")
    if df_fc.empty:
        st.info("Not enough history for forecast yet.")
    else:
        fig = px.line(df_fc, x="ds", y="yhat", color="show_id", height=350,
                      labels={"ds":"Date","yhat":"Tickets"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Seven-day Prophet forecast per show enables proactive capacity actions.")

    if "capacity" in df_filt:
        heat = (
            df_filt.assign(hour=df_filt.show_time.dt.hour)
            .groupby(["theatre_id","hour"])
            .agg(occ=("tickets_sold","sum"), cap=("capacity","sum"))
            .assign(occ_pct=lambda x: x.occ/x.cap)
            .reset_index()
        )
        fig2 = px.density_heatmap(heat, x="hour", y="theatre_id",
                                  z="occ_pct", color_continuous_scale="Reds",
                                  height=350)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Red cells signal chronic under-utilisation by slot and theatre.")

    st.expander("Model Metrics").write(f"Seat-emptiness classifier AUC = {seat_auc:.3f}")

# ── TAB 2 LIFECYCLE ──────────────────────────────────────────────
with tabs[1]:
    st.subheader("Customer Segment Breakdown")
    seg_bar = px.bar(
        rfm.segment.value_counts().rename_axis("segment").reset_index(name="users"),
        x="segment", y="users", text="users", height=330, color="segment",
        color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(seg_bar, use_container_width=True)
    st.caption("Relative size of behavioural segments (K-Means, K = 4).")

    st.subheader("Repurchase Probability Deciles")
    rfm["decile"] = pd.qcut(rfm.repurchase_p, 10, labels=False)
    dec = rfm.groupby("decile").repurchase_p.mean().reset_index()
    dec_fig = px.bar(dec, x="decile", y="repurchase_p", text="repurchase_p",
                     height=280, labels={"repurchase_p":"Avg P"})
    st.plotly_chart(dec_fig, use_container_width=True)
    st.caption("Higher deciles = warmer leads for loyalty triggers.")

    st.expander("Model Metrics").markdown(
        f"- Churn AUC = **{churn_auc:.3f}**   \n- Avg CLV = **₹{rfm.clv.mean():,.0f}**"
    )

# ── TAB 3 PRICING ────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Elastic Demand Curve")
    sel_price = st.slider("Ticket Price (₹)", 100, 500, 250, 25)
    demand_pred = price_reg.predict([[sel_price]])[0]
    st.metric("Projected Tickets", f"{demand_pred:,.0f}")

    curve_x = np.arange(100, 501, 10)
    curve_y = price_reg.predict(curve_x.reshape(-1, 1))
    fig = go.Figure(go.Scatter(x=curve_x, y=curve_y, mode="lines"))
    fig.update_layout(xaxis_title="Price (₹)", yaxis_title="Predicted Tickets", height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Identify revenue-optimal pricing sweet spots.")

    st.expander("Model Metrics").write(f"Gradient-Boost RMSE = {price_rmse:,.1f}")

# ── TAB 4 SCHEDULE ───────────────────────────────────────────────
with tabs[3]:
    st.subheader("Slot Demand Index")
    fig = px.bar(slots, x="slot", y="demand_index", color="cluster",
                 height=330, labels={"slot":"Hour","demand_index":"Index"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hours with higher indices warrant more screens or premium pricing.")

    uplift = st.slider("Schedule Uplift (%)", 0, 100, 15, 5)
    base = df_filt.tickets_sold.sum()*df_filt.ticket_price.mean()
    st.metric("Projected Monthly Revenue", f"₹{base*(1+uplift/100):,.0f}", delta=f"{uplift}%")

# ── TAB 5 PAYMENT ────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Payment Failure Risk Map")
    if pay_clf:
        risk = pay_clf.predict_proba(pd.get_dummies(df[["payment_method","city"]]))[:,1]
        risk_df = df.assign(risk=risk).groupby(["payment_method","city"]).risk.mean().reset_index()
        fig = px.density_heatmap(risk_df, x="payment_method", y="city", z="risk",
                                 color_continuous_scale="Blues", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Target backup rails & incentives where risk is elevated.")
        st.expander("Model Metrics").write(f"Payment-risk AUC = {pay_auc:.3f}")
    else:
        st.info("Payment-level data unavailable in this file.")

# ─────────────────────────────────────────────────────────────────────
st.markdown("""<center style='color:#999;font-size:.8rem'>
© 2025 Cinema Performance • Crafted with Streamlit • Accent #E50914
</center>""", unsafe_allow_html=True)
