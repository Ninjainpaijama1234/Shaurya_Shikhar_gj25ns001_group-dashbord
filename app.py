# app.py — Cinema Performance Command‑Center ✨ 2025 (beautified + safe plotting)
# =============================================================================
# Features
#   • Glow KPI cards & neumorphic panels
#   • K‑Means customer segmentation
#   • Genre‑level association‑rule mining
#   • Four classifiers (K‑NN, DecisionTree, RandomForest, GradientBoost)
#   • Interactive predictor for High‑Value customers
#   • Robust checks to avoid empty‑data errors (Plotly ValueError fixed)
# -----------------------------------------------------------------------------
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
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlxtend.frequent_patterns import apriori, association_rules

# ────────────────────────── PAGE CONFIG / THEME ──────────────────────────────
st.set_page_config(page_title="Cinema Command‑Center", page_icon="🎬", layout="wide")

THEME_CSS = """
<style>
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]>div:first-child{background:linear-gradient(180deg,#202632 0%,#1e2228 100%);}
.kpi{background:#ffffff;border-radius:18px;padding:18px;box-shadow:0 8px 18px rgba(0,0,0,.08);transition:all .3s;}
.kpi:hover{transform:translateY(-4px);box-shadow:0 12px 24px rgba(0,0,0,.14);}
.kpi .v{font-size:2rem;font-weight:800;color:#E50914;text-align:center;margin:0;}
.kpi .l{font-size:.8rem;text-align:center;color:#666;margin-top:-6px;}
div.js-plotly-plot .modebar{display:none !important;}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ────────────────────────── COLUMN MAP ───────────────────────────────────────
RAW_MAP = {
    "tickets": ["total_tickets"],
    "price"  : ["price_per_ticket"],
    "cap"    : ["available_seats"],
    "bdate"  : ["booking_date"],
    "sdate"  : ["show_date"],
    "stime"  : ["start_time"],
    "genre"  : ["genre"],
}
DEF_NUM = {"tickets":0, "price":0.0, "cap":1}


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, coerce dtypes, derive show_time & show_day."""
    lower = {c.lower(): c for c in df.columns}
    for canon, alts in RAW_MAP.items():
        for a in alts:
            if a.lower() in lower:
                df = df.rename(columns={lower[a.lower()]: canon}); break
    for col, default in DEF_NUM.items():
        df[col] = pd.to_numeric(df.get(col, default), errors="coerce").fillna(default)
    for dt in ["bdate", "sdate", "stime"]:
        if dt in df:
            df[dt] = pd.to_datetime(df[dt], errors="coerce")
    if {"sdate", "stime"}.issubset(df.columns):
        df["show_time"] = df["sdate"] + (df["stime"] - df["stime"].dt.normalize())
    df["show_day"] = pd.to_datetime(df.get("show_time")).dt.normalize()
    df["sales"] = df["tickets"] * df["price"]
    return df


@st.cache_data(show_spinner=True)
def load_csv(path="BookMyShow_Combined_Clean_v2.csv") -> pd.DataFrame:
    if not Path(path).is_file():
        st.error("🗂 CSV not found. Upload file then click Rerun.")
        st.stop()
    return tidy(pd.read_csv(path))

df = load_csv()

# ────────────────────────── SIDEBAR FILTERS ────────────────────────────────
with st.sidebar:
    st.title("Filters")
    if df["show_day"].notna().any():
        dmin, dmax = df["show_day"].min().date(), df["show_day"].max().date()
        start_date, end_date = st.date_input("Show window", (dmin, dmax))
        df = df[df["show_day"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]

# ────────────────────────── K‑MEANS SEGMENTATION ───────────────────────────
snapshot = (df["bdate"].max() + timedelta(days=1)) if "bdate" in df else pd.Timestamp.today()
rfm = (df.groupby("user_id")
         .agg(rec=("bdate", lambda s: (snapshot - s.max()).days if s.notna().any() else 9999),
              freq=("tickets", "sum"),
              mon=("sales", "sum"))
         .reset_index())
if len(rfm):
    scaler = StandardScaler().fit_transform(rfm[["rec", "freq", "mon"]])
    kmeans = KMeans(5, random_state=42, n_init="auto").fit(scaler)
    rfm["cluster"] = kmeans.labels_
    sil = silhouette_score(scaler, kmeans.labels_)
else:
    sil = 0; rfm["cluster"] = 0
segment_labels = {0: "New", 1: "One‑Timer", 2: "Low Value", 3: "Mid Value", 4: "High Value"}
rfm["segment"] = rfm["cluster"].map(segment_labels)

df = df.merge(rfm[["user_id", "segment"]], on="user_id", how="left")

# ────────────────────────── APRIORI GENRE RULES ────────────────────────────
@st.cache_data(show_spinner=True)
def genre_rules(data: pd.DataFrame):
    if "genre" not in data:
        return pd.DataFrame()
    basket = (data[["booking_id", "genre"]].dropna().drop_duplicates()
              .assign(v=1).pivot_table(index="booking_id", columns="genre", values="v", fill_value=0))
    freq = apriori(basket, min_support=0.02, use_colnames=True)
    return (association_rules(freq, metric="confidence", min_threshold=0.30)
            .sort_values("lift", ascending=False)) if not freq.empty else pd.DataFrame()

rules = genre_rules(df)

# ────────────────────────── KPI CARDS ───────────────────────────────────────
col1, col2, col3 = st.columns(3)
for col, val, label in zip((col1, col2, col3),
                           (f"{df.tickets.sum():,}", f"{(1 - df.tickets.sum() / df.cap.sum()) * 100:,.1f}%", f"{sil:.2f}"),
                           ("Tickets Sold", "Vacancy", "Silhouette")):
    with col:
        st.markdown(f"<div class='kpi'><p class='v'>{val}</p><p class='l'>{label}</p></div>", unsafe_allow_html=True)

# ────────────────────────── TABS ────────────────────────────────────────────
tab_cap, tab_seg, tab_ml = st.tabs(["📈 Capacity", "🧩 Segments", "🤖 Predict"])

# === Capacity Tab ===
with tab_cap:
    st.subheader("Hourly Seat Occupancy")
    if df["show_time"].notna().any():
        hourly = (df.assign(hr=df.show_time.dt.hour)
                    .groupby("hr").agg(occ=("tickets", "sum"), cap=("cap", "sum"))
                    .assign(pct=lambda x: x.occ / x.cap).reset_index())
        st.plotly_chart(px.area(hourly, x="hr", y="pct", labels={"pct": "Occupancy %"}, template="plotly_white"),
                        use_container_width=True)
    else:
        st.info("Dataset lacks `show_time`; occupancy cannot be plotted.")

# === Segments Tab ===
with tab_seg:
    st.subheader("Segment Distribution")
    seg_counts = rfm.segment.value_counts().reset_index(name="users").rename(columns={"index": "segment"})
    if not seg_counts.empty:
        st.plotly_chart(
            px.bar(seg_counts, x="segment", y="users", text="users", color="segment", template="simple_white", height=340),
            use_container_width=True)
    else:
        st.info("No segment data available.")

    st.markdown("---")
    st.subheader("Top Genre Association Rules")
    if rules.empty:
        st.info("Insufficient genre variety to generate rules.")
    else:
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]]
                     .head(10)
                     .style.format({"support": "{:.1%}", "confidence": "{:.1%}", "lift": "{:.2f}"}),
                     use_container_width=True)

# === Predict Tab ===
with tab_ml:
    st.subheader("Train & Test Accuracy")
    model_df = rfm.copy()
    model_df["target"] = (model_df.segment == "High Value").astype(int)
    X = model_df[["rec", "freq", "mon"]]
    y = model_df["target"]

    if y.nunique() < 2:
        st.warning("Need both classes to train models.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
        MODELS = {
            "K-NN": KNeighborsClassifier(5),
            "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
            "Random Forest": RandomForestClassifier(200, random_state=42),
            "Grad Boost": GradientBoostingClassifier(random_state=42),
        }
        results, fitted = [], {}
        for name, model in MODELS.items():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            results.append({"Model": name, "Accuracy": f"{acc:.2%}"})
            fitted[name] = model
        st.table(pd.DataFrame(results))

        st.markdown("---")
        st.subheader("🎯 Interactive High‑Value Prediction")
        with st.form("predict"):
            c1, c2, c3 = st.columns(3)
            rec_in = c1.number_input("Recency (days)", min_value=0, value=30, step=5)
            freq_in = c2.number_input("Total Tickets", min_value=0, value=5, step=1)
            mon_in = c3.number_input("Total Spend (₹)", min_value=0.0, value=1500.0, step=100.0, format="%.0f")
            chosen_model = st.selectbox("Model", list(MODELS.keys()), index=2)
            submit = st.form_submit_button("Predict")
        if submit:
            prob = fitted[chosen_model].predict_proba([[rec_in, freq_in, mon_in]])[0][1]
            pred = "High Value" if prob >= 0.5 else "Other"
            st.success(f"**Prediction:** {pred}  •  **Probability:** {prob:.1%}")

# Footer
st.markdown("<center style='font-size:.75rem;color:#888'>© 2025 • Glow‑UI + safe plotting build ✅</center>", unsafe_allow_html=True)
