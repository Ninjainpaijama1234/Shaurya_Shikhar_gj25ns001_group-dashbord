# app.py — Cinema Performance Command-Center ✨ 2025 (interactive edition)
# ==========================================================================
# • Glow-UI + neumorphic panels
# • K-Means segmentation  • Genre association-rules
# • Four classifiers (K-NN, DT, RF, GB) with LIVE accuracy table
# • **NEW**: interactive “Predict High-Value” form (recency, frequency, spend)
# --------------------------------------------------------------------------
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
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

# ─────────────────────────── PAGE CONFIG / THEME ────────────────────────────
st.set_page_config(page_title="Cinema Command-Center", page_icon="🎬", layout="wide")
st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#202632 0%,#1e2228 100%);}
.kpi{background:#fff;border-radius:18px;padding:18px;box-shadow:0 8px 18px rgba(0,0,0,.08);transition:.3s}
.kpi:hover{transform:translateY(-4px);box-shadow:0 12px 24px rgba(0,0,0,.12);}
.kpi .v{font-size:2rem;font-weight:800;color:#E50914;text-align:center;margin:0}
.kpi .l{text-align:center;font-size:.8rem;color:#666;margin-top:-6px}
div.js-plotly-plot .modebar{display:none!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── COLUMN MAP ─────────────────────────────────────
RAW_MAP = {
    "tickets": ["total_tickets"], "price": ["price_per_ticket"], "cap": ["available_seats"],
    "bdate": ["booking_date"], "sdate": ["show_date"], "stime": ["start_time"], "genre": ["genre"],
}
DEF_NUM = {"tickets": 0, "price": 0.0, "cap": 1}

def tidy(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}
    for canon, alts in RAW_MAP.items():
        for h in alts:
            if h.lower() in lower: df = df.rename(columns={lower[h.lower()]: canon}); break
    for c, d in DEF_NUM.items(): df[c] = pd.to_numeric(df.get(c, d), errors="coerce").fillna(d)
    for dt in ["bdate", "sdate", "stime"]: 
        if dt in df: df[dt] = pd.to_datetime(df[dt], errors="coerce")
    if {"sdate", "stime"} <= set(df): df["show_time"] = df["sdate"] + (df["stime"] - df["stime"].dt.normalize())
    df["show_day"] = pd.to_datetime(df.get("show_time")).dt.normalize()
    df["sales"] = df["tickets"] * df["price"]
    return df

@st.cache_data(show_spinner=True)
def load(csv="BookMyShow_Combined_Clean_v2.csv"):
    if not Path(csv).is_file(): st.error("🗂 CSV missing — upload & rerun."); st.stop()
    return tidy(pd.read_csv(csv))

df = load()

# ─────────────────────────── SIDEBAR FILTERS ────────────────────────────────
with st.sidebar:
    st.title("Filters")
    if df["show_day"].notna().any():
        d0, d1 = df["show_day"].min().date(), df["show_day"].max().date()
        r = st.date_input("Show window", (d0, d1))
        df = df[df["show_day"].between(pd.Timestamp(r[0]), pd.Timestamp(r[1]))]

# ─────────────────────────── K-MEANS SEGMENTS ───────────────────────────────
snap = (df["bdate"].max() + timedelta(days=1)) if "bdate" in df else pd.Timestamp.today()
rfm = (df.groupby("user_id")
         .agg(rec=("bdate", lambda s: (snap - s.max()).days if s.notna().any() else 9999),
              freq=("tickets", "sum"), mon=("sales", "sum"))
         .reset_index())
if len(rfm):
    Sc = StandardScaler().fit_transform(rfm[["rec", "freq", "mon"]])
    km = KMeans(5, random_state=42, n_init="auto").fit(Sc)
    rfm["cluster"] = km.labels_
    sil = silhouette_score(Sc, km.labels_)
else: sil = 0; rfm["cluster"] = 0
LABELS = {0: "New", 1: "One-Timer", 2: "Low Value", 3: "Mid Value", 4: "High Value"}
rfm["segment"] = rfm["cluster"].map(LABELS)
df = df.merge(rfm[["user_id", "segment"]], on="user_id", how="left")

# ─────────────────────────── GENRE RULES ────────────────────────────────────
@st.cache_data(show_spinner=True)
def genre_rules(data: pd.DataFrame):
    if "genre" not in data: return pd.DataFrame()
    basket = (data[["booking_id", "genre"]].dropna().drop_duplicates()
                .assign(v=1).pivot_table(index="booking_id", columns="genre", values="v", fill_value=0))
    freq = apriori(basket, min_support=0.02, use_colnames=True)
    return (association_rules(freq, metric="confidence", min_threshold=0.3)
            .sort_values("lift", ascending=False)) if not freq.empty else pd.DataFrame()

rules = genre_rules(df)

# ─────────────────────────── KPI CARDS ──────────────────────────────────────
def card(v, l): st.markdown(f'<div class="kpi"><p class="v">{v}</p><p class="l">{l}</p></div>', unsafe_allow_html=True)
k1, k2, k3 = st.columns(3)
with k1: card(f"{df.tickets.sum():,}", "Tickets Sold")
with k2: card(f"{(1 - df.tickets.sum() / df.cap.sum()) * 100:,.1f} %", "Vacancy")
with k3: card(f"{sil:.2f}", "Silhouette")

# ─────────────────────────── TABS ───────────────────────────────────────────
tab_cap, tab_seg, tab_ml = st.tabs(["📈 Capacity", "🧩 Segments", "🤖 Predict"])

# Capacity Tab
with tab_cap:
    st.markdown("#### Hourly Seat Occupancy")
    if df["show_time"].notna().any():
        h = (df.assign(hr=df.show_time.dt.hour)
               .groupby("hr").agg(occ=("tickets", "sum"), cap=("cap", "sum"))
               .assign(p=lambda x: x.occ / x.cap).reset_index())
        fig = px.area(h, x="hr", y="p", labels={"p": "Occupancy %"}, template="plotly_white")
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No `show_time` available in dataset.")

# Segments Tab
with tab_seg:
    st.markdown("#### Segment Distribution")
    sc = rfm.segment.value_counts().reset_index().rename(columns={"index": "segment", "segment": "users"})
    st.plotly_chart(px.bar(sc, x="segment", y="users", text="users", color="segment",
                           template="simple_white", height=340), use_container_width=True)
    st.divider()
    st.markdown("#### Top Genre → Genre Rules")
    if rules.empty: st.info("Not enough genre diversity for rules.")
    else:
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]]
                     .head(10)
                     .style.format({"support": "{:.1%}", "confidence": "{:.1%}", "lift": "{:.2f}"}),
                     use_container_width=True)

# Predict Tab
with tab_ml:
    st.markdown("#### Train-and-Test Accuracy")
    mdl_df = rfm.copy()
    mdl_df["target"] = (mdl_df.segment == "High Value").astype(int)
    X = mdl_df[["rec", "freq", "mon"]]; y = mdl_df["target"]
    if y.nunique() < 2: st.info("Need both classes to build models."); st.stop()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
    MODELS = {
        "K-NN": KNeighborsClassifier(5),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(200, random_state=42),
        "Grad Boost": GradientBoostingClassifier(random_state=42),
    }
    res, fitted = [], {}
    for n, m in MODELS.items():
        m.fit(Xtr, ytr); acc = accuracy_score(yte, m.predict(Xte))
        res.append({"Model": n, "Accuracy": f"{acc:.2%}"}); fitted[n] = m
    st.table(pd.DataFrame(res))

    st.divider()
    st.markdown("### 🎯 Interactive Prediction")
    with st.form("predict_form"):
        colA, colB, colC = st.columns(3)
        rec_in = colA.number_input("Recency (days)", min_value=0, value=30, step=5)
        freq_in = colB.number_input("Total Tickets", min_value=0, value=5, step=1)
        mon_in = colC.number_input("Total Spend (₹)", min_value=0.0, value=1500.0, step=100.0, format="%.0f")
        model_name = st.selectbox("Choose model", list(MODELS.keys()), index=2)
        btn = st.form_submit_button("Predict")
    if btn:
        prob = fitted[model_name].predict_proba([[rec_in, freq_in, mon_in]])[0][1]
        pred = "High Value" if prob >= 0.5 else "Other"
        st.success(f"**Segment prediction:** {pred}  \n**Probability (High-Value): {prob:.1%}**")

# Footer
st.markdown("<center style='font-size:.75rem;color:#888'>© 2025 • Glow-UI + interactive ML ✅</center>",
            unsafe_allow_html=True)
