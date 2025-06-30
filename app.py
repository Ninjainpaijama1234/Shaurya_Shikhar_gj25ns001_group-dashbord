# app.py — Cinema Performance Command-Center ⭐️ 2025
# A polished Streamlit cockpit with:
# • K-Means customer segments  • Apriori association-rule mining
# • Four predictive ML models (K-NN, Decision-Tree, Random-Forest, Gradient-Boost)
# =============================================================================
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlxtend.frequent_patterns import apriori, association_rules

# ─────────────────────────── 0. THEME ────────────────────────────
st.set_page_config("🎬 Cinema Command-Center", "🎟️", "wide")
st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.sidebar .sidebar-content{background:#f7f8fa;}
.kpi{background:linear-gradient(135deg,#fff,#f1f3f6);border-radius:14px;padding:14px;box-shadow:0 4px 8px rgba(0,0,0,.04);}
.kpi .v{font-size:1.9rem;font-weight:800;color:#E50914;text-align:center;margin:0;}
.kpi .l{text-align:center;font-size:.8rem;color:#666;margin-top:-4px;}
.metric-box{display:flex;align-items:center;gap:.5rem}
div.js-plotly-plot .modebar{display:none !important;}
</style>""", unsafe_allow_html=True)

# ─────────────────────────── 1. COLUMN MAP ──────────────────────
MAP = {  # canonical : your header(s)
    "tickets":  ["total_tickets"],
    "price":    ["price_per_ticket"],
    "capacity": ["available_seats"],
    "bdate":    ["booking_date"],
    "sdate":    ["show_date"],
    "stime":    ["start_time"],
    "theatre":  ["theater_id"],
    "genre":    ["genre"],
}
DEF_NUM = {"tickets":0,"price":0.0,"capacity":1}

def prep_df(raw: pd.DataFrame) -> pd.DataFrame:
    # 1️⃣ rename
    lower = {c.lower(): c for c in raw.columns}
    for canon, alts in MAP.items():
        for h in alts:
            if h.lower() in lower: raw = raw.rename(columns={lower[h.lower()]: canon}); break
    # 2️⃣ numeric defaults
    for c,d in DEF_NUM.items(): raw[c] = pd.to_numeric(raw.get(c,d),errors="coerce").fillna(d)
    # 3️⃣ dates
    for dt in ["bdate","sdate","stime"]: 
        if dt in raw: raw[dt] = pd.to_datetime(raw[dt],errors="coerce")
    if {"sdate","stime"}<= set(raw):
        raw["show_time"] = raw["sdate"] + (raw["stime"]-raw["stime"].dt.normalize())
    raw["sales"] = raw["tickets"]*raw["price"]
    raw["show_day"] = pd.to_datetime(raw.get("show_time")).dt.normalize()
    return raw

@st.cache_data(show_spinner=True)
def load(csv="BookMyShow_Combined_Clean_v2.csv"):
    if not Path(csv).is_file():
        st.error(f"CSV **{csv}** missing. Upload and rerun."); st.stop()
    return prep_df(pd.read_csv(csv))

df = load()

# ──────────────── 2. SIDEBAR (date filter only) ─────────────────
with st.sidebar:
    st.title("🎛 Filters")
    if df["show_day"].notna().any():
        dmin,dmax = df["show_day"].min().date(), df["show_day"].max().date()
        s,e = st.date_input("Show Day", (dmin,dmax))
        df = df[df["show_day"].between(pd.Timestamp(s), pd.Timestamp(e))]

# ──────────────── 3. K-MEANS SEGMENTS (5 clusters) ──────────────
snap = (df["bdate"].max()+timedelta(days=1)) if "bdate" in df else pd.Timestamp.today()
rfm = (df.groupby("user_id")
         .agg(rec=("bdate",lambda s:(snap-s.max()).days if s.notna().any() else 9999),
              freq=("tickets","sum"), mon=("sales","sum"))
         .reset_index())
if len(rfm):
    X = StandardScaler().fit_transform(rfm[["rec","freq","mon"]])
    km = KMeans(5,random_state=42,n_init="auto").fit(X)
    rfm["cluster"] = km.labels_
    sil = silhouette_score(X, km.labels_)
else: sil = 0; rfm["cluster"]=[]

label_logic = {0:"Low Value",1:"Mid Value",2:"High Value",3:"New",4:"One-Timer"}
rfm["segment"] = rfm["cluster"].map(label_logic)
df = df.merge(rfm[["user_id","segment"]],on="user_id",how="left")

# ──────────────── 4. ASSOCIATION RULES (genre) ──────────────────
@st.cache_data(show_spinner=True)
def mine_rules(tdf: pd.DataFrame):
    if "genre" not in tdf: return pd.DataFrame()
    basket = (tdf[["booking_id","genre"]].dropna().drop_duplicates()
                .assign(v=1).pivot_table(index="booking_id", columns="genre", values="v", fill_value=0))
    freq = apriori(basket,min_support=0.02,use_colnames=True)
    if freq.empty: return pd.DataFrame()
    return association_rules(freq,metric="confidence",min_threshold=0.3)\
            .sort_values("lift",ascending=False)

rules = mine_rules(df)

# ──────────────── 5. KPI STRIP ──────────────────────────────────
def kpi(v,l): st.markdown(f'<div class="kpi"><p class="v">{v}</p><p class="l">{l}</p></div>',unsafe_allow_html=True)
a,b,c = st.columns(3)
with a:kpi(f"{df.tickets.sum():,}","Tickets Sold")
with b:kpi(f"{(1-df.tickets.sum()/df.capacity.sum())*100:,.1f}%","Vacancy")
with c:kpi(f"{sil:.2f}","Silhouette")

# ──────────────── 6. TABS LAYOUT ────────────────────────────────
tab_cap, tab_seg, tab_ml = st.tabs(["📊 Capacity","🧩 Segments","🤖 Predict"])

# Capacity
with tab_cap:
    st.subheader("Hourly Seat Occupancy")
    if df["show_time"].notna().any():
        hourly = (df.assign(h=df.show_time.dt.hour)
                    .groupby("h").agg(o=("tickets","sum"),c=("capacity","sum"))
                    .assign(occ=lambda x:x.o/x.c).reset_index())
        st.plotly_chart(px.bar(hourly,x="h",y="occ",labels={"occ":"Occupancy %"},height=380),use_container_width=True)
    else: st.info("No `show_time` data.")

# Segments
with tab_seg:
    st.subheader("Customer Segments (K-Means k=5)")
    seg_cnt = rfm.segment.value_counts().reset_index(); seg_cnt.columns=["segment","users"]
    st.plotly_chart(px.bar(seg_cnt,x="segment",y="users",text="users",height=350),use_container_width=True)
    st.dataframe(seg_cnt,use_container_width=True)
    st.divider(); st.subheader("Top Genre → Genre Rules")
    if rules.empty: st.info("Not enough genre variety.")
    else:
        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].head(10)
                     .style.format({"support":"{:.1%}","confidence":"{:.1%}","lift":"{:.2f}"}),
                     use_container_width=True)

# Predict
with tab_ml:
    st.subheader("Predict High-Value vs Other Customers")
    # target = 1 if segment == High Value
    mdl_df = rfm.copy()
    mdl_df["target"] = (mdl_df.segment=="High Value").astype(int)
    X = mdl_df[["rec","freq","mon"]]; y = mdl_df["target"]
    if y.nunique() < 2:
        st.info("Need both classes to train models."); st.stop()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.3,random_state=42,stratify=y)

    MODELS = {
        "K-NN (k=5)":KNeighborsClassifier(5),
        "Decision-Tree":DecisionTreeClassifier(max_depth=4,random_state=42),
        "Random-Forest":RandomForestClassifier(n_estimators=200,random_state=42),
        "Gradient-Boost":GradientBoostingClassifier(random_state=42),
    }
    results=[]
    for name,model in MODELS.items():
        model.fit(Xtr,ytr); acc=accuracy_score(yte,model.predict(Xte))
        results.append({"Model":name,"Accuracy":f"{acc:.2%}"})
    st.table(pd.DataFrame(results))
    st.caption("Predictive baseline: identify likely high-value customers for targeted retention.")

# Footer
st.markdown("<center style='font-size:.75rem;color:#888'>© 2025 • Enhanced dashboard — segments, rules, ML insights ⚡️</center>",unsafe_allow_html=True)
