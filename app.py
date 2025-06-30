# app.py — Cinema Performance Command-Center ✨ 2025
# -----------------------------------------------------------------------------
# • Glowing KPI cards and neumorphic panels
# • K-Means customer segments  • Genre-level association-rule mining
# • 4 predictive classifiers (K-NN, DT, RF, G-Boost) with live accuracy table
# -----------------------------------------------------------------------------
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

# ──────────────────────────── PAGE SET-UP ─────────────────────────────────────
st.set_page_config(page_title="Cinema Command-Center", page_icon="🎬", layout="wide")

# ——— Custom CSS (soft shadows + glow on hover) ——————————————
st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
/* sidebar */
[data-testid="stSidebar"] > div:first-child{background:linear-gradient(180deg,#202632 0%,#1e2127 100%);}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] label{color:#f4f4f4;}
/* KPI cards */
.kpi{background:#ffffff;border-radius:18px;padding:18px;box-shadow:0 8px 18px rgba(0,0,0,.08);transition:all .3s}
.kpi:hover{transform:translateY(-4px);box-shadow:0 12px 24px rgba(0,0,0,.12);}
.kpi .v{font-size:2.1rem;font-weight:800;color:#E50914;margin:0;text-align:center;}
.kpi .l{text-align:center;font-size:.8rem;color:#666;margin-top:-6px;}
/* neumorphic panels */
.block{background:#f6f7fa;border-radius:18px;padding:22px;box-shadow:8px 8px 14px #dfe0e4,-8px -8px 14px #ffffff;}
div.js-plotly-plot .modebar{display:none!important;}
/* tab icons alignment */
button[data-baseweb="tab"]{padding:8px 16px;font-weight:600}
</style>""", unsafe_allow_html=True)

# ──────────────────────────── COLUMN MAP ─────────────────────────────────────
RAW_MAP = {
    "tickets" : ["total_tickets"],
    "price"   : ["price_per_ticket"],
    "cap"     : ["available_seats"],
    "bdate"   : ["booking_date"],
    "sdate"   : ["show_date"],
    "stime"   : ["start_time"],
    "genre"   : ["genre"],
}
DEF_NUM = {"tickets":0,"price":0.,"cap":1}

def tidy(df: pd.DataFrame)->pd.DataFrame:
    # rename headers ➜ canonical
    low = {c.lower():c for c in df.columns}
    for canon,alts in RAW_MAP.items():
        for a in alts:
            if a.lower() in low:
                df=df.rename(columns={low[a.lower()]:canon});break
    # numeric sanitise
    for c,d in DEF_NUM.items():
        df[c] = pd.to_numeric(df.get(c,d),errors="coerce").fillna(d)
    # date parsing
    for dt in ["bdate","sdate","stime"]:
        if dt in df: df[dt]=pd.to_datetime(df[dt],errors="coerce")
    if {"sdate","stime"}<=set(df):
        df["show_time"]=df["sdate"]+(df["stime"]-df["stime"].dt.normalize())
    df["show_day"]=pd.to_datetime(df.get("show_time")).dt.normalize()
    df["sales"]=df["tickets"]*df["price"]
    return df

@st.cache_data(show_spinner=True)
def load(csv="BookMyShow_Combined_Clean_v2.csv"):
    if not Path(csv).is_file():
        st.error("📂 **CSV not found.** Upload the dataset and hit *Rerun*."); st.stop()
    return tidy(pd.read_csv(csv))

df=load()

# ──────────────────────────── SIDEBAR ─────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    if df["show_day"].notna().any():
        d0,d1=df["show_day"].min().date(),df["show_day"].max().date()
        d_from,d_to=st.date_input("Show window",(d0,d1))
        df=df[df["show_day"].between(pd.Timestamp(d_from),pd.Timestamp(d_to))]

# ──────────────────────────── K-MEANS SEGMENTS ───────────────────────────────
snap=(df["bdate"].max()+timedelta(days=1)) if "bdate" in df else pd.Timestamp.today()
rfm=(df.groupby("user_id")
       .agg(rec=("bdate",lambda s:(snap-s.max()).days if s.notna().any() else 9999),
            freq=("tickets","sum"),
            mon =("sales","sum"))
       .reset_index())
if len(rfm):
    X=StandardScaler().fit_transform(rfm[["rec","freq","mon"]])
    km=KMeans(5,random_state=42,n_init="auto").fit(X); rfm["cluster"]=km.labels_
    sil=silhouette_score(X,km.labels_)
else: sil=0; rfm["cluster"]=0
SEG_LABEL={0:"New",1:"One-Timer",2:"Low Value",3:"Mid Value",4:"High Value"}
rfm["segment"]=rfm["cluster"].map(SEG_LABEL)
df=df.merge(rfm[["user_id","segment"]],on="user_id",how="left")

# ──────────────────────────── ASSOCIATION RULES ─────────────────────────────
@st.cache_data(show_spinner=True)
def genre_rules(tdf:pd.DataFrame):
    if "genre" not in tdf: return pd.DataFrame()
    basket=(tdf[["booking_id","genre"]].dropna().drop_duplicates()
              .assign(v=1).pivot_table(index="booking_id",columns="genre",values="v",fill_value=0))
    freq=apriori(basket,min_support=0.02,use_colnames=True)
    if freq.empty: return pd.DataFrame()
    return association_rules(freq,metric="confidence",min_threshold=0.3)\
            .sort_values("lift",ascending=False)

rules=genre_rules(df)

# ──────────────────────────── KPI BAR ────────────────────────────────────────
def card(v,l): st.markdown(f'<div class="kpi"><p class="v">{v}</p><p class="l">{l}</p></div>',unsafe_allow_html=True)
c1,c2,c3=st.columns(3)
with c1: card(f"{df.tickets.sum():,}","Tickets Sold")
with c2: card(f"{(1-df.tickets.sum()/df.cap.sum())*100:,.1f}%","Vacancy")
with c3: card(f"{sil:.2f}","Silhouette")

# ──────────────────────────── TABS ───────────────────────────────────────────
tab_cap,tab_seg,tab_ml=st.tabs(
    ["📈 Capacity","🧩 Segments","🤖 Predict"]
)

with tab_cap:
    st.markdown("### 🕒 Hourly Seat Occupancy")
    if df["show_time"].notna().any():
        hourly=(df.assign(hr=df.show_time.dt.hour)
                  .groupby("hr").agg(occ=("tickets","sum"),cap=("cap","sum"))
                  .assign(pct=lambda x:x.occ/x.cap).reset_index())
        fig=px.area(hourly,x="hr",y="pct",labels={"pct":"Occupancy %"},template="plotly_white")
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig,use_container_width=True)
    else: st.warning("Dataset has no `show_time` information.")

with tab_seg:
    st.markdown("### 👥 Segment Distribution")
    seg_ct=rfm.segment.value_counts().rename_axis("segment").reset_index(name="users")
    st.plotly_chart(px.bar(seg_ct,x="segment",y="users",text="users",color="segment",
                           template="simple_white",height=350),use_container_width=True)
    st.subheader("🎬 Top Genre → Genre Rules")
    if rules.empty: st.info("Not enough genre diversity for rules.")
    else:
        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]]
                     .head(12).style.format({"support":"{:.1%}","confidence":"{:.1%}","lift":"{:.2f}"}),
                     use_container_width=True)

with tab_ml:
    st.markdown("### 🏅 Predict High-Value Customers")
    mdl=rfm.copy(); mdl["target"]=(mdl.segment=="High Value").astype(int)
    X=mdl[["rec","freq","mon"]]; y=mdl["target"]
    if y.nunique()<2: st.info("Need both classes to train."); st.stop()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
    MODELS={
        "K-NN":KNeighborsClassifier(5),
        "Decision Tree":DecisionTreeClassifier(max_depth=4,random_state=42),
        "Random Forest":RandomForestClassifier(200,random_state=42),
        "Grad Boost":GradientBoostingClassifier(random_state=42),
    }
    res=[]
    for n,m in MODELS.items():
        m.fit(Xtr,ytr); res.append({"Model":n,"Accuracy":f"{accuracy_score(yte,m.predict(Xte)):.2%}"})
    st.table(pd.DataFrame(res))

st.markdown("<center style='font-size:.75rem;color:#888'>© 2025 • Glow-UI build — data, segments, rules & ML at a glance 🚀</center>",
            unsafe_allow_html=True)
