# app.py - Streamlit Dashboard with Theatre Utilization Clustering

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Page Configuration ---
st.set_page_config(page_title="🎬 BookMyShow Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("BookMyShow_Combined_Clean_v2.csv", parse_dates=["booking_date", "payment_date", "show_date"], dayfirst=True)
    df["Monetary"] = df["price_per_ticket"] * df["total_tickets"]
    df["occupancy"] = df["total_tickets"] / (df["total_tickets"] + df["available_seats"])
    df["hour"] = pd.to_datetime(df["start_time"]).dt.hour
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("🔍 Global Filters")
genre_filter = st.sidebar.multiselect("🎭 Genre", options=df['genre'].unique())
language_filter = st.sidebar.multiselect("🗣️ Language", options=df['language'].unique())
screen_filter = st.sidebar.multiselect("🖥️ Screen ID", options=df['screen_id'].unique())

df_filtered = df.copy()
if genre_filter:
    df_filtered = df_filtered[df_filtered['genre'].isin(genre_filter)]
if language_filter:
    df_filtered = df_filtered[df_filtered['language'].isin(language_filter)]
if screen_filter:
    df_filtered = df_filtered[df_filtered['screen_id'].isin(screen_filter)]

# --- Tabs ---
tabs = st.tabs([
    "🏠 Overview", "📊 RFM Segmentation", "🚨 Churn Prediction", "💰 CLV Forecasting",
    "🔁 Next Purchase", "📈 Sales Forecasting", "🎭 Theatre Utilization Clustering"
])

# --- Overview ---
with tabs[0]:
    st.title("📊 BookMyShow Customer Intelligence Dashboard")
    st.markdown("Gain actionable insights into user behavior, segmentation, churn risk, and revenue forecasting.")
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Users", df_filtered['user_id'].nunique())
    col2.metric("🎟️ Total Bookings", df_filtered['booking_id'].nunique())
    col3.metric("💵 Total Revenue", f"₹{df_filtered['Monetary'].sum():,.0f}")
    st.markdown("---")
    st.subheader("📅 Recent Transactions")
    st.dataframe(df_filtered.sort_values(by="booking_date", ascending=False).head(100), use_container_width=True)

# --- RFM Segmentation ---
with tabs[1]:
    st.header("📊 RFM Customer Segmentation")
    rfm = df_filtered.groupby("user_id").agg({"Recency": "min", "Frequency": "sum", "Monetary": "sum"}).reset_index()
    rfm["R_Score"] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm["F_Score"] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm["M_Score"] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

    def segment_user(row):
        if row["RFM_Score"] >= "445": return "High Value"
        elif row["RFM_Score"] >= "344": return "Mid Value"
        elif row["RFM_Score"] >= "233": return "Low Value"
        elif row["Frequency"] == 1: return "One Timer"
        else: return "New"

    rfm["Segment"] = rfm.apply(segment_user, axis=1)
    fig_rfm = px.pie(rfm, names="Segment", title="RFM Segments", hole=0.4)
    st.plotly_chart(fig_rfm, use_container_width=True)
    st.dataframe(rfm.head(20))

# --- Churn Prediction ---
with tabs[2]:
    st.header("🚨 Customer Churn Prediction")
    churn_df = df_filtered.groupby("user_id").agg({"Recency": "min", "Frequency": "sum", "Monetary": "sum"}).reset_index()
    churn_df["churn"] = churn_df["Recency"].apply(lambda x: 1 if x > 300 else 0)
    X = churn_df[["Recency", "Frequency", "Monetary"]]
    y = churn_df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    models = {
        "K-NN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boost": GradientBoostingClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader(f"Model: {name}")
        st.text(classification_report(y_test, y_pred))
        st.write("Accuracy Score:", accuracy_score(y_test, y_pred))

# --- CLV Forecasting ---
with tabs[3]:
    st.header("💰 Customer Lifetime Value Forecasting")
    years = st.slider("Projection Horizon (Years)", 1, 5, 3)
    clv_df = df_filtered.groupby("user_id").agg({"Frequency": "sum", "Monetary": "sum"}).reset_index()
    clv_df["Forecasted_CLV"] = clv_df["Frequency"] * clv_df["Monetary"] * years
    fig_clv = px.histogram(clv_df, x="Forecasted_CLV", nbins=40, title=f"Projected CLV for {years} Years")
    st.plotly_chart(fig_clv, use_container_width=True)
    st.dataframe(clv_df.sort_values(by="Forecasted_CLV", ascending=False).head(30))

# --- Next Purchase ---
with tabs[4]:
    st.header("🔁 Next Purchase Prediction")
    recent_bookings = df_filtered.groupby("user_id")["booking_date"].max().reset_index()
    recent_bookings["Predicted_Next_Purchase"] = recent_bookings["booking_date"] + pd.to_timedelta(30, unit='d')
    st.dataframe(recent_bookings.sort_values(by="Predicted_Next_Purchase"))

# --- Sales Forecasting ---
with tabs[5]:
    st.header("📈 Monthly Sales Trend Forecasting")
    df_filtered["Month"] = df_filtered["booking_date"].dt.to_period("M").dt.to_timestamp()
    monthly_sales = df_filtered.groupby("Month").agg({"total_tickets": "sum", "Monetary": "sum"}).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_sales["Month"], y=monthly_sales["total_tickets"], name="Tickets Sold", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=monthly_sales["Month"], y=monthly_sales["Monetary"], name="Revenue", mode="lines+markers"))
    fig.update_layout(title="📅 Monthly Tickets & Revenue", xaxis_title="Month", yaxis_title="Value", legend_title="Metric")

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(monthly_sales.tail(12))

# --- 🎭 Theatre Utilization Clustering ---
with tabs[6]:
    st.header("🎭 Theatre Utilization Clustering")
    usage_df = df_filtered.groupby(["theater_id", "hour"]).agg({
        "total_tickets": "sum",
        "available_seats": "sum"
    }).reset_index()
    usage_df["occupancy"] = usage_df["total_tickets"] / (usage_df["total_tickets"] + usage_df["available_seats"] + 1e-5)

    X = usage_df[["occupancy"]]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    usage_df["cluster"] = kmeans.labels_

    fig = px.scatter(usage_df, x="hour", y="occupancy", color="cluster", hover_data=["theater_id"], title="Usage Clustering by Time & Theatre")
    st.plotly_chart(fig, use_container_width=True)

    heatmap_data = usage_df.pivot_table(index="theater_id", columns="hour", values="occupancy")
    fig_heat = px.imshow(heatmap_data, labels=dict(x="Hour", y="Theatre", color="Occupancy"), title="🔥 Heatmap of Theatre Usage")
    st.plotly_chart(fig_heat, use_container_width=True)
    st.dataframe(usage_df.sort_values(by="occupancy", ascending=True).head(10))
