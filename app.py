
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="BookMyShow Analytics Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("BookMyShow_Combined_Clean_v2.csv", parse_dates=["booking_date", "payment_date", "show_date"])
    df["price_per_ticket"] = pd.to_numeric(df["price_per_ticket"], errors='coerce')
    df["Monetary"] = df["price_per_ticket"] * df["total_tickets"]
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
genre_filter = st.sidebar.multiselect("Genre", df['genre'].unique())
language_filter = st.sidebar.multiselect("Language", df['language'].unique())
screen_filter = st.sidebar.multiselect("Screen", df['screen_id'].unique())

filtered_df = df.copy()
if genre_filter:
    filtered_df = filtered_df[filtered_df['genre'].isin(genre_filter)]
if language_filter:
    filtered_df = filtered_df[filtered_df['language'].isin(language_filter)]
if screen_filter:
    filtered_df = filtered_df[filtered_df['screen_id'].isin(screen_filter)]

# --- Tabs Layout ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🧮 RFM Segmentation",
    "🚨 Churn Prediction",
    "📈 Customer Value Forecasting",
    "📅 Next Purchase Prediction",
    "📉 Sales Forecasting"
])

# --- Tab 1: Overview ---
with tab1:
    st.title("🎬 BookMyShow Analytics Dashboard")
    st.markdown("Comprehensive Customer Intelligence Platform for User Segmentation, Retention Forecasting, and Revenue Optimization.")
    st.dataframe(filtered_df.head(100), use_container_width=True)

# --- Tab 2: RFM Segmentation ---
with tab2:
    st.header("🧮 RFM Segmentation")
    rfm = filtered_df.groupby("user_id").agg({
        "Recency": "min",
        "Frequency": "sum",
        "Monetary": "sum"
    }).reset_index()

    # Scoring
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5])
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

    # Segment customers
    def segment_customer(row):
        score = row["RFM_Score"]
        if score >= '445': return "High Value"
        elif score >= '344': return "Mid Value"
        elif score >= '233': return "Low Value"
        elif row["Frequency"] == 1: return "One Timer"
        else: return "New"

    rfm["Segment"] = rfm.apply(segment_customer, axis=1)
    st.plotly_chart(px.pie(rfm, names="Segment", title="Customer Segments"), use_container_width=True)
    st.dataframe(rfm.head(), use_container_width=True)

# --- Tab 3: Churn Prediction ---
with tab3:
    st.header("🚨 Churn Prediction")
    df_cp = filtered_df.groupby("user_id").agg({
        "Recency": "min",
        "Frequency": "sum",
        "Monetary": "sum"
    }).reset_index()

    # Create churn label
    df_cp["churn"] = df_cp["Recency"].apply(lambda x: 1 if x > 365 else 0)
    X = df_cp[["Recency", "Frequency", "Monetary"]]
    y = df_cp["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "K-NN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boost": GradientBoostingClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        st.subheader(name)
        st.text(classification_report(y_test, pred))

# --- Tab 4: Customer Value Forecasting ---
with tab4:
    st.header("📈 CLV Forecasting (1-5 Years)")
    year_slider = st.slider("Select Forecast Horizon (Years)", 1, 5, 3)
    df_cf = df.groupby("user_id").agg({
        "Frequency": "sum",
        "Monetary": "sum"
    }).reset_index()
    df_cf["Forecasted_CLV"] = df_cf["Monetary"] * df_cf["Frequency"] * year_slider
    st.plotly_chart(px.histogram(df_cf, x="Forecasted_CLV", nbins=50, title="Forecasted Customer Value"), use_container_width=True)
    st.dataframe(df_cf[["user_id", "Forecasted_CLV"]].sort_values(by="Forecasted_CLV", ascending=False), use_container_width=True)

# --- Tab 5: Next Purchase Prediction ---
with tab5:
    st.header("📅 Next Purchase Prediction")
    df_np = df.sort_values(by=["user_id", "booking_date"])
    next_purchase = df_np.groupby("user_id")["booking_date"].max().reset_index()
    next_purchase["next_predicted_date"] = next_purchase["booking_date"] + pd.to_timedelta(30, unit='d')
    st.dataframe(next_purchase.sort_values(by="next_predicted_date"), use_container_width=True)

# --- Tab 6: Sales Forecasting ---
with tab6:
    st.header("📉 LSTM Sales Forecasting")
    df['booking_date'] = pd.to_datetime(df['booking_date'])
    sales_data = df.groupby(df['booking_date'].dt.to_period('M')).agg({
        "total_tickets": "sum"
    }).reset_index()
    sales_data['booking_date'] = sales_data['booking_date'].dt.to_timestamp()
    sales_data.set_index("booking_date", inplace=True)

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(sales_data)

    # Prepare sequence data
    sequence_length = 3
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=1, verbose=0)

    future_preds = []
    last_seq = scaled_data[-sequence_length:]
    for _ in range(12):
        pred = model.predict(last_seq.reshape(1, sequence_length, 1), verbose=0)
        future_preds.append(pred[0][0])
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    future_dates = pd.date_range(start=sales_data.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='MS')
    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Forecasted Sales": scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    })

    st.plotly_chart(px.line(forecast_df, x="Month", y="Forecasted Sales", title="12-Month Sales Forecast"), use_container_width=True)
    st.dataframe(forecast_df, use_container_width=True)
