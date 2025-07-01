# app.py - Streamlit Dashboard with Corrected LSTM using tensorflow.keras

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="🎬 BookMyShow Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("BookMyShow_Combined_Clean_v2.csv", parse_dates=["booking_date", "payment_date", "show_date"], dayfirst=True)
    df["Monetary"] = df["price_per_ticket"] * df["total_tickets"]
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("🔍 Filters")
genre_filter = st.sidebar.multiselect("Select Genre", options=df['genre'].unique())
language_filter = st.sidebar.multiselect("Select Language", options=df['language'].unique())
screen_filter = st.sidebar.multiselect("Select Screen ID", options=df['screen_id'].unique())

df_filtered = df.copy()
if genre_filter:
    df_filtered = df_filtered[df_filtered['genre'].isin(genre_filter)]
if language_filter:
    df_filtered = df_filtered[df_filtered['language'].isin(language_filter)]
if screen_filter:
    df_filtered = df_filtered[df_filtered['screen_id'].isin(screen_filter)]

# --- Tabs ---
tabs = st.tabs([
    "Overview", "RFM Segmentation", "Churn Prediction", "CLV Forecasting", "Next Purchase", "Sales Forecasting (LSTM)"
])

# --- Tab1: Overview ---
with tabs[0]:
    st.title("🎬 BookMyShow - Customer Analytics")
    st.markdown("Preview of latest bookings and KPIs")
    st.dataframe(df_filtered.sort_values(by="booking_date", ascending=False).head(100))
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", df_filtered['user_id'].nunique())
    col2.metric("Total Bookings", df_filtered['booking_id'].nunique())
    col3.metric("Total Revenue", f"₹{df_filtered['Monetary'].sum():,.0f}")

# --- Tab2: RFM Segmentation ---
with tabs[1]:
    st.header("🧮 RFM Segmentation")
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
    st.plotly_chart(px.pie(rfm, names="Segment", title="Customer Segments"))
    st.dataframe(rfm.head(20))

# --- Tab3: Churn Prediction ---
with tabs[2]:
    st.header("🚨 Churn Prediction")
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
        st.subheader(f"{name} Results")
        st.text(classification_report(y_test, y_pred))
        st.write("Accuracy:", accuracy_score(y_test, y_pred))

# --- Tab4: CLV Forecasting ---
with tabs[3]:
    st.header("📈 CLV Forecasting")
    years = st.slider("Years to Forecast", 1, 5, 3)
    clv_df = df_filtered.groupby("user_id").agg({"Frequency": "sum", "Monetary": "sum"}).reset_index()
    clv_df["Forecasted_CLV"] = clv_df["Frequency"] * clv_df["Monetary"] * years
    st.plotly_chart(px.histogram(clv_df, x="Forecasted_CLV", nbins=40, title=f"Forecasted CLV ({years} years)"))
    st.dataframe(clv_df.sort_values(by="Forecasted_CLV", ascending=False).head(30))

# --- Tab5: Next Purchase Prediction ---
with tabs[4]:
    st.header("📅 Next Purchase Prediction")
    recent_bookings = df_filtered.groupby("user_id")["booking_date"].max().reset_index()
    recent_bookings["Predicted_Next_Purchase"] = recent_bookings["booking_date"] + pd.to_timedelta(30, unit='d')
    st.dataframe(recent_bookings.sort_values(by="Predicted_Next_Purchase"))

# --- Tab6: Sales Forecasting with LSTM ---
with tabs[5]:
    st.header("📉 LSTM-Based Sales Forecast")
    data = df_filtered[['booking_date', 'total_tickets']].copy()
    data = data.groupby(data['booking_date'].dt.to_period('M')).sum().reset_index()
    data['booking_date'] = data['booking_date'].dt.to_timestamp()
    data.set_index('booking_date', inplace=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(12, len(scaled)):
        X.append(scaled[i-12:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=1, verbose=0)

    input_seq = scaled[-12:]
    preds = []
    for _ in range(12):
        pred = model.predict(input_seq.reshape(1, 12, 1), verbose=0)
        preds.append(pred[0][0])
        input_seq = np.append(input_seq[1:], [pred], axis=0)

    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    forecast_df = pd.DataFrame({"Month": future_dates, "Forecasted_Tickets": scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()})
    st.plotly_chart(px.line(forecast_df, x="Month", y="Forecasted_Tickets", title="12-Month Ticket Forecast"))
    st.dataframe(forecast_df)
