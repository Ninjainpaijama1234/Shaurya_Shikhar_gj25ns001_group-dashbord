# app.py
import streamlit as st
import pandas as pd
from modules.segmentation import segmentation_tab
from modules.churn import churn_prediction_tab
from modules.forecasting import value_forecasting_tab
from modules.next_purchase import next_purchase_tab
from modules.sales import sales_forecasting_tab

# --- Page Config ---
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("BookMyShow_Combined_Clean_v2.csv", parse_dates=["booking_date", "payment_date", "show_date"])
    df["price_per_ticket"] = pd.to_numeric(df["price_per_ticket"], errors='coerce')
    df["Monetary"] = df["price_per_ticket"] * df["total_tickets"]
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("🔎 Filters")
genre_filter = st.sidebar.multiselect("Select Genre", df['genre'].unique())
language_filter = st.sidebar.multiselect("Select Language", df['language'].unique())
screen_filter = st.sidebar.multiselect("Select Screen", df['screen_id'].unique())

filtered_df = df.copy()
if genre_filter:
    filtered_df = filtered_df[filtered_df['genre'].isin(genre_filter)]
if language_filter:
    filtered_df = filtered_df[filtered_df['language'].isin(language_filter)]
if screen_filter:
    filtered_df = filtered_df[filtered_df['screen_id'].isin(screen_filter)]

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Segmentation & CLV",
    "🚨 Churn Prediction",
    "💰 Value Forecasting",
    "📅 Sales Forecasting"
])

# Overview Tab
with tab1:
    st.title("🎬 BookMyShow - Customer Analytics")
    st.markdown("""
    This dashboard presents an end-to-end customer intelligence platform:
    - RFM-based customer segmentation
    - Churn prediction models
    - CLV forecasting
    - Next purchase prediction
    - LSTM-based revenue forecasts
    """)
    st.dataframe(filtered_df.head(), use_container_width=True)

# Segmentation & CLV
with tab2:
    segmentation_tab(filtered_df)

# Churn Prediction
with tab3:
    churn_prediction_tab(filtered_df)

# Customer Value Forecasting
with tab4:
    value_forecasting_tab(filtered_df)

# Sales Forecasting (LSTM)
with tab5:
    sales_forecasting_tab(filtered_df)
