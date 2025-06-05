# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

# --- Load Data ---
df = pd.read_csv("air.csv", skiprows=3)
df.columns = df.columns.str.strip()  # Clean column names

# --- Convert to datetime if there's a 'Date' column ---
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # Drop rows with invalid dates
    df.set_index('Date', inplace=True)

st.title("🌍 Air Quality Dashboard with Forecasting")


# --- Select numeric column to analyze ---
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in the dataset.")
    st.stop()

selected_col = st.selectbox("📊 Choose a variable to analyze & forecast", numeric_cols)

# --- Date filter (if datetime index exists) ---
if df.index.dtype == 'datetime64[ns]':
    date_range = st.date_input("📅 Select date range", [df.index.min(), df.index.max()])
    if isinstance(date_range, list) and len(date_range) == 2:
        df = df.loc[date_range[0]:date_range[1]]

# --- Plot selected column ---
st.subheader("📈 Historical Data")
st.line_chart(df[selected_col])

# --- Forecast Section ---
st.subheader("🔮 Predict Future Values")
future_days = st.slider("Select number of days to forecast", 7, 90, 30)

# Prepare data
df_filtered = df[[selected_col]].dropna().copy()

# Ensure datetime index
if not isinstance(df_filtered.index, pd.DatetimeIndex):
    df_filtered.index = pd.date_range(start="2023-01-01", periods=len(df_filtered), freq="D")

df_filtered["day_index"] = range(len(df_filtered))
X = df_filtered[["day_index"]]
y = df_filtered[selected_col]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future values
last_index = df_filtered["day_index"].iloc[-1]
future_index = np.arange(last_index + 1, last_index + 1 + future_days)
last_date = df_filtered.index[-1]
future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
future_preds = model.predict(future_index.reshape(-1, 1))

# Combine and display forecast
forecast_df = pd.DataFrame({selected_col: future_preds}, index=future_dates)
combined_df = pd.concat([df_filtered[[selected_col]], forecast_df])

st.line_chart(combined_df)

# Download button
csv = forecast_df.to_csv().encode('utf-8')
st.download_button("⬇️ Download Forecast CSV", csv, f"{selected_col}_forecast.csv", "text/csv")
