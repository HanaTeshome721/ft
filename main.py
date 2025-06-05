# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("air.csv", skiprows=3)

# Title
st.title("Air Quality Dashboard")

# Filter
column = st.selectbox("Choose a column to analyze", df.columns)

# Line chart
st.line_chart(df[column])

# Stats
st.write("Summary Statistics")
st.write(df[column].describe())
