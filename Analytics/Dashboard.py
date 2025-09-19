import pandas as pd
import plotly.express as px
import streamlit as st
import statsmodels.api as sm
import numpy as np

# Load dataset
df = pd.read_csv("../Dataset/car_sales_data.csv")

# st.set_page_config(page_title="Car Sales Dashboard", layout="wide")
st.title("🚗 Car Sales Dashboard")

st.markdown("Explore the car sales dataset with interactive charts and regression insights")


st.subheader("📊 Dataset Information")
st.write(f"Number of Rows: {df.shape[0]}")
st.write(f"Number of Columns: {df.shape[1]}")
st.write("Columns:", list(df.columns))

st.subheader("📝 Dataset Description")
st.dataframe(df.describe())

# Sidebar filters
manufacturer = st.sidebar.selectbox("Select Manufacturer", options=["All"] + list(df["Manufacturer"].unique()))
fuel_type = st.sidebar.selectbox("Select Fuel Type", options=["All"] + list(df["Fuel type"].unique()))

filtered_df = df.copy()
if manufacturer != "All":
    filtered_df = filtered_df[filtered_df["Manufacturer"] == manufacturer]
if fuel_type != "All":
    filtered_df = filtered_df[filtered_df["Fuel type"] == fuel_type]

# Show dataset preview "نعرضها ؟"
#st.subheader("📊 Data Preview")
#st.dataframe(filtered_df.head())

# visual 1 Price Distribution
fig1 = px.histogram(filtered_df, x="Price", nbins=50, title="Price Distribution")
st.plotly_chart(fig1)

# visual 2 Mileage vs Price
fig2 = px.scatter(filtered_df, x="Mileage", y="Price", color="Fuel type", title="Mileage vs Price")
st.plotly_chart(fig2)

# visual 3: Avg Price by Manufacturer
fig3 = px.bar(filtered_df.groupby("Manufacturer")["Price"].mean().reset_index(),
              x="Manufacturer", y="Price", title="Average Price by Manufacturer")
st.plotly_chart(fig3)