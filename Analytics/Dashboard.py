# Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import io
import base64
import os
import plotly.express as px
import plotly.graph_objects as go
import requests

# Set page configuration
st.set_page_config(page_title="Car Sales Dashboard", layout="wide")

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #ff7f0e; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem;}
    .metric-card {background-color: #f9f9f9; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .info-text {font-size: 1rem; color: #333;}
    .stButton>button {background-color: #1f77b4; color: white;}
    .dataset-description {background-color: #black; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;}
    .features {border: solid 1px; border-radius:8px; border-color: #ff7f0e; text-align:center; font-size: 12px; color: #ff7f0e; padding: 5px; margin-bottom: 25px;}
    [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 350px;
            font-size: 3px;
        }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🚗 Car Sales Dashboard</h1>', unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        # Load your dataset .read_csv("../Dataset/winners_f1_cleaned.csv")
        df = pd.read_csv("../Dataset/processed_car_sales_data_cleaning.csv")
        
        return df
    except:
        st.error("Dataset not found. Please place 'Dataset\car_sales_data.csv' in the same folder.")
        return pd.DataFrame()
        
# Load data
data = load_data()

# Sidebar for dataset describtion and filters and information
with st.sidebar:
    st.image("car.png", width=250)
    with st.sidebar:
        filters_tab, about_tab = st.tabs(["Filters", "About",])

        with filters_tab:  
            # Manufacturer filter
            manufacturers = st.multiselect(
            "Select Manufacturers",
            options=data['Manufacturer'].unique(),
            default=data['Manufacturer'].unique()
            )
    
            # Fuel type filter
            fuel_types = st.multiselect(
            "Select Fuel Types",
            options=data['Fuel type'].unique(),
            default=data['Fuel type'].unique()
            )
    
            # Mileage range filter
            min_mileage, max_mileage = st.slider(
            "Select Mileage Range",
            min_value=int(data['Mileage'].min()),
            max_value=int(data['Mileage'].max()),
            value=(int(data['Mileage'].min()), int(data['Mileage'].max()))
            )
    
            # Year range filter
            min_year, max_year = st.slider(
            "Select Year Range",
            min_value=int(data['Year of manufacture'].min()),
            max_value=int(data['Year of manufacture'].max()),
            value=(int(data['Year of manufacture'].min()), int(data['Year of manufacture'].max()))
            )
    
            # Price range filter
            min_price, max_price = st.slider(
            "Select Price Range ($)",
            min_value=int(data['Price'].min()),
            max_value=int(data['Price'].max()),
            value=(int(data['Price'].min()), int(data['Price'].max()))
            )

            # Filter data based on selections
            filtered_data = data[
                (data['Manufacturer'].isin(manufacturers)) &
                (data['Fuel type'].isin(fuel_types)) &
                (data['Mileage'] >= min_mileage) &
                (data['Mileage'] <= max_mileage) &
                (data['Year of manufacture'] >= min_year) &
                (data['Year of manufacture'] <= max_year) &
                (data['Price'] >= min_price) &
                (data['Price'] <= max_price)
            ]

        with about_tab:
            st.markdown("""
                <div style="font-size:13px; line-height:1.6; text-align:justify;">
        <h3>About the Dataset</h3>
        <p>This dashboard analyzes a comprehensive car sales dataset containing information about various vehicles 
        including their specifications, pricing, and sales information. The dataset includes both numerical and 
        categorical attributes that help in understanding the car market trends.</p>

        ### Analytics Objectives
        - **Pricing Trends:** How prices vary by brand, model, year, and features.  
        - **Market Preferences:** Popular car categories, fuel types, and engine sizes.  
        - **Sales Performance:** Top-performing manufacturers and models.  
        - **Feature Correlations:** Attribute relationships and their impact on price.  
        - **Outlier Detection:** Spotting unusual patterns or anomalies.  

        ### Key Metrics
        - Average prices by manufacturer and fuel type.  
        - Car distribution by year of manufacture.  
        - Mileage trends and their effect on price.  
        - Engine size preferences across segments.

        ### Data Source 
        Kaggle - [Car Sales Data](https://www.kaggle.com/datasets/minahilfatima12328/car-sales-info/data")
                </div>
            """, unsafe_allow_html=True)    
    
# main sections
data_overview, descriptive_statistics, visualization, prediction = st.tabs(["Data Overview", "Descriptive Statistics", "Visualizations", "Price Prediction"])

# data overview section
with data_overview:
    row_num, col_num, cat_num, num_num = st.columns(4)
    with row_num:
        st.metric("Records Number", filtered_data.shape[0])
    with col_num:
        st.metric("Features Number", filtered_data.shape[1])
    with cat_num:
        st.metric("Categorical Data", (filtered_data.dtypes == 'object').sum())
    with num_num:
        st.metric("Numerical Data", sum(np.issubdtype(dt, np.number) for dt in filtered_data.dtypes))

    st.write("Features")
    cols = st.columns(len(filtered_data.columns))

    for col, name in zip(cols, filtered_data.columns):
        with col:
            st.markdown(
            f"""
            <div class="features">
                {name}
            </div>
            """,
            unsafe_allow_html=True
        )

    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
    with insight_col1:
        avg_mileage = filtered_data['Mileage'].mean() / 1000
        st.metric("Average Mileage", f"{avg_mileage:,.2f}k mi")
    with insight_col2:
        most_common_fuel = filtered_data['Fuel type'].mode()[0] if len(filtered_data) > 0 else "N/A"
        st.metric("Most Common Fuel", most_common_fuel)
    with insight_col3:
        most_common_manufacturer = filtered_data['Manufacturer'].mode()[0] if len(filtered_data) > 0 else "N/A"
        st.metric("Most Common Manufacturer", most_common_manufacturer)
    with insight_col4:
        if len(filtered_data) > 0:
            newest_car = filtered_data['Year of manufacture'].max()
            oldest_car = filtered_data['Year of manufacture'].min()
            st.metric("Year Range", f"{oldest_car} - {newest_car}")
        else:
            st.metric("Year Range", "N/A")

    if st.checkbox("📊 View Data"):
        st.write(data)


# Descriptive statistics section
with descriptive_statistics:
    st.markdown('<p class="sub-header">Data Description</p>', unsafe_allow_html=True)
    
    if len(filtered_data) > 0:
        numerical_cols = filtered_data.select_dtypes(include=np.number).columns
        numerical = filtered_data[numerical_cols]
        
        # Calculate statistics
        stats_df = numerical.describe().T
        stats_df['range'] = stats_df['max'] - stats_df['min']
        st.dataframe(stats_df)
    else:
        st.warning("No data available for analysis after filtering")

    st.markdown('<p class="sub-header">Correlation Heatmap</p>', unsafe_allow_html=True)
    corr = numerical.corr()
    fig = px.imshow(
        corr,
        text_auto=True,        # show correlation values on heatmap
        aspect="auto",         # keeps cells square
        color_continuous_scale="OrRd", # red-blue reversed, like coolwarm
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Outlier Analysis
    st.markdown('<p class="sub-header">Outlier Analysis</p>', unsafe_allow_html=True)
    
    if len(filtered_data) > 0:
        method_col, column_col = st.columns(2)
        with method_col:
            outlier_method = st.selectbox("**Outlier Detection Method**", ["IQR Method", "Z-Score Method"])
        with column_col:
            selected_col = st.selectbox("**Feature of Outlier Analysis**", numerical_cols)
        
        st.markdown("### Outliers Graph")  # Heading before the graph

        if outlier_method == "IQR Method":
            Q1 = filtered_data[selected_col].quantile(0.25)
            Q3 = filtered_data[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = filtered_data[(filtered_data[selected_col] < lower_bound) | 
                                     (filtered_data[selected_col] > upper_bound)]
            
            iqr_col, low_col, up_col = st.columns(3)
            iqr_col.info(f"**IQR:** {IQR:.2f}")
            low_col.info(f"**Lower Bound:** {lower_bound:.2f}")
            up_col.info(f"**Upper Bound:** {upper_bound:.2f}")
            st.success(f"**Number of Outliers:** {len(outliers)}")
            
            if len(outliers) > 0:
                if st.checkbox("🔎 View Detected Outliers"): 
                    st.dataframe(outliers[['Manufacturer', 'Model', 'Year of manufacture', selected_col]])

            # Plotly Boxplot
            fig = px.box(
                filtered_data, 
                y=selected_col,
                points="all",  # shows all points
                title=f"Boxplot of {selected_col} with Outliers"
            )
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

            # Insights & Recommendations
            st.markdown("##### 📊 Insights & Recommendations")
            st.write(f"- {len(outliers)} outliers detected in {selected_col}.")
            if len(outliers) > 0:
                st.info("- Outliers may distort statistical analyses; consider removing or transforming them for modeling.")
            else:
                st.success("- No significant outliers detected; data is relatively clean.")

        else:  # Z-Score Method
            z_scores = np.abs((filtered_data[selected_col] - filtered_data[selected_col].mean()) / 
                               filtered_data[selected_col].std())
            outliers = filtered_data[z_scores > 3]
            
            st.success(f"**Number of Outliers:** {len(outliers)}")
            
            if len(outliers) > 0:
                if st.checkbox("🔎 View Detected Outliers"): 
                    st.dataframe(outliers[['Manufacturer', 'Model', 'Year of manufacture', selected_col]])

            # Plotly Histogram with outliers highlighted
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered_data[selected_col],
                name='Normal',
                nbinsx=30,
                marker_color='#636EFA'
            ))
            if len(outliers) > 0:
                fig.add_trace(go.Histogram(
                    x=outliers[selected_col],
                    name='Outliers',
                    nbinsx=30,
                    marker_color='red'
                ))
            fig.update_layout(
                barmode='overlay',
                title=f"Distribution of {selected_col} with Outliers Highlighted",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            fig.update_traces(opacity=0.75)
            st.plotly_chart(fig, use_container_width=True)

            # Insights & Recommendations
            st.markdown("##### 📊 Insights & Recommendations")
            st.write(f"- {len(outliers)} outliers detected using Z-Score method in {selected_col}.")
            if len(outliers) > 0:
                st.info("- Outliers could affect mean-based statistics; consider trimming or transforming values.")
            else:
                st.success("- No extreme outliers found; distribution is acceptable.")
    else:
        st.warning("No data available for outlier analysis after filtering")
with visualization:
    st.markdown('<p class="sub-header">Data Visualizations</p>', unsafe_allow_html=True)

    if len(filtered_data) > 0:
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Histograms", "Scatter Plots", "Categorical Analysis"]
        )

        # ------------------- Histograms -------------------
        if viz_type == "Histograms":
            selected_col = st.selectbox("Select Column", numerical_cols)

            # Row with histogram and boxplot
            row1_col1, row1_col2 = st.columns([1,1], gap="large")

            with row1_col1:
                fig = px.histogram(
                    filtered_data, 
                    x=selected_col, 
                    nbins=30, 
                    marginal="box",
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#636EFA']
                )
                fig.update_layout(bargap=0.1, margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

                # Insights & Recommendations
                mean_val = filtered_data[selected_col].mean()
                median_val = filtered_data[selected_col].median()
                std_val = filtered_data[selected_col].std()
                skew_val = filtered_data[selected_col].skew()
                st.markdown("##### 📊 Insights & Recommendations")
                st.write(f"- Mean: {mean_val:.2f}, Median: {median_val:.2f}, std: {std_val:.2f}")
                if abs(skew_val) > 0.5:
                    skew_direction = "right-skewed" if skew_val > 0 else "left-skewed"
                    st.info(f"- Distribution is {skew_direction} (skewness: {skew_val:.2f}) → consider transformations for modeling")
                else:
                    st.success("- Distribution approximately symmetric → suitable for most analyses")

            with row1_col2:
                fig = go.Figure()
                for col in numerical_cols:
                    fig.add_trace(go.Box(y=filtered_data[col], name=col))
                fig.update_layout(title="Boxplot of Numerical Features", margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### 📊 Insights & Recommendations")
                for col in numerical_cols:
                    iqr = filtered_data[col].quantile(0.75) - filtered_data[col].quantile(0.25)
                    st.write(f"- {col}: IQR = {iqr:.2f}")
                most_variable = numerical_cols[np.argmax([filtered_data[col].std() for col in numerical_cols])]
                st.info(f"- Most variable feature: {most_variable} → consider normalizing for modeling")

        # ------------------- Scatter Plots -------------------
        elif viz_type == "Scatter Plots":
            row1_col1, row1_col2 = st.columns([1,1], gap="large")
            with row1_col1:
                x_axis = st.selectbox("X-Axis", numerical_cols, index=0)
            with row1_col2:
                y_axis = st.selectbox("Y-Axis", numerical_cols, index=len(numerical_cols)-1)

            fig = px.scatter(
                filtered_data,
                x=x_axis,
                y=y_axis,
                color='Fuel type',
                trendline="ols",
                title=f"{y_axis} vs {x_axis} by Fuel Type"
            )
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### 📊 Insights & Recommendations")
            correlation = filtered_data[x_axis].corr(filtered_data[y_axis])
            correlation_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
            direction = "positive" if correlation > 0 else "negative"
            st.write(f"- Correlation coefficient: {correlation:.3f} ({correlation_strength} {direction})")
            if abs(correlation) > 0.3:
                st.success(f"- As {x_axis} increases, {y_axis} tends to {'increase' if correlation > 0 else 'decrease'} → useful predictor for modeling")
            else:
                st.warning("- Weak linear relationship → consider nonlinear models or feature engineering")

        # ------------------- Categorical Analysis -------------------
        elif viz_type == "Categorical Analysis":
            categorical_cols = filtered_data.select_dtypes(include='object').columns
            categorical_col = st.selectbox("Select Categorical Column", categorical_cols)

            row1_col1, row1_col2 = st.columns([1,1], gap="large")

            with row1_col1:
                counts = filtered_data[categorical_col].value_counts()
                fig = px.bar(
                    x=counts.values, 
                    y=counts.index, 
                    orientation='h',
                    title=f"Count of Cars by {categorical_col}",
                    text=counts.values,
                    color_discrete_sequence=['#636EFA']
                )
                fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### 📊 Insights & Recommendations")
                dominant_category = counts.index[0]
                dominant_percentage = (counts.iloc[0] / counts.sum()) * 100
                st.write(f"- Most common category: {dominant_category} ({dominant_percentage:.1f}%)")
                if dominant_percentage > 50:
                    st.success(f"- {dominant_category} dominates the category → consider balanced sampling if modeling")

            with row1_col2:
                avg_price = filtered_data.groupby(categorical_col)['Price'].mean().sort_values(ascending=False)
                fig = px.bar(
                    x=avg_price.values, 
                    y=avg_price.index, 
                    orientation='h',
                    title=f"Average Price by {categorical_col}",
                    text=[f"${v:,.0f}" for v in avg_price.values],
                    color_discrete_sequence=['#EF553B']
                )
                fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### 📊 Insights & Recommendations")
                highest_price_cat = avg_price.index[0]
                lowest_price_cat = avg_price.index[-1]
                price_ratio = avg_price.iloc[0] / avg_price.iloc[-1]
                st.write(f"- Price range: {highest_price_cat} vs {lowest_price_cat} → {price_ratio:.1f}x difference")
                if price_ratio > 3:
                    st.info("- High price disparity → consider price normalization or log transformation for analysis")

       

    else:
        st.warning("No data available for visualization after filtering")

with prediction:
    st.header("Car Price Prediction")

    # Mapping manufacturer to models
    manufacturer_models = {
        "BMW": ["M5", "X3", "Z4"],
        "Ford": ["Fiesta", "Focus", "Mondeo"],
        "Porsche": ["911", "718 Cayman", "Cayenne"],
        "Toyota": ["Prius", "RAV4", "Yaris"],
        "VW": ["Golf", "Passat", "Polo"]
    }

    f1, f2, f3, = st.columns(3)
    with f1:
        engine_size = st.selectbox("Engine size", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0, 3.5, 4.0, 4.4, 5.0])

    with f2:
        year = st.slider("Year", min_value=1984, max_value=2025, value=2025)

    with f3:
        mileage = st.slider("Mileage", min_value=0, max_value=500000, value=100000, step=1000)

    f4, f5, f6, = st.columns(3)
    with f4:
        manufacturer = st.selectbox("Manufacturer", list(manufacturer_models.keys()))

    with f5:
        model = st.selectbox("Model", manufacturer_models[manufacturer])

    with f6:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])

    if st.button("Predict"):
        url = "http://localhost:8000/predict"
        payload = {
            "engine_size": float(engine_size),
            "year": int(year),
            "mileage": int(mileage),
            "manufacturer": manufacturer,
            "model": model,
            "fuel_type": fuel_type,
        }
        try:
            # send JSON body (POST)
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            stat, ml, dl, = st.columns(2)
            with stat:
                st.success(f"Price (statsmodels): ${abs(result.get('stat_price')):,.2f}")
            with ml:
                st.success(f"Price (Randomforest): ${result.get('ml_price'):,.2f}")
        except Exception as e:
            st.error(f"Error connecting to prediction API: {e}")
            st.info("Make sure your FastAPI server is running on http://127.0.0.1:8000")


# FastAPI endpoint configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 API Configuration")
api_url = st.sidebar.text_input(
    "FastAPI Endpoint URL", 
    value="http://localhost:8000/predict",
    help="Enter the URL of your FastAPI prediction endpoint"
)

#st.markdown('<p class="sub-header">Insights & Recommendations</p>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Car Sales Dashboard | Created with Streamlit")