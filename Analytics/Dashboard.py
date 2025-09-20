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
    .dataset-description {background-color: #f0f8ff; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🚗 Car Sales Dashboard</h1>', unsafe_allow_html=True)

# Dataset Description
with st.expander("📊 Dataset Description and Analysis Overview", expanded=True):
    st.markdown("""
    <div class="dataset-description">
    <h3>About the Dataset</h3>
    <p>This dashboard analyzes a comprehensive car sales dataset containing information about various vehicles 
    including their specifications, pricing, and sales information. The dataset includes both numerical and 
    categorical attributes that help in understanding the car market trends.</p>
    
    <h3>What We're Analyzing</h3>
    <ul>
        <li><strong>Pricing Trends:</strong> How car prices vary by manufacturer, model, year, and features</li>
        <li><strong>Market Preferences:</strong> Popular car types, fuel types, and engine sizes</li>
        <li><strong>Sales Performance:</strong> Which manufacturers and models are performing best in the market</li>
        <li><strong>Feature Correlations:</strong> Relationships between different car attributes and their impact on price</li>
        <li><strong>Outlier Detection:</strong> Identifying unusual patterns or anomalies in the data</li>
    </ul>
    
    <h3>Key Metrics Tracked</h3>
    <ul>
        <li>Average car prices by manufacturer and fuel type</li>
        <li>Distribution of cars by year of manufacture</li>
        <li>Mileage patterns and their relationship to price</li>
        <li>Engine size preferences across different car segments</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("Explore the car sales dataset with interactive charts and statistical insights")

# Function to load data
@st.cache_data
def load_data():
    try:
        # Load your dataset .read_csv("../Dataset/winners_f1_cleaned.csv")
        df = pd.read_csv("../Dataset/car_sales_data.csv")
        
        st.success(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns!")
        return df
    except:
        st.error("Dataset not found. Please place 'Dataset\car_sales_data.csv' in the same folder.")
        return pd.DataFrame()

   

        
# Load data
data = load_data()

# Sidebar for dataset describtion and filters and information
with st.sidebar:
    st.image("https://static.thenounproject.com/png/789273-200.png", width=100)
    with st.sidebar:
        st.header("About the Dataset")
    st.markdown("""
    **Car Sale Information**
    
    The car sales dataset includes detailed information about individual 
    vehicle trade. It contain the columns like manufacturer, model, 
    engine size, fuel type ,year of manfacture , mileage and price
    """)


    
    st.title("Filters")
    
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

# Display dataset information
st.markdown('<p class="sub-header">Dataset Information</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Number of Rows", filtered_data.shape[0])
with col2:
    st.metric("Number of Columns", filtered_data.shape[1])


# Quick insights based on filters
st.markdown("### 🔍 Quick Insights")
insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    avg_mileage = filtered_data['Mileage'].mean()
    st.metric("Average Mileage", f"{avg_mileage:,.0f} miles")
    
with insight_col2:
    most_common_fuel = filtered_data['Fuel type'].mode()[0] if len(filtered_data) > 0 else "N/A"
    st.metric("Most Common Fuel Type", most_common_fuel)
    
with insight_col3:
    if len(filtered_data) > 0:
        newest_car = filtered_data['Year of manufacture'].max()
        oldest_car = filtered_data['Year of manufacture'].min()
        st.metric("Year Range", f"{oldest_car} - {newest_car}")
    else:
        st.metric("Year Range", "N/A")

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 , tab7 = st.tabs(["Data Preview", "Descriptive Statistics", "Visualizations", "Outlier Analysis", "Data Quality", "Fuel & Mileage Analysis","Prediction"])

with tab1:
    st.markdown('<p class="sub-header">Data Preview</p>', unsafe_allow_html=True)
    st.dataframe(filtered_data, use_container_width=True)
    
    # Download button
    if len(filtered_data) > 0:
        csv = filtered_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="filtered_car_sales_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No data to download after filtering")

with tab2:
    st.markdown('<p class="sub-header">Descriptive Statistics</p>', unsafe_allow_html=True)
    
    if len(filtered_data) > 0:
        # Numerical columns
        numerical_cols = filtered_data.select_dtypes(include=np.number).columns
        numerical = filtered_data[numerical_cols]
        
        # Calculate statistics
        st.write("### Numerical Data Statistics")
        stats_df = numerical.describe().T
        stats_df['range'] = stats_df['max'] - stats_df['min']
        st.dataframe(stats_df)
        
       


        # Correlation heatmap
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numerical.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.write("### 📊 Correlation Insights ")
        st.info("""
        -Price is driven up by newer manufacturing year and larger engine size.\n
        -Price is driven down by higher mileage.\n
        -Newer cars → lower mileage, explaining the negative link between mileage and year.\n
        -Engine size doesn’t meaningfully affect mileage or age.\n
        """)
        
    else:
        st.warning("No data available for analysis after filtering")

with tab3:
    st.markdown('<p class="sub-header">Data Visualizations</p>', unsafe_allow_html=True)
    
    if len(filtered_data) > 0:
        # Select visualization type
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Histograms", "Scatter Plots", "Categorical Analysis", "Price Distribution", "Mileage Analysis"]
        )
        
        if viz_type == "Histograms":
            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox("Select Column", numerical_cols)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(filtered_data[selected_col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {selected_col}")
                st.pyplot(fig)
                
                # Histogram insights
                st.markdown("### 📊 Histogram Insights")
                mean_val = filtered_data[selected_col].mean()
                median_val = filtered_data[selected_col].median()
                std_val = filtered_data[selected_col].std()
                skew_val = filtered_data[selected_col].skew()
                
                st.metric("Mean", f"{mean_val:.2f}")
                st.metric("Median", f"{median_val:.2f}")
                st.metric("Standard Deviation", f"{std_val:.2f}")
                
                if abs(skew_val) > 0.5:
                    skew_direction = "right-skewed" if skew_val > 0 else "left-skewed"
                    st.info(f"**Distribution Shape:** {skew_direction} (skewness: {skew_val:.2f})")
                else:
                    st.success("**Distribution Shape:** Approximately symmetric")
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                numerical.boxplot(ax=ax)
                ax.set_title("Boxplot of Numerical Features")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Boxplot insights
                st.markdown("### 📊 Boxplot Insights")
                st.write("**Spread Comparison:**")
                for col in numerical_cols:
                    iqr = filtered_data[col].quantile(0.75) - filtered_data[col].quantile(0.25)
                    st.write(f"- {col}: IQR = {iqr:.2f}")
                
                # Identify most variable feature
                most_variable = numerical_cols[np.argmax([filtered_data[col].std() for col in numerical_cols])]
                st.success(f"**Most Variable Feature:** {most_variable}")
        
        elif viz_type == "Scatter Plots":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-Axis", numerical_cols, index=0)
            with col2:
                y_axis = st.selectbox("Y-Axis", numerical_cols, index=len(numerical_cols)-1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=filtered_data, x=x_axis, y=y_axis, hue='Fuel type', ax=ax)
            ax.set_title(f"{y_axis} vs {x_axis} by Fuel Type")
            st.pyplot(fig)
            
            # Scatter plot insights
            st.markdown("### 📊 Scatter Plot Insights")
            
            # Calculate correlation
            correlation = filtered_data[x_axis].corr(filtered_data[y_axis])
            correlation_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
            direction = "positive" if correlation > 0 else "negative"
            
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
            st.info(f"**Relationship:** {correlation_strength} {direction} correlation")
            
            # Regression line insight
            if abs(correlation) > 0.3:
                st.success(f"As {x_axis} increases, {y_axis} tends to {'increase' if correlation > 0 else 'decrease'}")
            else:
                st.warning("No strong linear relationship detected between these variables")
        
        elif viz_type == "Categorical Analysis":
            categorical_cols = filtered_data.select_dtypes(include='object').columns
            categorical_col = st.selectbox("Select Categorical Column", categorical_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                # Count plot
                fig, ax = plt.subplots(figsize=(10, 6))
                counts = filtered_data[categorical_col].value_counts()
                sns.countplot(data=filtered_data, y=categorical_col, ax=ax, 
                             order=counts.index)
                ax.set_title(f"Count of Cars by {categorical_col}")
                st.pyplot(fig)
                
                # Count plot insights
                st.markdown("### 📊 Count Insights")
                dominant_category = counts.index[0]
                dominant_percentage = (counts.iloc[0] / counts.sum()) * 100
                
                st.metric("Most Common", dominant_category)
                st.metric("Market Share", f"{dominant_percentage:.1f}%")
                st.metric("Unique Values", len(counts))
                
                if dominant_percentage > 50:
                    st.success(f"**Market Dominance:** {dominant_category} dominates this category")
            
            with col2:
                # Average price by category
                fig, ax = plt.subplots(figsize=(10, 6))
                avg_price = filtered_data.groupby(categorical_col)['Price'].mean().sort_values(ascending=False)
                sns.barplot(x=avg_price.values, y=avg_price.index, ax=ax)
                ax.set_title(f"Average Price by {categorical_col}")
                ax.set_xlabel("Average Price")
                st.pyplot(fig)
                
                # Price insights
                st.markdown("### 📊 Price Insights")
                highest_price_cat = avg_price.index[0]
                highest_price = avg_price.iloc[0]
                lowest_price_cat = avg_price.index[-1]
                lowest_price = avg_price.iloc[-1]
                
                st.metric("Highest Average Price", f"{highest_price_cat} (${highest_price:,.0f})")
                st.metric("Lowest Average Price", f"{lowest_price_cat} (${lowest_price:,.0f})")
                
                price_ratio = highest_price / lowest_price
                st.info(f"**Price Range:** {price_ratio:.1f}x difference between highest and lowest")
        
        elif viz_type == "Price Distribution":
            # Price distribution by manufacturer
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=filtered_data, x='Manufacturer', y='Price', ax=ax)
            ax.set_title("Price Distribution by Manufacturer")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Price distribution insights
            st.markdown("### 📊 Price Distribution Insights")
            
            # Calculate price statistics by manufacturer
            price_stats = filtered_data.groupby('Manufacturer')['Price'].agg(['mean', 'median', 'std']).round(2)
            price_stats = price_stats.sort_values('mean', ascending=False)
            
            st.write("**Price Statistics by Manufacturer:**")
            st.dataframe(price_stats.style.format({'mean': '${:,.2f}', 'median': '${:,.2f}', 'std': '${:,.2f}'}))
            
            # Identify manufacturers with highest and lowest prices
            most_expensive = price_stats.index[0]
            least_expensive = price_stats.index[-1]
            price_difference = price_stats['mean'].iloc[0] - price_stats['mean'].iloc[-1]
            
            st.success(f"**Price Range:** {most_expensive} is ${price_difference:,.0f} more expensive than {least_expensive} on average")
            
        elif viz_type == "Mileage Analysis":
            col1, col2 = st.columns(2)
            
            with col1:
                # Mileage distribution by fuel type
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=filtered_data, x='Fuel type', y='Mileage', ax=ax)
                ax.set_title("Mileage Distribution by Fuel Type")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Mileage by fuel type insights
                st.markdown("### 📊 Mileage by Fuel Type")
                mileage_stats = filtered_data.groupby('Fuel type')['Mileage'].agg(['mean', 'median']).round(0)
                mileage_stats = mileage_stats.sort_values('mean', ascending=False)
                
                for fuel_type, stats in mileage_stats.iterrows():
                    st.metric(f"{fuel_type} Avg Mileage", f"{stats['mean']:,.0f} mi")
                
                highest_mileage_fuel = mileage_stats.index[0]
                st.info(f"**Highest Mileage:** {highest_mileage_fuel} vehicles have the highest average mileage")
            
            with col2:
                # Mileage vs Price scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=filtered_data, x='Mileage', y='Price', hue='Fuel type', ax=ax)
                ax.set_title("Price vs Mileage by Fuel Type")
                st.pyplot(fig)
                
                # Mileage vs Price insights
                st.markdown("### 📊 Price-Mileage Relationship")
                
                correlation = filtered_data['Mileage'].corr(filtered_data['Price'])
                st.metric("Correlation", f"{correlation:.3f}")
                
                if correlation < -0.3:
                    st.success("**Strong Negative Relationship:** Higher mileage generally correlates with lower prices")
                elif correlation > 0.3:
                    st.warning("**Unexpected Positive Relationship:** Higher mileage correlates with higher prices")
                else:
                    st.info("**Weak Relationship:** Mileage has little correlation with price in this dataset")
                
                # Average price per mileage quartile
                filtered_data['Mileage_Quartile'] = pd.qcut(filtered_data['Mileage'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
                quartile_prices = filtered_data.groupby('Mileage_Quartile')['Price'].mean()
                
                st.write("**Average Price by Mileage Quartile:**")
                for quartile, price in quartile_prices.items():
                    st.write(f"- {quartile} Mileage: ${price:,.0f}")
    else:
        st.warning("No data available for visualization after filtering")





with tab4:
    st.markdown('<p class="sub-header">Outlier Analysis</p>', unsafe_allow_html=True)
    
    if len(filtered_data) > 0:
        # Select method for outlier detection
        method = st.radio(
            "Select Outlier Detection Method",
            ["IQR Method", "Z-Score Method"]
        )
        
        selected_col = st.selectbox("Select Column for Outlier Analysis", numerical_cols)
        
        if method == "IQR Method":
            # Calculate IQR
            Q1 = filtered_data[selected_col].quantile(0.25)
            Q3 = filtered_data[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Identify outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = filtered_data[(filtered_data[selected_col] < lower_bound) | 
                                    (filtered_data[selected_col] > upper_bound)]
            
            st.write(f"**IQR:** {IQR:.2f}")
            st.write(f"**Lower Bound:** {lower_bound:.2f}")
            st.write(f"**Upper Bound:** {upper_bound:.2f}")
            st.write(f"**Number of Outliers:** {len(outliers)}")
            
            # Show outliers
            if len(outliers) > 0:
                st.write("**Outliers:**")
                st.dataframe(outliers[['Manufacturer', 'Model', 'Year of manufacture', selected_col]])
            
            # Boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=filtered_data[selected_col], ax=ax)
            ax.set_title(f"Boxplot of {selected_col} with Outliers")
            st.pyplot(fig)
        
        else:  # Z-Score Method
            # Calculate z-scores
            z_scores = np.abs((filtered_data[selected_col] - filtered_data[selected_col].mean()) / 
                             filtered_data[selected_col].std())
            
            # Identify outliers (z-score > 3)
            outliers = filtered_data[z_scores > 3]
            
            st.write(f"**Number of Outliers (z-score > 3):** {len(outliers)}")
            
            # Show outliers
            if len(outliers) > 0:
                st.write("**Outliers:**")
                st.dataframe(outliers[['Manufacturer', 'Model', 'Year of manufacture', selected_col]])
            
            # Distribution plot with outliers highlighted
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(filtered_data[selected_col], kde=True, ax=ax, label='Normal')
            if len(outliers) > 0:
                sns.histplot(outliers[selected_col], kde=True, ax=ax, color='red', label='Outliers')
            ax.set_title(f"Distribution of {selected_col} with Outliers Highlighted")
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("No data available for outlier analysis after filtering")

with tab5:
    st.markdown('<p class="sub-header">Data Quality Assessment</p>', unsafe_allow_html=True)
    
    if len(filtered_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values
            st.write("### Missing Values")
            missing = filtered_data.isnull().sum()
            missing_df = pd.DataFrame({'Column': missing.index, 'Missing Values': missing.values})
            st.dataframe(missing_df)
        
        with col2:
            # Duplicates
            st.write("### Duplicate Records")
            duplicates = filtered_data.duplicated().sum()
            st.metric("Number of Duplicates", duplicates)
            
            if duplicates > 0:
                if st.button("Remove Duplicates"):
                    filtered_data = filtered_data.drop_duplicates()
                    st.success(f"Removed {duplicates} duplicate records!")
                    st.experimental_rerun()
        
        # Data types
        st.write("### Data Types")
        dtypes_df = pd.DataFrame({'Column': filtered_data.columns, 
                                 'Data Type': filtered_data.dtypes.astype(str)})
        st.dataframe(dtypes_df)
    else:
        st.warning("No data available for quality assessment after filtering")

with tab6:
    st.markdown('<p class="sub-header">Fuel Type and Mileage Analysis</p>', unsafe_allow_html=True)
    
    if len(filtered_data) > 0:
        # Create two columns - one for the chart, one for insights
        chart_col, insight_col = st.columns([2, 1])

        with chart_col:
            st.write("### Car Distribution by Fuel Type")
            
            # Calculate value counts for fuel type (using your EDA method)
            counts = filtered_data['Fuel type'].value_counts(normalize=True) * 100  # percentages

            # Separate values < 5% into "Others" (following your EDA approach)
            small = counts[counts < 5].sum()   # sum of all small categories
            counts = counts[counts >= 5]       # keep big categories
            if small > 0:
                counts["Others"] = small       # add Others category

            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(
                counts.values, 
                labels=counts.index, 
                autopct='%1.1f%%', 
                startangle=140,
                colors=sns.color_palette('pastel'),
                explode=[0.05] * len(counts)  # Slightly separate slices
            )
            
            # Style the percentages
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.axis('equal')  # Ensure pie is drawn as a circle
            ax.set_title("Fuel Type Distribution")
            st.pyplot(fig)

        with insight_col:
            st.write("### 📊 Insights")
            st.markdown("""
            <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4;'>
            <h4 style='color: #1f77b4; margin-top: 0;'>Fuel Type Distribution Analysis</h4>
            """, unsafe_allow_html=True)
            
            # Calculate insights
            total_cars = len(filtered_data)
            dominant_fuel = counts.index[0]
            dominant_percentage = counts.values[0]
            
            # Display insights
            st.metric("Total Cars Analyzed", total_cars)
            st.metric("Most Common Fuel Type", f"{dominant_fuel} ({dominant_percentage:.1f}%)")
            
            # Add textual insights based on your EDA findings
            if len(counts) == 3:
                second_fuel = counts.index[1]
                second_percentage = counts.values[1]
                third_fuel = counts.index[2]
                third_percentage = counts.values[2]
                
                st.metric("Second Most Common", f"{second_fuel} ({second_percentage:.1f}%)")
                st.metric("Third Most Common", f"{third_fuel} ({third_percentage:.1f}%)")
                
                st.info(f"""
                **Key Observations:**
                - {dominant_fuel} vehicles dominate the market with {dominant_percentage:.1f}% share
                - {second_fuel} represents {second_percentage:.1f}% of the market
                - {third_fuel} accounts for {third_percentage:.1f}% of vehicles
                - The market shows a clear preference for {dominant_fuel} over other fuel types
                """)
            else:
                # Handle case where we have more or fewer categories
                st.info(f"""
                **Key Observations:**
                - {dominant_fuel} is the most popular fuel type with {dominant_percentage:.1f}% market share
                - The market includes {len(counts)} different fuel types
                - Fuel type distribution shows {'good' if dominant_percentage < 60 else 'strong'} preference for {dominant_fuel}
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)

        #for the Prediction tab 
            with tab7:
                st.markdown('<p class="sub-header"> Prediction</p>', unsafe_allow_html=True)
                st.markdown("""
    <div style='background-color: #1f77b4; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 20px;'>
    <h4 style='color: #f0f8ff; margin-top: 0;'>Machine Learning Price Prediction</h4>
    <p>Use our trained machine learning model to predict car prices based on vehicle features. 
    The model analyzes historical sales data to provide accurate price estimates.</p>
    </div>
    """, unsafe_allow_html=True)
                
                # FastAPI endpoint configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 API Configuration")
    api_url = st.sidebar.text_input(
        "FastAPI Endpoint URL", 
        value="http://localhost:8000/predict",
        help="Enter the URL of your FastAPI prediction endpoint"
    )



             
            

        


       
# Footer
st.markdown("---")
st.markdown("**Car Sales Dashboard** - Created with Streamlit | Data Source: Kaggle https://www.kaggle.com/datasets/minahilfatima12328/car-sales-info/data")