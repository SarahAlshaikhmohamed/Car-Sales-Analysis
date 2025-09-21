# 🚗 Car Sales Analysis & Prediction

This is a data science and machine learning project that analyzes car sales trends, builds predictive models, and provides an interactive dashboard with real-time price predictions. Developed during the Tuwaiq Academy Data Science & ML Bootcamp (Week 2 Project).

---
### 🎯 Objectives
- Analyze car sales data to uncover trends and patterns.
- Identify key factors influencing car prices.
- Build and compare multiple regression models (Statsmodels, Scikit-learn, Keras).
- Deploy a FastAPI backend with endpoints for real-time car price prediction.
- Create an interactive dashboard (Streamlit + Plotly) for data visualization and user input.

---
### ✨ Features
- 📊 Exploratory Data Analysis (EDA): Distribution analysis, correlations, and statistical summaries.
- 🤖 Predictive Modeling:
      - Statsmodels regression.
      - Scikit-learn regression model.
      - Deep Learning (Keras Sequential model).
- ⚡ FastAPI Backend: REST API for car price predictions (/predict).
- 📈 Interactive Dashboard: Built with Streamlit + Plotly for user-friendly visualizations.
- ☁️ Model Hosting: Large model files stored in Google Drive, automatically downloaded at runtime.

---
### 🛠 Tech Stack
- Python: NumPy, Pandas, Matplotlib, Seaborn.
- Visualization: Streamlit, Plotly.
- Machine Learning: Scikit-learn, Statsmodels.
- Deep Learning: TensorFlow / Keras.
- Backend: FastAPI (for API endpoints).
- Model Hosting: Google Drive + gdown for large model files.
- Version Control: Git + GitHub.

---
### 📂 Project Structure
```bash
Car-Sales-Analysis/
│── Analytics/                                        # Preprocessing & Dashboard
    └── Dashboard.py                                  # Streamlit Dashboard App
    └── Preprocessing.ipynb                           # EDA & Cleaning Notebook
    └── car.png                                       # asset
│── Dataset/                                          # Raw & cleaned Datasets
    └── car_sales_data.csv                            # Original Dataset
    └── processed_car_sales_data_cleaning.scv         # Procssed Dataset
│── Model/                                            # Model Train & Interface
    │── Models/
       └── price_model.pkl                            # Trained ML Model
       └── price_model_statsmodels.pkl                # Trained statmodels Model
       └── price_prediction_model.keras               # Trained DL Model
    └── Machine_Learning_Train.ipynb                  # ML Model Train
    └── Statmodels_Train.ipynb                        # statmodels Model Train
    └── Deep_Learning_Train.ipynb                     # DL Model Train
    └── Interface.py                                  # Fast API App
    └── Model.pkl                                     # Trained Model
    └── model_metadata                                # Models Metadata
│── requirements.txt                                  # Dependencies
│── README.md                                         # Project documentation
```

---
### 📊 Dataset
This dataset is useful for analyzing trends in the automotive industry, such as identifying popular car models, and pricing patterns. It can support projective model for car prices, comparisons between new and used vehicle sales, and understanding customer priorties over time. The data may sourced from dealership records, online marketplaces, or scraping automobile websites.

- Contains the attributes:manufacturer, model, engine size, fuel type ,year of manfacture , mileage and price.
- Used for cleaning, visualization, and predictive modeling.
- Source: [Kaggle – Car Sales Info](https://www.kaggle.com/datasets/minahilfatima12328/car-sales-info/data).

---
### ⚙️ Installation
1. Clone Repository
   ``` bash
   git clone https://github.com/SarahAlshaikhmohamed/Car-Sales-Analysis.git
   cd Car-Sales-Analysis
   ```
2. (optional) Create a Virtual Environment
   1. UV Environment:
      ```bash
      pip install uv
      uv venv my-venv
      my-venv\Scripts\Activate
      uv init
      ```
   2. Virtual Environment (Windows):
      ```bash
      python -m venv my-venv
      my-venv\Scripts\Activate
      ```
   3. Virtual Environment (Linux):
      ```bash
      python3 -m venv my-venv
      source my-venv/bin/activate
      ```
3. Install Dependencies
   1. UV Environment:
      ```bash
      uv add requirements.txt
      ```
   2. Virtual Environment
      ```bash
      pip install -r requirements.txt
      ```

---
### ▶️ Usage
- Run EDA & Preprocessing
  ```bash
  python eda/eda_script.py
  ```
- Run Streamlit dashboard
  ```bash
  python -m streamlit run Dashboard.py
  ```
- Run FastAPI Server
  ```bash
  python -m uvicorn Interface:app --reload
  ```
- View Dashboard
  
  [Car Sales Dashboard](http://192.168.0.60:8501)
  
---
### 🌐 API Endpoints
The FastAPI backend exposes several endpoints:
| Method | Endpoint | Description | Request Body (JSON) | Response (JSON) |
|--------|----------|-------------|----------------------|-----------------|
| GET | `/` | Verify that the API is running | None | { "message": "Price Prediction API is running!" }
| POST | `/predict` | Predict car price using Statsmodels, Scikit-learn, and Keras models | json { "engine_size": 2.0, "year": 2018, "mileage": 50000, "manufacturer": "Ford", "model": "Focus", "fuel_type": "Petrol" } | json { "stat_price": 15234.56, "ml_price": 14987.33, "dl_price": 15100.12 }

---
### 📈 Results & Insights
1. Outlier Detection: Identified extreme values in Price, Mileage, and Year that could represent data entry errors or rare luxury/classic vehicles.
2. Price and mileage are two sides of the same coin and are fundamental for setting competitive market prices.
3. Finding: Price has a strong positive correlation with two key factors:
   - Year of Manufacture (+0.7): Newer cars are priced higher, as expected.
   - Engine Size (+0.6): Vehicles with larger engines are more expensive.
4. Mileage showed a weaker negative correlation with Price, suggesting that while higher mileage reduces price, the car's age and engine size are more influential determinants of its value.

---
### 🚀 Recommendations & Future Work
 - Dealers should focus on competitive pricing in popular ranges.
 - Buyers should consider mileage as a major factor for value.
 - Fine tuning for the ML and state model.
 - Deploy API & dashboard on cloud (Heroku / Render / AWS).
 - Expand dataset with more real-world data.

---
### 👥 Contributors
- Khalid Khubrani.
- Nouf Almutiri.
- Sarah Alshaikhmohamed.

---
### 📽️ Presentation
[Project Presentation](https://www.canva.com/design/DAGzhjHhfYQ/oyH98ndXGWj5ZIAcw43jTQ/edit?utm_content=DAGzhjHhfYQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---
### 📜 License
This project is licensed under the MIT License.
