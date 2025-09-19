# 🚗 Car Sales Analysis & Prediction

A Data analysis and machine learning project that explores car sales trends, builds predictive models, and provides interactive dashboards for key insights. This project was built as part of the Data Science and Machine Learning Bootcamp at Tuwaiq Academy  - Week Two Project.

---
### 🎯 Objectives
- Analyze car sales trends and identify key factors influencing sales.
- Build a regression model to predict price and year.
- Develop an interactive dashboard for visualization.
- Provide actionable recommendations based on findings.

---
### ✨ Features
- 📊 Exploratory Data Analysis (EDA): Trends, correlations, and statistical summaries.
- 🤖 Regression Modeling: Predict price and year using features like price, fuel type, engine size, etc.
- 📈 Interactive Dashboard: Built with Streamlit + Plotly.
- ⚡ Fast API: Used for backend integration with regression model.

---
### 🛠 Tech Stack
- Python (NumPy, Pandas, Matplotlib).
- Streamlit (dashboard visualization).
- Plotly (dashboard plots).
- FastAPI (backend model integration).
- Scikit-learn (model training).

---
### 📂 Project Structure
```bash
Car-Sales-Analysis/
│── Analytics/                # EDA, Preprocessing, & Dashboard
    └── Dashboard.py          # Streamlit Dashboard App
    └── Preprocessing.ipynb   # EDA & Cleaning Notebook
│── Dataset/                  # Raw & cleaned Datasets
    └── car_sales_data.csv    # Original Dataset
    └── cleand.scv            # Procssed Dataset
│── Model/                    # Model Training, Saved Model, & Interface
    └── Train.ipynb           # Model Build & Train
    └── Interface.py          # Fast API App
    └── Model.pkl             # Trained Model
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
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
- Run FastAPI Server
  ```bash
  python -m uvicorn Interface:app --reload
  ```
- Run Streamlit dashboard
  ```bash
  python -m streamlit run Dashboard.py
  ```
- View Dashboard
  
  [Car Sales Dashboard](http://192.168.0.60:8501)
  
---
### 🌐 API Endpoints (Will be updated)
The FastAPI backend exposes several endpoints:
| Method | Endpoint | Description | 
|--------|---------------|--------------------------------------| 
| GET | `/` | Welcome message / API health check | 
| POST | `/predict` | Predicts car sales given input data | 
| GET | `/sales-trends` | Returns processed sales trends data | 
| GET | `/cars` | Fetches list of available cars |

Example: 
- Prediction Request
  ```bash
  POST /predict
  Content-Type: application/json

  {
    "year": 2018,
    "price": 20000,
    "fuel_type": "Petrol",
    "engine_size": 1800
  }
  ```
- Prediction Response
  ```bash
  {
  "predicted_sales": 345
  }
  ```

---
### 📈 Results & Insights (Will be updated)
- Key trends identified in pricing, sales distribution, and fuel type.
- Regression model trained to predict car sales.
- Dashboard provides interactive exploration of results.
- (Screenshots / GIFs of dashboard can be added here later.)

---
### 🚀 Recommendations & Future Work (Will be updated)
- Add more ML models (Random Forest, XGBoost).
- Deploy API & dashboard on cloud (Heroku / Render / AWS).
- Expand dataset with more real-world data.

---
### 👥 Contributors
- Khalid Khubrani.
- Nouf Almutiri.
- Sarah Alshaikhmohamed.

---
### 📽️ Presentation  (Will be updated)
[Project Presentation]()

---
### 📜 License
This project is licensed under the MIT License.
