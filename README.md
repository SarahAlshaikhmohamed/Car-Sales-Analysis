# 🚗 Car Sales Analysis & Prediction

This is a data science and machine learning project that analyzes car sales trends, builds predictive models, and provides an interactive dashboard with real-time price predictions. Developed during the Tuwaiq Academy Data Science & ML Bootcamp (Week 2 Project).

---
### 🎯 Objectives
- Analyze car sales data to uncover trends and patterns.
- Identify key factors influencing car prices.
- Build and compare regression models (Statsmodels & Scikit-learn).
- Deploy a FastAPI backend with endpoints for real-time car price prediction.
- Create an interactive dashboard (Streamlit + Plotly) for data visualization and user input.

---
### ✨ Features
- 📊 Exploratory Data Analysis (EDA): Distribution analysis, correlations, and statistical summaries.
- 🤖 Predictive Modeling:
      - Statsmodels regression.
      - Scikit-learn regression model.
- ⚡ FastAPI Backend: REST API for car price predictions (/predict).
- 📈 Interactive Dashboard: Built with Streamlit + Plotly for user-friendly visualizations.
- ☁️ Models are preloaded locally in Docker for fast inference.

---
### 🛠 Tech Stack
- Python: NumPy, Pandas, Matplotlib, Seaborn.
- Visualization: Streamlit, Plotly.
- Machine Learning: Scikit-learn, Statsmodels.
- Backend: FastAPI (for API endpoints).
- Version Control: Git + GitHub.

---
### 🐳 Docker Setup
This project is containerized for reproducibility and easy deployment.

```bash
# Build the image
docker build -t car-sales-analysis .

# Run container
docker run -p 8000:8000 -p 7860:7860 car-sales-analysis
```

---
### 📂 Project Structure
```bash
Car-Sales-Analysis/
│
├── .streamlit/                            # Streamlit configuration
│   └── config.toml
│
├── Analytics/                             # Streamlit dashboard and preprocessing
│   ├── .gitattributes
│   ├── Dashboard.py                       # Main Streamlit dashboard (frontend UI)
│   ├── Preprocessing.ipynb                # Data preprocessing notebook
│   └── car.png                            # Dashboard image
│
├── Dataset/                               # Raw and cleaned datasets
│   ├── car_sales_data.csv
│   └── processed_car_sales_data_cleaning.csv
│
├── app/                                   # FastAPI backend and ML models
│   ├── RF/                                # Random Forest model files
│   │   ├── model_metadata.json
│   │   ├── rf_model.joblib
│   │   └── scaler.pkl
│   │
│   ├── STM/                               # Statsmodels regression model files
│   │   ├── model_metadata.json
│   │   ├── st_model.joblib
│   │   └── scaler.pkl
│   │
│   ├── __pycache__/                       # Compiled Python cache
│   │   ├── Encoder.cpython-310.pyc
│   │   └── Interface.cpython-310.pyc
│   │
│   ├── Encoder.py                         # Encoding and feature transformation logic
│   ├── Machine_Learning_Train.ipynb       # Random Forest model training notebook
│   ├── Statmodels_Train.ipynb             # Statsmodels training notebook
│   ├── __init__.py
│   └── app.py                             # FastAPI entry point (backend API)
│
├── .gitattributes
├── Dockerfile                             # Docker container definition (runs Streamlit + FastAPI)
├── README.md                              # Project documentation
├── app.sh                                 # Shell script to launch both services
├── requirements.txt                       # Python dependencies
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
Start both the FastAPI backend (port 8000) and Streamlit dashboard (port 7860):
```bash
bash app.sh
```

Then open:

🌐 Dashboard: [http://localhost:7860](http://localhost:7860)

⚙️ API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

  
---
### 🌐 API Endpoints
The FastAPI backend exposes several endpoints:
| Method | Endpoint | Description | Request Body (JSON) | Response (JSON) |
|--------|----------|-------------|----------------------|-----------------|
| GET | `/` | Verify that the API is running | None | { "message": "Price Prediction API is running!" }
| POST | `/predict` | Predict car price using Statsmodels and Scikit-learn models | json { "engine_size": 2.0, "year": 2018, "mileage": 50000, "manufacturer": "Ford", "model": "Focus", "fuel_type": "Petrol" } | json { "stat_price": 15234.56, "ml_price": 14987.33 }

---
### 🚀 Deployment
This project is fully deployed on **Hugging Face Spaces** using a Docker container that runs both:
- 🧠 FastAPI backend on port `8000`
- 💻 Streamlit dashboard on port `7860`

Visit the live app here:  
[Car Sales Dashboard](https://huggingface.co/spaces/SAliiv52/Car-Sales-Analysis)

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
 - Fine-tuning for the Statsmodels regression model.
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
