from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import gdown
import os


# Load trained model using googledive
MODEL_DIR = "Models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Direct download links
stat_model_url = "https://drive.google.com/uc?id=1D2tHQZF-7zdofnOCdN0b5P4-mL9g_DN9"
ml_model_url   = "https://drive.google.com/uc?id=1ugWkLWTWYCid9zupNV6JZvXbeR6i2G17"
dl_model_url   = "https://drive.google.com/uc?id=1APHXvn-olqsqnmYQ2i0XFXP2UanR7FJg"

# Local paths
stat_model_path = os.path.join(MODEL_DIR, "price_model_statsmodels.pkl")
ml_model_path   = os.path.join(MODEL_DIR, "price_model.pkl")
dl_model_path   = os.path.join(MODEL_DIR, "price_prediction_model.keras")

# Download (only if missing)
if not os.path.exists(stat_model_path):
    gdown.download(stat_model_url, stat_model_path, quiet=False)
if not os.path.exists(ml_model_path):
    gdown.download(ml_model_url, ml_model_path, quiet=False)
if not os.path.exists(dl_model_path):
    gdown.download(dl_model_url, dl_model_path, quiet=False)

# Load models
stat_model = joblib.load(stat_model_path)
ml_model   = joblib.load(ml_model_path)
dl_model   = keras.models.load_model(dl_model_path)

app = FastAPI()

FEATURES = [
    "Engine size", "Year of manufacture", "Mileage",
    "Manufacturer_Ford", "Manufacturer_Porsche", "Manufacturer_Toyota", "Manufacturer_VW",
    "Model_911", "Model_Cayenne", "Model_Fiesta", "Model_Focus", "Model_Golf",
    "Model_M5", "Model_Mondeo", "Model_Passat", "Model_Polo", "Model_Prius",
    "Model_RAV4", "Model_X3", "Model_Yaris", "Model_Z4",
    "Fuel type_Hybrid", "Fuel type_Petrol"
]

# Define request body schema
class PredictionInput(BaseModel):
    engine_size: float
    year: int 
    mileage: int 
    manufacturer: str
    model: str 
    fuel_type: str 

@app.get("/")
def home():
    return {"message": "Price Prediction API is running!"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    def encode_input(user: PredictionInput) -> pd.DataFrame:
        row = {col: 0 for col in FEATURES}
        row["Engine size"] = user.engine_size
        row["Year of manufacture"] = user.year
        row["Mileage"] = user.mileage

        manuf_col = f"Manufacturer_{user.manufacturer}"
        if manuf_col in row:
            row[manuf_col] = 1

        model_col = f"Model_{user.model}"
        if model_col in row:
            row[model_col] = 1

        fuel_col = f"Fuel type_{user.fuel_type}"
        if fuel_col in row:
            row[fuel_col] = 1

        return pd.DataFrame([row], columns=FEATURES)

    # Encode input
    data = encode_input(input_data)

    # --- Statsmodels ---
    data_sm = sm.add_constant(data, has_constant="add")
    stat_model_pred = stat_model.predict(data_sm)[0]

    # --- Scikit-learn ---
    ml_model_pred = ml_model.predict(data)[0]

    # --- Keras ---
    dl_input = data[["Engine size", "Mileage"]].to_numpy()
    dl_model_pred = dl_model.predict(dl_input)[0][0]

    return {
        "stat_price": float(stat_model_pred),
        "ml_price": float(ml_model_pred),
        "dl_price": float(dl_model_pred),
    }