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

os.makedirs("Models", exist_ok=True)

def safe_download(url, output, min_size=50_000):
    if not os.path.exists(output) or os.path.getsize(output) < min_size:
        print(f"Downloading {output}...")
        gdown.download(url, output, quiet=False, fuzzy=True)

    # Double-check file size
    size = os.path.getsize(output)
    if size < min_size:
        raise ValueError(f"Download of {output} seems incomplete (size={size} bytes).")
    return output

# Direct Google Drive links
stat_model_url = "https://drive.google.com/uc?id=1D2tHQZF-7zdofnOCdN0b5P4-mL9g_DN9"
ml_model_url   = "https://drive.google.com/uc?id=1ugWkLWTWYCid9zupNV6JZvXbeR6i2G17"
dl_model_url   = "https://drive.google.com/uc?id=1APHXvn-olqsqnmYQ2i0XFXP2UanR7FJg"

# Paths for saving
stat_model_path = safe_download(stat_model_url, "Models/price_model_statsmodels.pkl", min_size=50_000)
ml_model_path   = safe_download(ml_model_url, "Models/price_model.pkl", min_size=50_000)
dl_model_path   = safe_download(dl_model_url, "Models/price_prediction_model.keras", min_size=50_000)

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