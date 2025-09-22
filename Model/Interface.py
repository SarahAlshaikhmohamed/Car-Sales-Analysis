from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import gdown
import os
from Encoder import encode, decode

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

dl_model = None

@app.post("/predict")
def predict(input_data: PredictionInput):
    global dl_model
    if dl_model is None:
        from tensorflow import keras
        dl_model = keras.models.load_model(dl_model_path)
        
    data = encode(input_data)

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
