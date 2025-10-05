from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import os
from Encoder import encode, decode


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STM_PATH = os.path.join(BASE_DIR, "../STM")
RF_PATH = os.path.join(BASE_DIR, "../RF")

# Load models
stat_model = joblib.load(os.path.join(STM_PATH, "st_model.joblib"))
stat_scaler = joblib.load(os.path.join(STM_PATH, "scaler.pkl"))
ml_model = joblib.load(os.path.join(RF_PATH, "rf_model.joblib"))
ml_scaler = joblib.load(os.path.join(RF_PATH, "scaler.pkl"))

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

    input_dict = input_data.dict()
    data = encode(input_dict)

    expected_features = 23
    current_features = data.shape[1]

    if current_features < expected_features:
        # pad with zeros (difference at the end)
        pad_width = expected_features - current_features
        data = np.hstack([data, np.zeros((data.shape[0], pad_width))])
    elif current_features > expected_features:
        # truncate extra features just in case
        data = data[:, :expected_features]

    # --- Statsmodels ---
    data_sm = stat_scaler.transform(data)
    data_sm = sm.add_constant(data_sm, has_constant="add")
    stat_model_pred = stat_model.predict(data_sm)[0]

    # --- Scikit-learn ---
    data_ml = ml_scaler.transform(data)
    ml_model_pred = ml_model.predict(data_ml)[0]

    return {
        "stat_price": float(stat_model_pred),
        "ml_price": float(ml_model_pred),
    }