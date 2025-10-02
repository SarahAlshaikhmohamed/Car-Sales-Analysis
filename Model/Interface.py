from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import os
from Encoder import encode, decode


# Load models
stat_model = joblib.load("./STM/st_model.joblib")
stat_scaler = pickle.load("./STM/scaler.pkl")
ml_model   = joblib.load("./RF/rf_model.joblib")
ml_scaler = pickle.load("./RF/scaler.pkl")

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
        
    data = encode(input_data)

    # --- Statsmodels ---
    data = stat_scaler.transform(data)
    data_sm = sm.add_constant(data, has_constant="add")
    stat_model_pred = stat_model.predict(data_sm)[0]

    # --- Scikit-learn ---
    ml_model_pred = ml_model.predict(data)[0]

    return {
        "stat_price": float(stat_model_pred),
        "ml_price": float(ml_model_pred),
    }