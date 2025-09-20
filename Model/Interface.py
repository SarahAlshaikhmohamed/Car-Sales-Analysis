from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load trained model
stat_model = joblib.load("price_model_statsmodels.pkl")
ml_model = joblib.load("price_model.pkl")
dl_model = keras.models.load_model("price_prediction_model.keras")

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