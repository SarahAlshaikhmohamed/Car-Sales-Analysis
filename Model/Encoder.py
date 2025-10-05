from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

encoder = OneHotEncoder(drop="first")

def encode(data: dict):
    df = pd.DataFrame([data]) 
    encoded = encoder.fit_transform(df[["manufacturer", "model", "fuel_type"]])
    num_features = df[["engine_size", "year", "mileage"]].values
    X = np.hstack([num_features, encoded.toarray()])
    return X

def decode(data):
    decoded = encoder.inverse_transform(data)
    return decoded