from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop="first")

# encoding function
def encode(data):
    encoded = encoder.fit_transform(data[["Manufacturer", "Model", "Fuel type"]])
    return encoded

# decoding function
def decode(data):
    decoded = encoder.inverse_transform(data)
    return decoded