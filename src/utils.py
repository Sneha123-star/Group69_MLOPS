# utils.py
import joblib
import numpy as np

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_input(data):
    scaler = joblib.load('models/scaler.pkl')
    data = np.array(data).reshape(1, -1)
    return scaler.transform(data)

