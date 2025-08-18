"""
streamline model loading and scaling
"""
import torch
import joblib
from app.ml.config import MODEL_PATH, SCALER_PATH, DEVICE
from app.ml.lstm_arch import LSTMModel

def load_model_and_scaler(features_len: int):
    model = LSTMModel(input_size=features_len, hidden_sizes=[256, 128], dropout=0.2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
