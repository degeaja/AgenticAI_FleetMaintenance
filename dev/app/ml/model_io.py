import torch, joblib
from app.config import MODEL_PATH, SCALER_PATH, DEVICE
from app.ml.lstm_arch import LSTMModel   # <-- you must define your LSTMModel somewhere

def load_model_and_scaler(features_len: int):
    model = LSTMModel(input_size=features_len, hidden_sizes=[200, 100, 50], dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
