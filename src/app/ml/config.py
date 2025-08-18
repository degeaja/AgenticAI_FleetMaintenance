import torch

SEQUENCE_LENGTH = 10
CSV_PATH = "app/ml/fleet_monitor_notscored_2.csv"
SCALER_PATH = "app/ml/scaler.save"
MODEL_PATH = "app/ml/best_model_08-18.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
