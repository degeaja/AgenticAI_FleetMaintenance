import torch
SEQUENCE_LENGTH = 10
CSV_PATH = "app/fleet_monitor_notscored_2.csv"
SCALER_PATH = "app/scaler.save"
MODEL_PATH = "app/best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
