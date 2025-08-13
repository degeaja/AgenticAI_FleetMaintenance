import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Configuration ---
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load and preprocess test data ---
df = pd.read_csv("fleet_monitor_notscored_2.csv")  # <-- Update as needed
df["Measurement_timestamp"] = pd.to_datetime(df["Measurement_timestamp"], format="%d%b%y:%H:%M:%S")
df = df.sort_values(by=["Region", "truckid", "Measurement_timestamp"]).reset_index(drop=True)

features = [
    "Vehicle_speed_sensor", "Vibration", "Engine_Load", "Engine_Coolant_Temp", "Intake_Manifold_Pressure",
    "Engine_RPM", "Speed_OBD", "Intake_Air_Temp", "Mass_Air_Flow_Rate", "Throttle_Pos_Manifold",
    "Voltage_Control_Module", "Ambient_air_temp", "Accel_Pedal_Pos_D", "Engine_Oil_Temp", "Speed_GPS",
    "Turbo_Boost_And_Vcm_Gauge", "Trip_Distance", "Litres_Per_100km_Inst", "Accel_Ssor_Total",
    "CO2_in_g_per_km_Inst", "Trip_Time_journey"
]

# Normalize using saved scaler
scaler = joblib.load("scaler.save")
df[features] = scaler.transform(df[features])

# --- Dataset with extra metadata for output ---
class FleetTestDataset(Dataset):
    def __init__(self, df, features, seq_len):
        self.sequences = []
        self.labels = []
        self.meta = []  # to store Region, truckid, timestamp
        for _, group in df.groupby(["Region", "truckid"]):
            if len(group) < seq_len:
                continue
            X = group[features].values
            y = group["Maintenance_flag"].values
            ts = group["Measurement_timestamp"].values
            for i in range(len(X) - seq_len):
                self.sequences.append(X[i:i+seq_len])
                self.labels.append(y[i+seq_len])
                self.meta.append({
                    "Region": group["Region"].iloc[i+seq_len],
                    "truckid": group["truckid"].iloc[i+seq_len],
                    "Measurement_timestamp": str(group["Measurement_timestamp"].iloc[i+seq_len])  # convert to string
                })


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y, self.meta[idx]
    
def custom_collate_fn(batch):
    xs, ys, metas = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys, metas  # metas stays as a list of dicts


test_dataset = FleetTestDataset(df, features, SEQUENCE_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

# Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[100, 50], dropout=0.2):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], batch_first=True))
        self.dropout_layers.append(nn.Dropout(dropout))

        # Subsequent LSTM layers
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(
                nn.LSTM(input_size=hidden_sizes[i-1], hidden_size=hidden_sizes[i], batch_first=True)
            )
            self.dropout_layers.append(nn.Dropout(dropout))

        # Final classification layer
        self.fc = nn.Linear(hidden_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)           # x: (batch, seq_len, hidden_size)
            x = dropout(x)

        last_out = x[:, -1, :]       # Take last time step output
        out = self.fc(last_out)
        return self.sigmoid(out).squeeze()
    
# --- Load trained model ---
model = LSTMModel(input_size=len(features), hidden_sizes=[100, 100], dropout=0.3).to(DEVICE)
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.eval()

# --- Inference and DataFrame creation ---
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
threshold = 0.5

results = []
all_true = []
all_pred = []
all_probs = []

with torch.no_grad():
    for X_batch, y_batch, meta_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        probs = model(X_batch)
        preds = (probs >= 0.5).int()

        for prob, pred, y_true, meta in zip(probs, preds, y_batch.numpy(), meta_batch):
            results.append({
                "Region": meta["Region"],
                "truckid": meta["truckid"],
                "Measurement_timestamp": meta["Measurement_timestamp"],
                "Maintenance_flag": int(y_true),
                "Prediction_Prob": float(prob),
                "Prediction_Label": int(pred)
            })
            all_true.append(int(y_true))
            all_pred.append(int(pred))
            all_probs.append(float(prob))

# --- Create result DataFrame ---
result_df = pd.DataFrame(results)
print(result_df.head())
result_df.to_csv("lstm_test_predictions.csv", index=False)

# --- Compute Metrics ---
conf_matrix = confusion_matrix(all_true, all_pred)
accuracy = accuracy_score(all_true, all_pred)
precision = precision_score(all_true, all_pred, zero_division=0)
recall = recall_score(all_true, all_pred, zero_division=0)
f1 = f1_score(all_true, all_pred, zero_division=0)
try:
    roc_auc = roc_auc_score(all_true, all_probs)
except:
    roc_auc = None

# --- Display Results ---
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
if roc_auc is not None:
    print(f"  ROC AUC:   {roc_auc:.4f}")
else:
    print("  ROC AUC:   Not computable (only one class in y_true)")

import seaborn as sns
import matplotlib.pyplot as plt

# --- Visualize Confusion Matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

# Save the figure
plt.savefig("lstm_confusion_matrix.png")
plt.show()
