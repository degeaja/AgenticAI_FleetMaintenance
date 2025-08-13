import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import joblib

# Load and preprocess
df = pd.read_csv("fleet_train.csv")
df["Measurement_timestamp"] = pd.to_datetime(df["Measurement_timestamp"], format="%d%b%y:%H:%M:%S")
df = df.sort_values(by=["Region", "truckid", "Measurement_timestamp"]).reset_index(drop=True)

# Sensor features
features = [
    "Vehicle_speed_sensor", "Vibration", "Engine_Load", "Engine_Coolant_Temp", "Intake_Manifold_Pressure",
    "Engine_RPM", "Speed_OBD", "Intake_Air_Temp", "Mass_Air_Flow_Rate", "Throttle_Pos_Manifold",
    "Voltage_Control_Module", "Ambient_air_temp", "Accel_Pedal_Pos_D", "Engine_Oil_Temp", "Speed_GPS",
    "Turbo_Boost_And_Vcm_Gauge", "Trip_Distance", "Litres_Per_100km_Inst", "Accel_Ssor_Total",
    "CO2_in_g_per_km_Inst", "Trip_Time_journey"
]

# Normalize
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
joblib.dump(scaler, "scaler.save")

# Sequence dataset
SEQUENCE_LENGTH = 20

class FleetDataset(Dataset):
    def __init__(self, df, features, seq_len):
        self.sequences = []
        self.labels = []
        for _, group in df.groupby(["Region", "truckid"]):
            if len(group) < seq_len:
                continue
            X = group[features].values
            y = group["Maintenance_flag"].values
            for i in range(len(X) - seq_len):
                self.sequences.append(X[i:i+seq_len])
                self.labels.append(y[i+seq_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Train/Val split
full_dataset = FleetDataset(df, features, SEQUENCE_LENGTH)
total_len = len(full_dataset)
train_len = int(0.8 * total_len)
val_len = total_len - train_len
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=len(features), hidden_sizes=[100, 50], dropout=0.2).to(device)
# model.load_state_dict(torch.load("best_model.pt", map_location=device))
criterion = nn.MSELoss()
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.1], device=device))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Early stopping parameters
best_loss = np.inf
patience = 10
trigger_times = 0

EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        loop.set_postfix(train_loss=loss.item())

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += criterion(pred, y).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")

    # Early stopping check
    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break
