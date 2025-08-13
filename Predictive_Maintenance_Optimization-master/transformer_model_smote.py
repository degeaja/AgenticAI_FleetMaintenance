import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

# Sequence extraction
SEQUENCE_LENGTH = 20
X_seq, y_seq = [], []

for _, group in df.groupby(["Region", "truckid"]):
    if len(group) < SEQUENCE_LENGTH:
        continue
    X = group[features].values
    y = group["Maintenance_flag"].values
    for i in range(len(X) - SEQUENCE_LENGTH):
        X_seq.append(X[i:i+SEQUENCE_LENGTH])
        y_seq.append(y[i+SEQUENCE_LENGTH])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Flatten for SMOTE
X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

# Train-val split before applying SMOTE
X_train, X_val, y_train, y_val = train_test_split(X_seq_flat, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Reshape back to (batch, sequence, features)
X_train_resampled = X_train_resampled.reshape(-1, SEQUENCE_LENGTH, len(features))

# Convert to tensors and loaders
train_dataset = TensorDataset(torch.tensor(X_train_resampled, dtype=torch.float32),
                              torch.tensor(y_train_resampled, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val.reshape(-1, SEQUENCE_LENGTH, len(features)), dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.15694964958255903):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)  # project input features to d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)

        # Transformer expects (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)

        x = x[-1, :, :]  # take last time step's output (seq_len dimension is 0 now)

        out = self.fc(x)
        return self.sigmoid(out).squeeze()


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_size=len(features), d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.15694964958255903).to(device)
# model.load_state_dict(torch.load("transformer_best_model.pt", map_location=device))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001841060759284923)

# Early stopping
best_loss = np.inf
patience = 20
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

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += criterion(pred, y).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "transformer_best_model.pt")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break
