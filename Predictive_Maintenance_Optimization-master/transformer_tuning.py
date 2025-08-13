import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# =========================
# Load and preprocess data
# =========================
df = pd.read_csv("fleet_train.csv")
df["Measurement_timestamp"] = pd.to_datetime(df["Measurement_timestamp"], format="%d%b%y:%H:%M:%S")
df = df.sort_values(by=["Region", "truckid", "Measurement_timestamp"]).reset_index(drop=True)

features = [
    "Vehicle_speed_sensor", "Vibration", "Engine_Load", "Engine_Coolant_Temp", "Intake_Manifold_Pressure",
    "Engine_RPM", "Speed_OBD", "Intake_Air_Temp", "Mass_Air_Flow_Rate", "Throttle_Pos_Manifold",
    "Voltage_Control_Module", "Ambient_air_temp", "Accel_Pedal_Pos_D", "Engine_Oil_Temp", "Speed_GPS",
    "Turbo_Boost_And_Vcm_Gauge", "Trip_Distance", "Litres_Per_100km_Inst", "Accel_Ssor_Total",
    "CO2_in_g_per_km_Inst", "Trip_Time_journey"
]

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

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
X_train, X_val, y_train, y_val = train_test_split(X_seq_flat, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
X_train_resampled = X_train_resampled.reshape(-1, SEQUENCE_LENGTH, len(features))
X_val = X_val.reshape(-1, SEQUENCE_LENGTH, len(features))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model definition
# =========================
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
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
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]
        out = self.fc(x)
        return self.sigmoid(out).squeeze()

# Objective function for Optuna
def objective(trial):
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8, 16])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    # Data loaders
    train_dataset = TensorDataset(torch.tensor(X_train_resampled, dtype=torch.float32),
                                  torch.tensor(y_train_resampled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = TransformerModel(input_size=len(features), d_model=d_model, nhead=nhead,
                             num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    EPOCHS = 10
    for epoch in tqdm(range(EPOCHS), desc=f"Trial {trial.number} Training", leave=False):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            val_loss += criterion(model(X), y).item()
    return val_loss / len(val_loader)

# =========================
# Run Optuna with outer tqdm
# =========================
N_TRIALS = 100
with tqdm(total=N_TRIALS, desc="Hyperparameter Search") as pbar:
    def tqdm_callback(study, trial):
        pbar.update(1)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[tqdm_callback])

print("Best trial:", study.best_trial.params)
