import pandas as pd
import torch
from app.config import CSV_PATH, SEQUENCE_LENGTH, DEVICE

def prepare_sequence(state, features, scaler):
    df = pd.read_csv(CSV_PATH)
    df["Measurement_timestamp"] = pd.to_datetime(df["Measurement_timestamp"], format="%d%b%y:%H:%M:%S")

    df = df[
        (df["fleetid"] == state["fleetid"]) &
        (df["Region"] == state["region"]) &
        (df["truckid"] == state["truckid"])
    ]

    new_df = pd.DataFrame([state["new_reading"]])
    new_df["Measurement_timestamp"] = pd.to_datetime(new_df["Measurement_timestamp"], format="%d%b%y:%H:%M:%S")
    df = pd.concat([df, new_df], ignore_index=True)
    df = df.sort_values(by="Measurement_timestamp")

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough data for truck {state['truckid']} in {state['region']}.")

    df[features] = scaler.transform(df[features])
    seq_data = df[features].tail(SEQUENCE_LENGTH).values
    state["X_seq"] = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return state
