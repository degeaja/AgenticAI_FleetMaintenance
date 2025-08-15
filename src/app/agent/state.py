from typing import TypedDict
import torch

class MaintenanceState(TypedDict):
    fleetid: str
    truckid: str
    region: str
    new_reading: dict
    X_seq: torch.Tensor
    prob: float
    maintenance_needed: bool
    explanation: str
    action: str
    urgency: str
