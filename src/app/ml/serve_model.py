# app/ml/serve_model.py
from fastapi import FastAPI
from pydantic import BaseModel
import math

app = FastAPI(title="Bare Maintenance Predictor (CPU)")

class Telemetry(BaseModel):
    vehicle_id: str
    timestamp: str
    odometer_km: float
    engine_hours: float
    avg_coolant_temp_c: float
    oil_pressure_kpa: float
    rpm: float
    speed_kmh: float
    dtc_codes: list[str] = []
    ambient_temp_c: float | None = None
    fuel_rate_lph: float | None = None
    tire_pressure_f_l: float | None = None
    tire_pressure_f_r: float | None = None
    tire_pressure_r_l: float | None = None
    tire_pressure_r_r: float | None = None

def clamp(x, lo=0.0, hi=1.0): 
    return max(lo, min(hi, x))

@app.post("/predict")
def predict(item: Telemetry):
    # Toy signal → probability
    coolant = item.avg_coolant_temp_c
    oilp    = item.oil_pressure_kpa
    dtc     = len(item.dtc_codes)

    # Normalize rough ranges
    coolant_term = (coolant - 85.0) / 25.0      # ~85C nominal
    oil_term     = (200.0 - oilp) / 200.0       # low oil pressure worse
    dtc_term     = min(dtc, 5) / 5.0

    raw = 0.55*coolant_term + 0.35*oil_term + 0.25*dtc_term
    p   = clamp(0.5*(1/(1+math.exp(-4*raw))))   # squashed to ~[0,0.5]
    p   = clamp(p + 0.15*dtc_term)              # bump if many DTCs

    bucket = "<7d" if p>=0.6 else "<30d" if p>=0.3 else ">30d"
    return {"vehicle_id": item.vehicle_id, "fail_prob_7d": float(p), "rul_bucket": bucket}
