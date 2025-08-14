from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Telemetry(BaseModel):
    vehicle_id: str
    timestamp: str
    odometer_km: float
    engine_hours: float
    avg_coolant_temp_c: float
    oil_pressure_kpa: float
    rpm: float
    speed_kmh: float
    dtc_codes: List[str] = []
    ambient_temp_c: Optional[float] = None
    fuel_rate_lph: Optional[float] = None
    tire_pressure_f_l: Optional[float] = None
    tire_pressure_f_r: Optional[float] = None
    tire_pressure_r_l: Optional[float] = None
    tire_pressure_r_r: Optional[float] = None

class RunRequest(BaseModel):
    telemetry: Telemetry
    thread_id: Optional[str] = Field(default=None, description="Thread for LangGraph memory")
    human_approval: Optional[bool] = False  # demo of HiTL

class StreamRequest(RunRequest):
    pass

class AgentResponse(BaseModel):
    decision: Dict[str, Any]
    model_result: Dict[str, Any]
    reasoning: str
