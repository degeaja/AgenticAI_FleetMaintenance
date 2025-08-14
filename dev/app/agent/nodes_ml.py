from app.config import DEVICE
from app.features import features
from app.ml.model_io import load_model_and_scaler
from app.ml.data_prep import prepare_sequence
import torch
from app.observability.instrumentation import instrument_node

MODEL, SCALER = load_model_and_scaler(len(features))

def get_new_sensor_reading(state):
    """
    Simulate/ingest new sensor reading and attach to state['new_reading'].
    """
    state["new_reading"] = {
        "Measurement_timestamp": "22FEB16:17:55:52",
        "Vehicle_speed_sensor": 26,
        "Vibration": 249.689536,
        "Engine_Load": 31.764706,
        "Engine_Coolant_Temp": 87,
        "Intake_Manifold_Pressure": 118,
        "Engine_RPM": 1129,
        "Speed_OBD": 26,
        "Intake_Air_Temp": 17,
        "Mass_Air_Flow_Rate": 19.02,
        "Throttle_Pos_Manifold": 45.09804,
        "Voltage_Control_Module": 14.28,
        "Ambient_air_temp": 6,
        "Accel_Pedal_Pos_D": 14.901961,
        "Engine_Oil_Temp": 83,
        "Speed_GPS": 27.578842,
        "Turbo_Boost_And_Vcm_Gauge": 8.550445794,
        "Trip_Distance": 49.66853177,
        "Litres_Per_100km_Inst": 21.3,
        "Accel_Ssor_Total": 156,
        "CO2_in_g_per_km_Inst": 3.045792,
        "Trip_Time_journey": 289.88788,
        "fleetid": state["fleetid"],
        "truckid": state["truckid"],
        "Region": state["region"]
    }
    return state

def prepare_data(state):
    """
    Load historical data, append reading, scale, and create X_seq tensor.
    """
    return prepare_sequence(state, features, SCALER)

def run_prediction(state):
    """
    Run the LSTM model to produce failure probability and threshold to maintenance_needed.
    """
    with torch.no_grad():
        prob = float(MODEL(state["X_seq"]).item())
    state["prob"] = prob
    state["maintenance_needed"] = prob >= 0.5
    return state

def store_new_reading(state):
    """
    Simulate persistence/logging of the new reading.
    """
    print(f"✅ Stored reading for truck {state['truckid']} in region {state['region']}")
    return state


get_new_sensor_reading = instrument_node("get_new_sensor_reading")(get_new_sensor_reading)
prepare_data          = instrument_node("prepare_data")(prepare_data)
run_prediction        = instrument_node("run_prediction")(run_prediction)
store_new_reading     = instrument_node("store_new_reading")(store_new_reading)
