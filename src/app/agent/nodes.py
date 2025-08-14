import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from .state import FleetState
from .tools import predict_failure, kb_lookup, schedule_service, log_event, notify

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def validate_node(state: FleetState) -> FleetState:
    assert "vehicle_id" in state["telemetry"], "missing vehicle_id"
    return state

def predict_node(state: FleetState) -> FleetState:
    res = predict_failure.run(json.dumps(state["telemetry"]))
    state["model_result"] = json.loads(res)
    return state

def retrieve_node(state: FleetState) -> FleetState:
    t = state["telemetry"]
    q = f"DTC: {t.get('dtc_codes', [])}, coolant: {t.get('avg_coolant_temp_c')}, oil_pressure: {t.get('oil_pressure_kpa')}"
    kb = kb_lookup.run(q)
    state["kb_context"] = json.loads(kb)
    return state

def explain_node(state: FleetState) -> FleetState:
    ctx = "\n\n".join([f"[{d['source']}] {d['text']}" for d in (state.get("kb_context") or [])])
    prompt = f"""You are a fleet maintenance advisor.
                Telemetry: {json.dumps(state['telemetry'])}
                Prediction: {json.dumps(state['model_result'])}
                KB:\n{ctx}\n
                Explain root cause and suggest a job type. 3 sentences max.
                Then output ONLY a JSON object with: decision (schedule|monitor), urgency (low|medium|high), job_type."""
    msg = llm.invoke([HumanMessage(content=prompt)]).content
    
    state["reasoning"] = msg
    # naive JSON extraction; harden in prod
    json_part = msg[msg.find("{"): msg.rfind("}")+1]
    try:
        state["decision"] = json.loads(json_part)
    except Exception:
        state["decision"] = {"decision":"monitor","urgency":"low","job_type":"General inspection"}
    return state

def policy_node(state: FleetState) -> FleetState:
    p = float(state["model_result"].get("fail_prob_7d", 0.0))
    d = state["decision"] or {}
    if p >= 0.6: d.update({"decision":"schedule","urgency":"high"})
    elif p >= 0.3: d.setdefault("urgency","medium")
    else: d.update({"decision":"monitor","urgency":"low"})
    state["decision"] = d
    return state

def act_node(state: FleetState) -> FleetState:
    if state["decision"]["decision"] == "schedule":
        act = schedule_service.run(json.dumps({
            "vehicle_id": state["telemetry"]["vehicle_id"],
            "job_type": state["decision"].get("job_type","General inspection"),
            "urgency": state["decision"]["urgency"]
        }))
        state["decision"]["slot"] = (json.loads(act))["slot"]
    notify.run(f"{state['telemetry']['vehicle_id']}: {state['decision']}")
    log_event.run(json.dumps({
        "vehicle_id": state["telemetry"]["vehicle_id"],
        "pred": state["model_result"],
        "decision": state["decision"],
        "reasoning": state["reasoning"]
    }))
    return state
