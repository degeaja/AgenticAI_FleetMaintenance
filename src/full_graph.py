# --- Add these imports near the top ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import os

# --- LLM setup (needs OPENAI_API_KEY in env) ---
# If you want to run locally via Ollama later, swap to ChatOllama.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# --- Extend state type (runtime is fine without total=False) ---
class MaintenanceState(TypedDict):
    fleetid: str
    truckid: str
    region: str
    new_reading: dict
    X_seq: torch.Tensor
    prob: float
    maintenance_needed: bool
    explanation: str        # NEW
    action: str             # NEW
    urgency: str            # NEW

# --- Add LLM node ---
def llm_explain(state: MaintenanceState):
    """
    Use the model's prob + the latest telemetry to produce a concise, actionable explanation.
    Responds in JSON: {explanation, action, urgency}.
    """
    reading = state.get("new_reading", {})
    prob = round(float(state.get("prob", 0.0)), 4)
    needed = bool(state.get("maintenance_needed", False))

    prompt = f"""
You are a senior fleet maintenance advisor. Given truck telemetry and a failure probability,
write a brief explanation and a specific next action.

Telemetry (JSON):
{json.dumps(reading, ensure_ascii=False)}

Model output:
- failure_probability_7d: {prob}
- maintenance_needed: {needed}

Rules:
- 2 short sentences for the explanation, reference the most relevant signals (e.g., coolant temp, oil pressure, DTCs if present).
- Action is a single imperative sentence (e.g., "Schedule cooling system inspection within 24h").
- Urgency: one of ["low","medium","high"].

Return ONLY a JSON object with keys: explanation, action, urgency.
"""

    msg = llm.invoke([HumanMessage(content=prompt)]).content
    # robust-ish JSON parse fallback
    try:
        j = json.loads(msg)
        state["explanation"] = j.get("explanation", "")
        state["action"] = j.get("action", "")
        state["urgency"] = j.get("urgency", "medium")
    except Exception:
        state["explanation"] = "Model indicates elevated risk given current temperatures/pressures."
        state["action"] = "Schedule inspection within 48 hours and run full diagnostic on cooling and lubrication systems."
        state["urgency"] = "medium"
    return state

# --- Wire node into the graph (add after your existing add_edge lines) ---
graph.add_node("llm_explain", llm_explain)
graph.add_edge("store_new_reading", "llm_explain")
graph.add_edge("llm_explain", END)

# Re-compile (if you compiled earlier)
agent = graph.compile()
