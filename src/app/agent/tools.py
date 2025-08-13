import os, json, requests
from langchain.tools import tool
from langchain_community.vectorstores import Chroma

PREDICT_URL = os.getenv("PREDICT_URL", "http://localhost:8000/predict")
# Chroma persists by default to ./chroma; you can set persist_directory explicitly when you build the index
vs = Chroma(collection_name="fleet_kb")

@tool("predict_failure", return_direct=False)
def predict_failure(telemetry_json: str) -> str:
    """Input: telemetry JSON string. Output: model result JSON string."""
    data = json.loads(telemetry_json)
    r = requests.post(PREDICT_URL, json=data, timeout=10)
    r.raise_for_status()
    return r.text

@tool("kb_lookup", return_direct=False)
def kb_lookup(query: str) -> str:
    """Return top passages as JSON list: [{source,text}]."""
    docs = vs.similarity_search(query, k=4)
    return json.dumps([{"source": d.metadata.get("source"), "text": d.page_content[:900]} for d in docs])

@tool("schedule_service", return_direct=False)
def schedule_service(action_json: str) -> str:
    req = json.loads(action_json)
    # mock scheduling – replace with your real service manager API
    return json.dumps({"slot": "2025-08-14T02:00:00Z", "duration_min": 90, "bay": "B-03", "tech":"AUTOASSIGN"})

@tool("log_event", return_direct=False)
def log_event(event_json: str) -> str:
    return event_json

@tool("notify", return_direct=False)
def notify(message: str) -> str:
    return f"NOTIFIED: {message[:200]}"
