import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
from dotenv import load_dotenv

# Load .env into environment
load_dotenv()

# Debug: check if key loaded
print("Loaded API key:", os.getenv("OPENAI_API_KEY"))


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key = os.getenv("OPENAI_API_KEY"))

def llm_explain(state):
    """
    Turn telemetry + prob into a short explanation/action/urgency JSON.
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
- 2 short sentences explanation (reference the most relevant signals).
- Action is one imperative sentence.
- Urgency: one of ["low","medium","high"].
Return ONLY a JSON object with keys: explanation, action, urgency.
"""
    msg = llm.invoke([HumanMessage(content=prompt)]).content
    try:
        j = json.loads(msg)
        state["explanation"] = j.get("explanation", "")
        state["action"] = j.get("action", "")
        state["urgency"] = j.get("urgency", "medium")
    except Exception:
        state["explanation"] = "Model indicates elevated risk given observed temperatures/pressures."
        state["action"] = "Schedule inspection within 48 hours and run full diagnostic."
        state["urgency"] = "medium"
    return state
