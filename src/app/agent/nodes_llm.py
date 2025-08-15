"""
app/agent/nodes_llm.py
"Fluid & reactive" LLM advisor node:
- configurable model/provider via ENV (OpenAI or Ollama)
- structured JSON output with Pydantic (no brittle json.loads)
- context-aware prompt (uses telemetry, DTCs, prob, maintenance flag)
- small business policy overlay (auto-raise urgency on high prob)
- graceful fallbacks & retries
- no secrets printed
"""

from __future__ import annotations
import os
import time
from typing import Literal, Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage

from app.observability.token_callback import TokenUsageHandler



# Provider-agnostic factory
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None  

# ------------------------------------------------------------------
# ENV / Config
# ------------------------------------------------------------------
load_dotenv()  # load once at import; no printing secrets

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()     # "openai" | "ollama"
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini")           # or ollama model
LLM_TEMP     = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT  = float(os.getenv("LLM_TIMEOUT_SECS", "30"))
MAX_RETRIES  = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Policy thresholds (tweak without code changes)
HIGH_PROB_URGENCY  = float(os.getenv("POLICY_HIGH_PROB", "0.6"))
MED_PROB_URGENCY   = float(os.getenv("POLICY_MED_PROB",  "0.3"))

# ------------------------------------------------------------------
# Structured output schema
# ------------------------------------------------------------------
class Advisory(BaseModel):
    explanation: str = Field(..., description="2 concise sentences in plain English, referencing key signals.")
    action: str      = Field(..., description="One imperative sentence with a clear next step.")
    urgency: Literal["low", "medium", "high"]

# ------------------------------------------------------------------
# LLM factory (OpenAI or Ollama)
# ------------------------------------------------------------------
def _build_llm():
    if LLM_PROVIDER == "ollama":
        if ChatOllama is None:
            raise RuntimeError("langchain-ollama not installed; pip install langchain-ollama")
        return ChatOllama(model=LLM_MODEL, temperature=LLM_TEMP, request_timeout=LLM_TIMEOUT)
    # default: openai
    if ChatOpenAI is None:
        raise RuntimeError("langchain-openai not installed; pip install langchain-openai")
    # api_key will be read from OPENAI_API_KEY in env automatically (or you can pass api_key=os.getenv("OPENAI_API_KEY"))
    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMP, timeout=LLM_TIMEOUT)

# Make a single module-level client (cheap to reuse)
_llm = _build_llm()
_llm_structured = _llm.with_structured_output(Advisory)

# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------
def _format_signals(reading: Dict[str, Any]) -> str:
    if not reading:
        return "No telemetry provided."
    # Highlight a few likely-interesting fields if present
    keys_of_interest = [
        "Engine_Coolant_Temp","Engine_Oil_Temp","Oil_Pressure","oil_pressure_kpa",
        "Mass_Air_Flow_Rate","Intake_Air_Temp","Engine_RPM","Vehicle_speed_sensor",
        "dtc_codes","DTC","Vibration","Throttle_Pos_Manifold","Voltage_Control_Module",
        "Ambient_air_temp","Litres_Per_100km_Inst"
    ]
    parts = []
    for k in keys_of_interest:
        if k in reading:
            parts.append(f"{k}={reading[k]}")
    # include short tail if empty
    if not parts:
        # fall back to top ~10 numeric-ish items to keep prompt short
        c = 0
        for k, v in reading.items():
            if isinstance(v, (int, float, str)) and c < 10:
                parts.append(f"{k}={v}")
                c += 1
    return ", ".join(map(str, parts))[:800]  # keep prompt small

def _build_messages(state: Dict[str, Any]) -> list:
    reading = state.get("new_reading", {}) or {}
    prob    = float(state.get("prob", 0.0))
    needed  = bool(state.get("maintenance_needed", False))

    # if dtc list exists under different keys, normalize for the LLM
    dtcs = reading.get("dtc_codes") or reading.get("DTC") or reading.get("dtc") or []
    if isinstance(dtcs, str):
        dtcs = [x.strip() for x in dtcs.split(",") if x.strip()]

    # a short, structured context to keep outputs stable
    sys = SystemMessage(content=(
        "You are a senior fleet maintenance advisor. "
        "You produce concise, practical guidance for technicians. "
        "Only factual, non-speculative statements. Keep explanations to 2 sentences."
    ))

    human = HumanMessage(content=(
        "Context:\n"
        f"- Failure probability (7d): {prob:.4f}\n"
        f"- Maintenance needed: {needed}\n"
        f"- DTC codes: {dtcs if dtcs else 'None'}\n"
        f"- Key signals: { _format_signals(reading) }\n\n"
        "Task:\n"
        "Produce a short, human-friendly summary for the driver/dispatcher and a concrete next action.\n"
        "Rules:\n"
        '1) Explanation: 2 short sentences, reference relevant signals.\n'
        '2) Action: 1 imperative sentence (what to do next, include timeframe if relevant).\n'
        '3) Urgency: one of ["low","medium","high"].\n'
        "Return a JSON object with exactly these keys: explanation, action, urgency."
    ))
    return [sys, human]

# ------------------------------------------------------------------
# Policy overlay (keeps outputs consistent with risk)
# ------------------------------------------------------------------
def _apply_policy(advice: Advisory, prob: float) -> Advisory:
    """
    Enforce urgency bins from probability:
      prob < MED_PROB_URGENCY        -> low
      MED_PROB_URGENCY <= prob < HIGH_PROB_URGENCY -> medium
      prob >= HIGH_PROB_URGENCY      -> high  (+ ensure 24h timeframe in action)
    This BOTH promotes or demotes the LLM's suggestion to match policy.
    """
    if prob >= HIGH_PROB_URGENCY:
        enforced = "high"
    elif prob >= MED_PROB_URGENCY:
        enforced = "medium"
    else:
        enforced = "low"

    advice.urgency = enforced

    # Ensure action carries time expectation when high urgency
    if advice.urgency == "high":
        a = advice.action or ""
        if "hour" not in a.lower() and "today" not in a.lower():
            advice.action = a.rstrip(".") + " within 24 hours."
    return advice

# ------------------------------------------------------------------
# Resilient call (few retries, structured parsing, safe fallback)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Node entry point
# ------------------------------------------------------------------
def llm_explain(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: enrich state with LLM-generated 'explanation', 'action', 'urgency'.
    - Uses provider/model/temperature from ENV
    - Structured output via Pydantic (no brittle JSON parsing)
    - Deterministic urgency overlay from policy thresholds
    """
    prob = float(state.get("prob", 0.0))
    msgs = _build_messages(state)
    advice = _advise_with_retries(msgs, prob)

    state["explanation"] = advice.explanation
    state["action"] = advice.action
    state["urgency"] = advice.urgency
    return state

# ...
def _advise_with_retries(messages: list, prob: float) -> Advisory:
    last_err = None
    cb = TokenUsageHandler()
    for i in range(MAX_RETRIES + 1):
        try:
            advice = _llm_structured.invoke(messages, config={"callbacks": [cb]})
            return _apply_policy(advice, prob)   # <-- enforces bins (demotes/promotes)
        except ValidationError as ve:
            last_err = ve
        except Exception as e:
            last_err = e
        time.sleep(min(0.4 * (2 ** i), 2.0))
    # fallback honors same bins
    urgency = "high" if prob >= HIGH_PROB_URGENCY else "medium" if prob >= MED_PROB_URGENCY else "low"
    return Advisory(
        explanation="Model indicates elevated risk given current operating signals. Prioritize system checks accordingly.",
        action="Schedule an inspection and run full diagnostics as soon as feasible.",
        urgency=urgency,
    )
