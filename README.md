# Fleet Maintenance Management (FMM) with LLM Advisor

This project is a **fleet maintenance management system** enhanced with **AI-powered advisory capabilities**.  
It integrates **LangGraph**, **OpenAI ChatGPT**, and AWS services to provide real-time explanations, actions, and urgency ratings based on telemetry and diagnostic data.

## 🚀 Features
- **LLM Advisor Node** (`app/agent/nodes_llm.py`)
  - Uses **OpenAI API**, with parameters configurable via `.env`
  - Structured JSON output with **Pydantic** (safe parsing)
  - Context-aware prompt including:
    - Telemetry data
    - Diagnostic Trouble Codes (DTC)
    - Failure probability
    - Maintenance flags
  - Business policy overlay to **raise or demote urgency** based on thresholds
  - Token usage tracking for OpenAI
- **AWS S3 Hosting**
  - Frontend assets served via S3 Static Website Hosting
  - API backend on EC2
- **Observability**
  - Tracks LLM token usage via `TokenUsageHandler`

## Installation

See [[src/README.MD]]

## 📂 Project Structure
```

app/
├── agent/
│   ├── nodes\_llm.py         # LLM advisor node
│   └── nodes\_ml.py          # ML prediction node
│   └── graph.py              # LangGraph Infrastructure
│   └── states.py             # state Class
├── api/
│   └── main.py               # everything needed for FastAPI
├── ml/
│   └── config.py             # config for ML Prediction
│   └── data_prep.py
│   └── features.py           # variable for features
│   └── lstm_arch.py
│   └── model_io.py           # streamlining model loading and scaling
│   └── best_model_08-18.pt   # saved model
│   └── scaler.save           # saved scaler
├── observability/
│   └── instrumentation.py    
│   └── logging_setup.py
│   └── token_callback.py     # OpenAI GPT token usage handler
│   └── usage.py
├── static/
│   └── index.html           # Frontend entry point (moved to S3 in prod)
├── logs/                    # logging needs
```

## 🧠 How Urgency is Determined

1. **LLM Decision** – The LLM outputs urgency based on context.
2. **Policy Overlay** – Business rules **can raise or lower urgency** based on `failure_probability`.

   * `>= 0.7` → **High**
   * `>= 0.3` → **Medium**
   * `< 0.3` → **Low**
3. If LLM and probability disagree:

   * Urgency is elevated for high risk, **user is notified through pop up message**.
   * Urgency is demoted if probability is low


## 📊 Token Usage Tracking

OpenAI calls are wrapped with `TokenUsageHandler` to log:

* Prompt tokens
* Completion tokens
* Total tokens
* Cost estimates

Which will be further saved into token.log.


