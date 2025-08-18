# Fleet Maintenance Management (FMM) with LLM Advisor

This project is a **fleet maintenance management system** enhanced with **AI-powered advisory capabilities**.  
It integrates **LangChain**, **OpenAI/Ollama LLMs**, and AWS services to provide real-time explanations, actions, and urgency ratings based on telemetry and diagnostic data.

## 🚀 Features
- **LLM Advisor Node** (`app/agent/nodes_llm.py`)
  - Configurable to use **OpenAI** or **Ollama** via `.env`
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
- **Resilient API**
  - Retries on failures with exponential backoff
  - Fallback outputs if LLM call fails



## 📂 Project Structure
```

app/
├── agent/
│   ├── nodes\_llm.py         # LLM advisor node
│   └── ...
├── observability/
│   └── token\_callback.py    # Tracks OpenAI token usage
src/
└── app/static/
└── index.html           # Frontend entry point (moved to S3 in prod)

```



## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/degeaja/AgenticAI_FleetMaintenance.git
cd AgenticAI_FleetMaintenance
````

### 2. Create & Activate Virtual Environment

```bash
conda create -y -n ai_agent python=3.11
conda activate ai_agent
```


### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


## 🔑 Environment Variables

Create a `.env` file in the project root:

```ini
LOG_LEVEL=INFO
LOG_FILE=logs/fleet_agent.log

OPENAI_API_KEY= api_key 
PREDICT_URL=http://localhost:8000/predict
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
LLM_TIMEOUT_SECS=40
LLM_MAX_RETRIES=2
POLICY_HIGH_PROB=0.7
POLICY_MED_PROB=0.4

DATA_PATH="./src/app/ml/fleet_monitor_notscored_2.csv"

S3_ENDPOINT=s3_static_webpoint
```

## ☁️ AWS Setup

### 1. Configure AWS CLI (persistent credentials)

```bash
aws configure
```

Fill in:

```
AWS Access Key ID: your_access_key
AWS Secret Access Key: your_secret_key
Default region name: region
Default output format: json
```

Then,
```bash
aws configure set aws_session_token "your_token"
```

### 2. Upload Frontend to S3 (as Dev)

```bash
aws s3 cp ./src/app/static/index.html s3://your-bucket-name --acl public-read
```

> If ACLs are disabled, omit `--acl public-read` and manage permissions via **Bucket Policy**.

### 3. Enable Static Website Hosting in S3

Set **index document** to `index.html`.


## ▶️ Running the Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Backend endpoints:

* `POST /run-agent` → Runs the LLM advisor and returns `explanation`, `action`, `urgency`
* Other API routes for telemetry upload and region management


## 📡 Deployment

### EC2 Backend

1. SSH into EC2
2. Pull latest code
3. Install dependencies
4. Run with `uvicorn`

### S3 Frontend

* Upload latest `index.html` or build artifacts to the bucket
* Ensure bucket policy allows public read for web assets

## 🧠 How Urgency is Determined

1. **LLM Decision** – The LLM outputs urgency based on context.
2. **Policy Overlay** – Business rules **can raise or lower urgency** based on `failure_probability`.

   * `>= 0.6` → **High**
   * `>= 0.3` → **Medium**
   * `< 0.3` → **Low**
3. If LLM and probability disagree:

   * Urgency is **elevated** for high risk, user is notified through pop up message.
   * Urgency is **demoted** if probability is low


## 📊 Token Usage Tracking

OpenAI calls are wrapped with `TokenUsageHandler` to log:

* Prompt tokens
* Completion tokens
* Total tokens
* Cost estimates


