# app/api/main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from app.observability.logging_setup import logger
from app.observability.instrumentation import start_run, end_run, run_id_var, truck_var, region_var
from app.agent.graph import agent   # import your compiled graph (or build_graph())
from dotenv import load_dotenv
import os

# Load .env into environment
load_dotenv()

# Debug: check if key loaded
print("Loaded API key:", os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Fleet Maintenance Agent")

# --------- Models ----------
class RunBody(BaseModel):
    fleetid: str
    truckid: str
    region: int
    new_reading: Dict[str, Any] | None = None  # optional; your get_new_sensor_reading can still simulate

# --------- Middleware to bind per-request context (optional but nice) ----------
@app.middleware("http")
async def bind_context(request: Request, call_next):
    try:
        # Try to read basics early for better log correlation
        if request.method == "POST" and request.url.path.endswith("/run"):
            body = await request.json()
            truckid = body.get("truckid", "-")
            region = body.get("region", "-")
        else:
            truckid, region = "-", "-"
        start_run(truckid=truckid, region=region)
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error("request_error", extra={"event": "request_error", "error": str(e)})
        raise
    finally:
        # Ensure a run_end even if endpoint forgets to call it
        rid = run_id_var.get()
        if rid:
            end_run(status="ok")

# --------- Endpoint ----------
@app.post("/run")
def run(b: RunBody):
    try:
        # In case middleware didn't set (e.g., different path), set here:
        if not run_id_var.get():
            start_run(truckid=b.truckid, region=b.region)

        # Build minimal state; your graph will synthesize new_reading if None
        state = {
            "fleetid": b.fleetid,
            "truckid": b.truckid,
            "region": b.region
        }
        if b.new_reading:
            state["new_reading"] = b.new_reading

        out = agent.invoke(state)
        end_run(status="ok", final_prob=out.get("prob"), maintenance=out.get("maintenance_needed"))
        return out
    except Exception as e:
        end_run(status="error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
