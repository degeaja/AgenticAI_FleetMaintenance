# app/api/main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from app.observability.logging_setup import logger
from app.observability.instrumentation import start_run, end_run, run_id_var, truck_var, region_var
from app.agent.graph import agent   # import your compiled graph (or build_graph())
from app.observability.usage import start_usage, get_usage, clear_usage, estimate_cost
import pandas as pd

from dotenv import load_dotenv
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load .env into environment
load_dotenv()

# Debug: check if key loaded
print("Loaded API key:", os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Fleet Maintenance Agent")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("app/static/index.html")


DATA_PATH = os.getenv("DATA_PATH", "app/fleet_monitor_notscored_2.csv")

# Load once (fast & simple). If file changes at runtime, reload logic can be added later.
try:
    _df_meta = pd.read_csv(DATA_PATH, dtype={"fleetid": "string", "truckid": "string"})
    # Normalize Region to int if possible; fall back to string.
    if "Region" in _df_meta.columns:
        try:
            _df_meta["Region"] = _df_meta["Region"].astype("Int64")
        except Exception:
            _df_meta["Region"] = _df_meta["Region"].astype("string")
    else:
        raise ValueError("CSV must contain a 'Region' column")
except Exception as e:
    raise RuntimeError(f"Failed to load {DATA_PATH}: {e}")

def _ensure_ok(df: pd.DataFrame):
    if not {"Region", "fleetid", "truckid"}.issubset(df.columns):
        raise RuntimeError("CSV must include columns: Region, fleetid, truckid")

_ensure_ok(_df_meta)

@app.get("/meta/regions")
def meta_regions():
    # Return sorted unique regions
    vals = _df_meta["Region"].dropna().unique().tolist()
    # Convert pandas Int64/NA types to plain python
    regions = [int(v) if isinstance(v, (int,)) or (hasattr(v, "item") and isinstance(v.item(), int)) else str(v) for v in vals]
    # Sort numbers numerically; strings lexicographically
    try:
        regions = sorted(regions, key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)))
    except Exception:
        regions = sorted(regions, key=lambda x: str(x))
    return {"regions": regions}

@app.get("/meta/fleets")
def meta_fleets(region: str | int):
    # filter by region
    df = _df_meta[_df_meta["Region"].astype(str) == str(region)]
    fleets = sorted(df["fleetid"].dropna().astype(str).unique().tolist())
    return {"fleets": fleets}

@app.get("/meta/trucks")
def meta_trucks(region: str | int, fleetid: str):
    df = _df_meta[
        (_df_meta["Region"].astype(str) == str(region)) &
        (_df_meta["fleetid"].astype(str) == str(fleetid))
    ]
    trucks = sorted(df["truckid"].dropna().astype(str).unique().tolist())
    return {"trucks": trucks}


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
        start_usage() 

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
@app.post("/run_untracked")
def run_untracked(b: RunBody):
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

@app.post("/run")
def run(b: RunBody):
    try:
        if not run_id_var.get():
            start_run(truckid=b.truckid, region=b.region)
            start_usage()  # ensure usage context exists

        state = {"fleetid": b.fleetid, "truckid": b.truckid, "region": b.region}
        if b.new_reading:
            state["new_reading"] = b.new_reading

        out = agent.invoke(state)

        # attach usage + estimated cost
        usage = get_usage()
        out["usage"] = {
            "model": usage.get("model"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "estimated_cost_usd": round(estimate_cost(usage), 6),
        }

        end_run(status="ok", final_prob=out.get("prob"), maintenance=out.get("maintenance_needed"))
        return out
    except Exception as e:
        end_run(status="error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        clear_usage()  # <--- important to reset between requests
