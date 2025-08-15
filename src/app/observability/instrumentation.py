# instrumentation.py
import time, uuid, functools, traceback
from typing import Callable, Any, Dict
from contextvars import ContextVar
from app.observability.logging_setup import logger

run_id_var: ContextVar[str] = ContextVar("run_id", default="")
truck_var: ContextVar[str] = ContextVar("truckid", default="")
region_var: ContextVar[str] = ContextVar("region", default="")

def start_run(truckid: str, region: Any) -> str:
    rid = uuid.uuid4().hex[:12]
    run_id_var.set(rid)
    truck_var.set(str(truckid))
    region_var.set(str(region))
    logger.info("run_start", extra={"event":"run_start","run_id":rid,"truckid":truckid,"region":region})
    return rid

def end_run(status: str = "ok", **kw):
    logger.info("run_end", extra={"event":"run_end","run_id":run_id_var.get(),"status":status, **kw})
    # reset (optional)
    run_id_var.set("")
    truck_var.set("")
    region_var.set("")

def instrument_node(node_name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            t0 = time.perf_counter()
            rid = run_id_var.get() or "-"
            truckid = state.get("truckid", truck_var.get() or "-")
            region = state.get("region", region_var.get() or "-")

            logger.info(
                "node_start",
                extra={"event":"node_start","run_id":rid,"node":node_name,"truckid":truckid,"region":region}
            )
            try:
                out = fn(state)
                dt = (time.perf_counter() - t0) * 1000
                # small, safe telemetry snapshot
                extras = {
                    "event":"node_end","run_id":rid,"node":node_name,"ms":round(dt,2),
                    "truckid":truckid,"region":region
                }
                if "prob" in out: extras["prob"] = float(out["prob"])
                if "maintenance_needed" in out: extras["maintenance_needed"] = bool(out["maintenance_needed"])
                logger.info("node_end", extra=extras)
                return out
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                logger.error(
                    "node_error",
                    extra={
                        "event":"node_error","run_id":rid,"node":node_name,"ms":round(dt,2),
                        "truckid":truckid,"region":region,"error":str(e),
                        "trace": traceback.format_exc(limit=3)
                    },
                )
                raise
        return wrapper
    return deco
