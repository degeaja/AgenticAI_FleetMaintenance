# app/observability/usage.py
from __future__ import annotations
from typing import Dict, Any
from contextvars import ContextVar
import os

# Per-run usage bucket keyed by run_id (from instrumentation.run_id_var)
_usage_ctx: ContextVar[Dict[str, Any]] = ContextVar("_usage_ctx", default=None)

def start_usage():
    _usage_ctx.set({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": None})

def add_usage(prompt: int = 0, completion: int = 0, total: int = 0, model: str | None = None):
    u = _usage_ctx.get() or {}
    u["prompt_tokens"] = (u.get("prompt_tokens", 0) + (prompt or 0))
    u["completion_tokens"] = (u.get("completion_tokens", 0) + (completion or 0))
    u["total_tokens"] = (u.get("total_tokens", 0) + (total or 0))
    if model:
        u["model"] = model
    _usage_ctx.set(u)

def get_usage() -> Dict[str, Any]:
    return _usage_ctx.get() or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": None}

def clear_usage():
    _usage_ctx.set(None)

# --- Pricing (configurable via env; defaults to 0 for safety) ---
# Prices are USD per 1K tokens. Set these in .env to your real model.
INPUT_PRICE_PER_1K = float(os.getenv("OPENAI_INPUT_PRICE_PER_1K", "0"))
OUTPUT_PRICE_PER_1K = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_1K", "0"))

def estimate_cost(usage: Dict[str, Any]) -> float:
    # simple linear estimate; adapt for per-model pricing if you use multiple models
    return (usage.get("prompt_tokens", 0) / 1000.0) * INPUT_PRICE_PER_1K + \
           (usage.get("completion_tokens", 0) / 1000.0) * OUTPUT_PRICE_PER_1K
