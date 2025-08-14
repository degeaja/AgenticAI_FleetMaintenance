# app/observability/token_callback.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from app.observability.usage import add_usage

class TokenUsageHandler(BaseCallbackHandler):
    """Aggregates token usage from LLM responses into our usage context."""
    def on_llm_end(self, response, *, run_id: str, parent_run_id: Optional[str] = None, **kwargs: Any) -> None:
        # response is LLMResult-like; .llm_output may contain token usage
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            # Standard place for LC v0.2 OpenAI:
            tu = (llm_output or {}).get("token_usage") or llm_output.get("usage") or {}
            prompt = int(tu.get("prompt_tokens", 0))
            completion = int(tu.get("completion_tokens", 0))
            total = int(tu.get("total_tokens", prompt + completion))
            model = (llm_output or {}).get("model_name") or (llm_output or {}).get("model")
            add_usage(prompt=prompt, completion=completion, total=total, model=model)
        except Exception:
            # be silent on parsing failures; usage stays zero
            pass
