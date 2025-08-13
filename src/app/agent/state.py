from typing import TypedDict, Optional, Dict, Any, List

class FleetState(TypedDict, total=False):
    telemetry: Dict[str, Any]
    model_result: Optional[Dict[str, Any]]
    kb_context: Optional[List[Dict[str, Any]]]
    reasoning: Optional[str]
    decision: Optional[Dict[str, Any]]
    logs: List[Dict[str, Any]]
