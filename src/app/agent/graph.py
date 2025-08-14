from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from .state import FleetState
from .nodes import validate_node, predict_node, retrieve_node, explain_node, policy_node, act_node

# Build graph in the canonical way (docs: Graph API & reference)
# add_node / add_edge / add_conditional_edges ; START/END for routing
builder = StateGraph(FleetState)

# Optional: conditional entry routing from START
def route_on_telemetry(state: FleetState) -> str:
    t = state.get("telemetry", {})
    # emergency guardrail
    if t.get("avg_coolant_temp_c", 0) >= 105 or t.get("oil_pressure_kpa", 999) <= 150:
        return "policy"   # jump straight to policy/act
    return "validate"

builder.add_node("validate", validate_node)
builder.add_node("predict",  predict_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("explain",  explain_node)
builder.add_node("policy",   policy_node)
builder.add_node("act",      act_node)

builder.add_conditional_edges(START, route_on_telemetry, {"validate":"validate", "policy":"policy"})
builder.add_edge("validate", "predict")
builder.add_edge("predict",  "retrieve")
builder.add_edge("retrieve", "explain")
builder.add_edge("explain",  "policy")
builder.add_edge("policy",   "act")
builder.add_edge("act",      END)

# Persistence (thread-scoped memory)
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
