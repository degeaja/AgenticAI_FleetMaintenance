from langgraph.graph import StateGraph, END
from app.agent.state import MaintenanceState
from app.agent.nodes_ml import get_new_sensor_reading, prepare_data, run_prediction, store_new_reading
from app.agent.nodes_llm import llm_explain


def build_graph():
    graph = StateGraph(MaintenanceState)
    graph.add_node("get_new_sensor_reading", get_new_sensor_reading)
    graph.add_node("prepare_data", prepare_data)
    graph.add_node("run_prediction", run_prediction)
    graph.add_node("store_new_reading", store_new_reading)
    graph.add_node("llm_explain", llm_explain)

    graph.set_entry_point("get_new_sensor_reading")
    graph.add_edge("get_new_sensor_reading", "prepare_data")
    graph.add_edge("prepare_data", "run_prediction")
    graph.add_edge("run_prediction", "store_new_reading")
    graph.add_edge("store_new_reading", "llm_explain")
    graph.add_edge("llm_explain", END)
    return graph.compile()

agent = build_graph()