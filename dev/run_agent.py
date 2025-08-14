from app.agent.graph import build_graph

agent = build_graph()

state_in = {
    "fleetid": "Fleet_00513F1",
    "truckid": "Truck_0051X1",
    "region": 1
}
out = agent.invoke(state_in)
print("prob:", out.get("prob"))
print("needs_maint:", out.get("maintenance_needed"))
print("urgency:", out.get("urgency"))
print("action:", out.get("action"))
print("why:", out.get("explanation"))
