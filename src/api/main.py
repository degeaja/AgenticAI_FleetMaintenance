import os, json, asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from app.agent.graph import graph
from app.agent.schemas import RunRequest, StreamRequest, AgentResponse

load_dotenv()
app = FastAPI(title="Fleet Maintenance Agent (LangGraph)")

def config_for(thread_id: str | None):
    # Per LangGraph docs: pass thread_id in config.configurable for persistence (memory)
    # https://langchain-ai.github.io/langgraph/concepts/persistence/
    cfg = {}
    if thread_id:
        cfg = {"configurable": {"thread_id": thread_id}}
    return cfg

@app.post("/run", response_model=AgentResponse)
def run(req: RunRequest):
    state_in = {
        "telemetry": req.telemetry.model_dump(),
        "logs": []
    }
    try:
        out = graph.invoke(state_in, config=config_for(req.thread_id))
    except Exception as e:
        raise HTTPException(400, str(e))
    return AgentResponse(
        decision=out.get("decision", {}),
        model_result=out.get("model_result", {}),
        reasoning=out.get("reasoning", "")
    )

@app.post("/stream")
async def stream(req: StreamRequest):
    state_in = {"telemetry": req.telemetry.model_dump(), "logs": []}
    async def eventgen():
        try:
            async for chunk in graph.astream(state_in, config=config_for(req.thread_id)):
                # chunk is a dict keyed by node name; yield SSE-ish lines
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
    return StreamingResponse(eventgen(), media_type="text/event-stream")
