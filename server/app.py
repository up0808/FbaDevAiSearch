import os
import json
from uuid import uuid4
from datetime import datetime
from typing import TypedDict, Annotated, Optional

from fastapi import FastAPI, Query, Depends, HTTPException, Header, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langgraph.graph import add_messages, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

#from v2 import generate_v2_chat_responses

# Load environment
load_dotenv()

APP_VERSION = "2.0.0"
DEPLOYMENT_TIME = datetime.utcnow().isoformat()

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Auth verification
async def verify_admin_api_key(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    if authorization.split(" ")[1] != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

# Setup for v1
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results=4, tavily_api_key=TAVILY_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools([search_tool])

async def model(state: State):
    return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}

async def tools_router(state: State):
    last_message = state["messages"][-1]
    return "tool_node" if getattr(last_message, "tool_calls", []) else END

async def tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    msgs = []
    for call in tool_calls:
        if call["name"] == "tavily_search_results_json":
            results = await search_tool.ainvoke(call["args"])
            msgs.append(ToolMessage(content=str(results), tool_call_id=call["id"], name=call["name"]))
    return {"messages": msgs}

graph = (
    StateGraph(State)
    .add_node("model", model)
    .add_node("tool_node", tool_node)
    .set_entry_point("model")
    .add_conditional_edges("model", tools_router)
    .add_edge("tool_node", "model")
    .compile(checkpointer=memory)
)

app = FastAPI(
    title="FBA Dev AI Search Engine",
    version=APP_VERSION,
    description="AI-powered search engine (v1 Tavily + v2 Vertex AI)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def serialise_chunk(chunk):
    if not isinstance(chunk, AIMessageChunk):
        raise TypeError(f"Invalid chunk type: {type(chunk).__name__}")
    return chunk.content

async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    new = checkpoint_id is None
    checkpoint = str(uuid4()) if new else checkpoint_id
    config = {"configurable": {"thread_id": checkpoint}}

    events = graph.astream_events({"messages": [HumanMessage(content=message)]}, version="v2", config=config)
    if new:
        yield f'data: {{"type": "checkpoint", "checkpoint_id": "{checkpoint}"}}\n\n'

    async for event in events:
        etype = event["event"]

        if etype == "on_chat_model_stream":
            chunk = serialise_chunk(event["data"]["chunk"])
            content = " ".join(map(str, chunk)) if isinstance(chunk, list) else chunk
            safe = str(content).replace("'", "\\'").replace("\n", "\\n")
            yield f'data: {{"type": "content", "content": "{safe}"}}\n\n'

        elif etype == "on_chat_model_end":
            tool_calls = getattr(event["data"]["output"], "tool_calls", [])
            for call in tool_calls:
                if call["name"] == "tavily_search_results_json":
                    query = call["args"].get("query", "")
                    safe_query = query.replace('"', '\\"').replace("\n", "\\n")
                    yield f'data: {{"type": "search_start", "query": "{safe_query}"}}\n\n'

        elif etype == "on_tool_end" and event["name"] == "tavily_search_results_json":
            output = event["data"]["output"]
            urls = [item["url"] for item in output if isinstance(item, dict) and "url" in item]
            yield f'data: {{"type": "search_results", "urls": {json.dumps(urls)}}}\n\n'

    yield 'data: {"type": "end"}\n\n'

# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": APP_VERSION, "deployed_at": DEPLOYMENT_TIME}

@app.get("/", dependencies=[Depends(verify_admin_api_key)])
async def root():
    return {"message": "Authenticated", "version": APP_VERSION}
'''
@app.get("/chat_stream/{message}", dependencies=[Depends(verify_admin_api_key)])
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(generate_chat_responses(message, checkpoint_id), media_type="text/event-stream")

@app.get("/v2/chat_stream/{message}", dependencies=[Depends(verify_admin_api_key)])
async def chat_stream_v2(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(generate_v2_chat_responses(message, checkpoint_id), media_type="text/event-stream")
'''
@app.get("/versions", dependencies=[Depends(verify_admin_api_key)])
async def versions():
    return {
        "versions": {
            "v1": {"model": "Gemini 2.5 Flash", "search": "Tavily"},
            "v2": {"model": "Vertex AI Gemini 1.5 Flash", "search": "Google Search API"}
        },
        "current": APP_VERSION
    }