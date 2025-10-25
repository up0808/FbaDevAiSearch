from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI  # Use Gemini models
import os
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, HTTPException, Header, status
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
import asyncio
from functools import partial

# LangChain Google Search utility & Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool

load_dotenv()

# App version - update on new version
APP_VERSION = "1.0.0"
DEPLOYMENT_TIME = datetime.utcnow().isoformat()

# Adding Authorisation Check Point
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

async def verify_admin_api_key(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header."
        )

    token = authorization.split(" ")[1]
    if token != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key."
        )

# Initialize memory saver for checkpointing
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ------------------------------------------------------------------
# Google Search tool setup
# Requires these env vars in your .env:
# GOOGLE_API_KEY and GOOGLE_CSE_ID
# ------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    # Log a warning in runtime environments where envs might be set later
    print("Warning: GOOGLE_API_KEY and/or GOOGLE_CSE_ID are not set. Google Search tool will fail if invoked.")

# Instantiate the LangChain GoogleSearch wrapper
# NOTE: GoogleSearchAPIWrapper uses the environment variables by default, but we still keep them explicit.
google_search = GoogleSearchAPIWrapper(api_key=GOOGLE_API_KEY, cse_id=GOOGLE_CSE_ID)

# Helper to run blocking search in an async context
async def google_search_async(query: str, num_results: int = 4):
    # The wrapper often provides "run" or "results" methods depending on version. We'll try both defensively.
    loop = asyncio.get_running_loop()

    def _sync_call():
        # Prefer `results` if available because it gives structured results; otherwise fallback to `run`.
        if hasattr(google_search, "results"):
            try:
                # some versions accept `num_results` as param
                return google_search.results(query, num_results=num_results)
            except TypeError:
                return google_search.results(query)
        elif hasattr(google_search, "run"):
            try:
                return google_search.run(query)
            except Exception as e:
                # final fallback: return the exception string
                return str(e)
        else:
            return "GoogleSearchAPIWrapper has no run/results method available."

    result = await loop.run_in_executor(None, _sync_call)
    return result

# Wrap the async helper in a LangChain Tool. Tool can accept an async callable depending on LangChain version.
search_tool = Tool(
    name="google_search_results_json",
    description="Use Google Custom Search API to get relevant search results as JSON-like output.",
    func=lambda query: google_search_async(query, num_results=4)
)

# Attach tools list to be bound with the LLM
tools = [search_tool]

# ------------------------------------------------------------------
# LLM initialization (Gemini via langchain_google_genai)
# ------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools=tools)

# ------------------------------------------------------------------
# Graph nodes and tool handling
# ------------------------------------------------------------------
async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])  # streaming-capable invocation
    return {"messages": [result]}

async def tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

async def tool_node(state):
    """Custom tool node that handles tool calls from the LLM.

    This now expects calls named "google_search_results_json" and will run
    the Google Search tool defined above.
    """
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call.get("id")

        if tool_name == "google_search_results_json":
            # Expecting tool_args to contain a 'query' key (or be the raw string query)
            query = None
            if isinstance(tool_args, dict):
                # common format: {"query": "..."}
                query = tool_args.get("query") or next(iter(tool_args.values()), None)
            elif isinstance(tool_args, str):
                query = tool_args

            if not query:
                tool_message = ToolMessage(
                    content=str({"error": "no_query_provided"}),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_message)
                continue

            # Invoke the google search tool and capture output
            try:
                search_output = await google_search_async(query, num_results=4)
            except Exception as exc:
                search_output = {"error": str(exc)}

            # Normalize output to string for ToolMessage content
            tool_message = ToolMessage(
                content=str(search_output),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)

    return {"messages": tool_messages}

# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)

# ------------------------------------------------------------------
# FastAPI app (unchanged endpoints and behavior; just replaced Tavily with Google)
# ------------------------------------------------------------------
app = FastAPI(
    title="FBA Intelligent Search",
    version=APP_VERSION,
    description="AI-powered search engine with conversational capabilities"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)


def serialise_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )


async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None

    if is_new_conversation:
        new_checkpoint_id = str(uuid4())
        config = {"configurable": {"thread_id": new_checkpoint_id}}

        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {"configurable": {"thread_id": checkpoint_id}}
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

    async for event in events:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            if isinstance(chunk_content, list):
                chunk_content = " ".join(str(item) for item in chunk_content)
            safe_content = str(chunk_content).replace("'", "\\'").replace("\n", "\\n")
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"

        elif event_type == "on_chat_model_end":
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "google_search_results_json"]

            if search_calls:
                search_query = search_calls[0]["args"].get("query", "")
                safe_query = search_query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"

        elif event_type == "on_tool_end" and event["name"] == "google_search_results_json":
            output = event["data"]["output"]

            if isinstance(output, list):
                urls = []
                for item in output:
                    if isinstance(item, dict) and "link" in item:
                        # Google Search "results" format often has 'link' or 'url' depending on wrapper version
                        urls.append(item.get("link") or item.get("url"))
                    elif isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])

                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json}}}\n\n"

    yield f"data: {{\"type\": \"end\"}}\n\n"


# Public Endpoint
@app.get("/health")
async def health_check():
    """
    Public health check endpoint for Cloud Run.
    """
    return JSONResponse({
        "status": "healthy",
        "service": "FBA Dev AI Search Engine",
        "version": APP_VERSION,
        "deployed_at": DEPLOYMENT_TIME,
        "timestamp": datetime.utcnow().isoformat()
    })


# Authenticated Endpoints
@app.get("/")
async def root(_auth=Depends(verify_admin_api_key)):
    return JSONResponse({
        "message": "Welcome to FBA Dev AI Search Engine. YOU Successfully authenticated",
        "version": APP_VERSION,
        "endpoints": {
            "health": "/health",
            "chat": "/chat_stream/{message}",
            "docs": "/docs"
        }
    })


@app.get("/chat_stream/{message}")
async def chat_stream(
    message: str,
    checkpoint_id: Optional[str] = Query(None),
    _auth=Depends(verify_admin_api_key)
):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id),
        media_type="text/event-stream"
    )

