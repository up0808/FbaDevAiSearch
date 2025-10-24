import json
import os
from datetime import datetime
from typing import TypedDict, Annotated, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Depends, HTTPException, Header, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Import the v2 chat generator function from v2.py
from v2 import generate_v2_chat_responses

# Load environment variables
load_dotenv()

APP_VERSION = "2.0.0"  # Updated to reflect V2 addition
DEPLOYMENT_TIME = datetime.utcnow().isoformat()

# API Keys
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# AUTHENTICATION

async def verify_admin_api_key(authorization: str = Header(None)):
    """
    Verify the admin API key from the Authorization header.
    """
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header. Use 'Bearer <token>'"
        )

    token = authorization.split(" ")[1]
    if token != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key."
        )

memory = MemorySaver()

class State(TypedDict):
    """State definition for V1 graph"""
    messages: Annotated[list, add_messages]

# Initialize V1 tools
search_tool = TavilySearchResults(max_results=4, tavily_api_key=TAVILY_API_KEY)
tools = [search_tool]

# Initialize V1 LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY)
llm_with_tools = llm.bind_tools(tools=tools)


async def model(state: State):
    """V1 Model node"""
    result = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [result]}


async def tools_router(state: State):
    """V1 Tools router"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    return END


async def tool_node(state):
    """V1 Tool execution node"""
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name == "tavily_search_results_json":
            search_results = await search_tool.ainvoke(tool_args)
            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)

    return {"messages": tool_messages}


# Build V1 graph
graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)


def serialise_ai_message_chunk(chunk):
    """Serialize AI message chunks for V1 streaming"""
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
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

        yield f'data: {{"type": "checkpoint", "checkpoint_id": "{new_checkpoint_id}"}}\n\n'
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
            yield f'data: {{"type": "content", "content": "{safe_content}"}}\n\n'

        elif event_type == "on_chat_model_end":
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "tavily_search_results_json"]

            if search_calls:
                search_query = search_calls[0]["args"].get("query", "")
                safe_query = search_query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                yield f'data: {{"type": "search_start", "query": "{safe_query}"}}\n\n'

        elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
            output = event["data"]["output"]

            if isinstance(output, list):
                urls = []
                for item in output:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])

                urls_json = json.dumps(urls)
                yield f'data: {{"type": "search_results", "urls": {urls_json}}}\n\n'

    yield 'data: {"type": "end"}\n\n'


app = FastAPI(
    title="FBA Dev AI Search Engine",
    version=APP_VERSION,
    description="AI-powered search engine with conversational capabilities (V1: Tavily + GenAI, V2: Google Search + Vertex AI)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

# PUBLIC ENDPOINTS (No Authentication)

@app.get("/health")
async def health_check():
    """
    Public health check endpoint for monitoring and load balancers.
    """
    return JSONResponse({
        "status": "healthy",
        "service": "FBA Dev AI Search Engine",
        "version": APP_VERSION,
        "deployed_at": DEPLOYMENT_TIME,
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "v1": "/chat_stream/{message}",
            "v2": "/v2/chat_stream/{message}",
            "versions": "/versions"
        }
    })

# AUTHENTICATED ENDPOINTS

@app.get("/")
async def root(_auth=Depends(verify_admin_api_key)):
    """
    Root endpoint - provides API information.
    Requires authentication.
    """
    return JSONResponse({
        "message": "Welcome to FBA Dev AI Search Engine. You successfully authenticated!",
        "version": APP_VERSION,
        "endpoints": {
            "health": "/health (public)",
            "v1_chat": "/chat_stream/{message} (authenticated)",
            "v2_chat": "/v2/chat_stream/{message} (authenticated)",
            "versions": "/versions (authenticated)",
            "docs": "/docs (public)"
        },
        "authentication": "Use 'Authorization: Bearer <API_KEY>' header"
    })


@app.get("/versions")
async def list_versions(_auth=Depends(verify_admin_api_key)):
    """
    List available API versions and their capabilities.
    Requires authentication.
    """
    return JSONResponse({
        "current_version": APP_VERSION,
        "versions": {
            "v1": {
                "endpoint": "/chat_stream/{message}",
                "model": "Google Gemini 2.0 Flash",
                "search": "Tavily Search API",
                "features": [
                    "streaming",
                    "checkpointing",
                    "tool_calling",
                    "curated_search_results"
                ],
                "best_for": "AI-focused search with curated sources"
            },
            "v2": {
                "endpoint": "/v2/chat_stream/{message}",
                "model": "Vertex AI Gemini 1.5 Flash",
                "search": "Google Custom Search API",
                "features": [
                    "streaming",
                    "checkpointing",
                    "tool_calling",
                    "web_scale_search",
                    "enterprise_ready",
                    "gcp_integration"
                ],
                "best_for": "Enterprise deployment with Google Cloud"
            }
        },
        "usage": {
            "new_conversation": "GET /{version}/chat_stream/{message}",
            "continue_conversation": "GET /{version}/chat_stream/{message}?checkpoint_id={id}",
            "authentication": "Include 'Authorization: Bearer <API_KEY>' header"
        }
    })

# V1 CHAT ENDPOINT

@app.get("/chat_stream/{message}")
async def chat_stream_v1(
    message: str,
    checkpoint_id: Optional[str] = Query(None, description="Checkpoint ID for conversation continuity"),
    _auth=Depends(verify_admin_api_key)
):
    """
    V1 Chat endpoint
    """
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id),
        media_type="text/event-stream"
    )


# V2 CHAT ENDPOINT

@app.get("/v2/chat_stream/{message}")
async def chat_stream_v2(
    message: str,
    checkpoint_id: Optional[str] = Query(None, description="Checkpoint ID for conversation continuity"),
    _auth=Depends(verify_admin_api_key)
):
    """
    V2 Chat endpoint using Google Search + Vertex AI.
    """
    return StreamingResponse(
        generate_v2_chat_responses(message, checkpoint_id),
        media_type="text/event-stream"
    )

# ERROR HANDLERS

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url)
        }
    )
