import os
from typing import TypedDict, Annotated, Optional, AsyncGenerator
from uuid import uuid4
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Memory for graph checkpointing
memory_v2 = MemorySaver()


class StateV2(TypedDict):
    messages: Annotated[list, add_messages]


def build_v2_graph(llm_with_tools, search_tool):
    """Constructs the state graph for model and tools routing."""

    async def model_node(state: StateV2) -> dict:
        result = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [result]}

    async def tools_router(state: StateV2) -> str:
        last_message = state["messages"][-1]
        return "tool_node" if getattr(last_message, "tool_calls", []) else END

    async def tool_node(state: StateV2) -> dict:
        tool_calls = state["messages"][-1].tool_calls
        tool_messages = []

        for call in tool_calls:
            if call["name"] == "google_search":
                query = call["args"].get("query", "")
                try:
                    search_results = await search_tool.arun(query)
                except Exception as e:
                    search_results = f"Search error: {e}"

                tool_messages.append(
                    ToolMessage(content=str(search_results), tool_call_id=call["id"], name=call["name"])
                )

        return {"messages": tool_messages}

    graph_builder = StateGraph(StateV2)
    graph_builder.add_node("model", model_node)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.set_entry_point("model")
    graph_builder.add_conditional_edges("model", tools_router)
    graph_builder.add_edge("tool_node", "model")

    return graph_builder.compile(checkpointer=memory_v2)


def initialize_v2_components():
    """Initializes Vertex AI + Google Search components."""

    required_vars = {
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "GOOGLE_CSE_ID": GOOGLE_CSE_ID,
        "GOOGLE_CLOUD_PROJECT": GOOGLE_CLOUD_PROJECT
    }

    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    google_search = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
        k=4
    )

    search_tool = Tool(
        name="google_search",
        description="Search Google for current information or news.",
        func=google_search.run,
        coroutine=google_search.arun
    )

    llm = ChatVertexAI(
        model_name="gemini-1.5-flash",
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        temperature=0.7,
        max_output_tokens=2048,
        streaming=True
    )

    llm_with_tools = llm.bind_tools([search_tool])
    graph = build_v2_graph(llm_with_tools, search_tool)

    return llm_with_tools, search_tool, graph


# Initialize globally
try:
    llm_v2, search_tool_v2, graph_v2 = initialize_v2_components()
    print("✅ V2 components initialized successfully")
except Exception as e:
    print(f"⚠️ Failed to initialize V2 components: {e}")
    llm_v2 = search_tool_v2 = graph_v2 = None


def serialize_ai_message_chunk_v2(chunk) -> str:
    if not isinstance(chunk, AIMessageChunk):
        raise TypeError(f"Invalid chunk type: {type(chunk).__name__}")
    return chunk.content


async def generate_v2_chat_responses(message: str, checkpoint_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Streams chat and search responses for Vertex AI."""

    if graph_v2 is None:
        yield 'data: {"type": "error", "message": "v2 not initialized. Check environment vars."}\n\n'
        return

    is_new = checkpoint_id is None
    checkpoint = str(uuid4()) if is_new else checkpoint_id
    config = {"configurable": {"thread_id": checkpoint}}

    events = graph_v2.astream_events(
        {"messages": [HumanMessage(content=message)]},
        version="v2",
        config=config
    )

    if is_new:
        yield f'data: {{"type": "checkpoint", "checkpoint_id": "{checkpoint}"}}\n\n'

    try:
        async for event in events:
            event_type = event["event"]

            if event_type == "on_chat_model_stream":
                chunk = serialize_ai_message_chunk_v2(event["data"]["chunk"])
                if isinstance(chunk, list):
                    chunk = " ".join(map(str, chunk))
                safe = str(chunk).replace('"', '\\"').replace("\n", "\\n")
                yield f'data: {{"type": "content", "content": "{safe}"}}\n\n'

            elif event_type == "on_chat_model_end":
                tool_calls = getattr(event["data"]["output"], "tool_calls", [])
                for call in tool_calls:
                    if call["name"] == "google_search":
                        query = call["args"].get("query", "")
                        safe_query = query.replace('"', '\\"').replace("\n", "\\n")
                        yield f'data: {{"type": "search_start", "query": "{safe_query}"}}\n\n'

            elif event_type == "on_tool_end" and event["name"] == "google_search":
                results = str(event["data"]["output"]).replace('"', '\\"').replace("\n", "\\n")
                yield f'data: {{"type": "search_results", "results": "{results}"}}\n\n'

    except Exception as e:
        msg = str(e).replace('"', '\\"').replace("\n", "\\n")
        yield f'data: {{"type": "error", "message": "{msg}"}}\n\n'

    finally:
        yield 'data: {"type": "end"}\n\n'


__all__ = [
    "generate_v2_chat_responses",
    "graph_v2",
    "memory_v2",
    "StateV2",
    "initialize_v2_components"
]