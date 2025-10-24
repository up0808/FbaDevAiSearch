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

load_dotenv()

# Globals (lazy init)
graph_v2 = None
llm_v2 = None
search_tool_v2 = None
memory_v2 = MemorySaver()


class StateV2(TypedDict):
    messages: Annotated[list, add_messages]


def build_v2_graph(llm_with_tools, search_tool):
    async def model_node(state: StateV2):
        result = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [result]}

    async def tools_router(state: StateV2):
        last_message = state["messages"][-1]
        return "tool_node" if getattr(last_message, "tool_calls", []) else END

    async def tool_node(state: StateV2):
        messages = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] == "google_search":
                query = call["args"].get("query", "")
                try:
                    result = await search_tool.arun(query)
                except Exception as e:
                    result = f"Search error: {e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"], name="google_search"))
        return {"messages": messages}

    graph = StateGraph(StateV2)
    graph.add_node("model", model_node)
    graph.add_node("tool_node", tool_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", tools_router)
    graph.add_edge("tool_node", "model")
    return graph.compile(checkpointer=memory_v2)


def ensure_v2_initialized():
    """Initialize v2 components only once, when actually needed."""
    global llm_v2, search_tool_v2, graph_v2
    if graph_v2:
        return llm_v2, search_tool_v2, graph_v2

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not all([GOOGLE_API_KEY, GOOGLE_CSE_ID, GOOGLE_CLOUD_PROJECT]):
        raise RuntimeError("Missing required environment variables for v2.")

    google_search = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
        k=4,
    )

    search_tool_v2 = Tool(
        name="google_search",
        description="Search Google for up-to-date information.",
        func=google_search.run,
        coroutine=google_search.arun,
    )

    llm_v2 = ChatVertexAI(
        model_name="gemini-1.5-flash",
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        temperature=0.7,
        streaming=True,
    )

    llm_with_tools = llm_v2.bind_tools([search_tool_v2])
    graph_v2 = build_v2_graph(llm_with_tools, search_tool_v2)
    print("✅ Vertex AI v2 initialized successfully.")
    return llm_v2, search_tool_v2, graph_v2


def serialize_ai_message_chunk_v2(chunk):
    if not isinstance(chunk, AIMessageChunk):
        raise TypeError(f"Invalid chunk type: {type(chunk).__name__}")
    return chunk.content


async def generate_v2_chat_responses(message: str, checkpoint_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    try:
        _, _, graph = ensure_v2_initialized()
    except Exception as e:
        yield f'data: {{"type": "error", "message": "V2 init failed: {e}"}}\n\n'
        return

    is_new = checkpoint_id is None
    thread_id = str(uuid4()) if is_new else checkpoint_id
    config = {"configurable": {"thread_id": thread_id}}

    events = graph.astream_events(
        {"messages": [HumanMessage(content=message)]},
        version="v2",
        config=config,
    )

    if is_new:
        yield f'data: {{"type": "checkpoint", "checkpoint_id": "{thread_id}"}}\n\n'

    try:
        async for event in events:
            etype = event["event"]

            if etype == "on_chat_model_stream":
                chunk = serialize_ai_message_chunk_v2(event["data"]["chunk"])
                safe = str(chunk).replace('"', '\\"').replace("\n", "\\n")
                yield f'data: {{"type": "content", "content": "{safe}"}}\n\n'

            elif etype == "on_tool_end" and event["name"] == "google_search":
                output = str(event["data"]["output"]).replace('"', '\\"').replace("\n", "\\n")
                yield f'data: {{"type": "search_results", "results": "{output}"}}\n\n'

    except Exception as e:
        err = str(e).replace('"', '\\"').replace("\n", "\\n")
        yield f'data: {{"type": "error", "message": "{err}"}}\n\n'

    finally:
        yield 'data: {"type": "end"}\n\n'