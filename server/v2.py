"""
v2.py - AI Search API v2 with Vertex AI and Google Search

This module provides version 2 of the AI search API using:
- ChatVertexAI for conversational AI (Google Cloud Vertex AI)
- GoogleSearchAPIWrapper for web search capabilities
- LangGraph for agentic workflow orchestration

The code follows proper Python function definition order:
helper functions are defined before they are called.
"""

import json
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

# Environment configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Initialize memory for conversation checkpointing
memory_v2 = MemorySaver()


class StateV2(TypedDict):
    
    messages: Annotated[list, add_messages]


def build_v2_graph(llm_with_tools, search_tool):
    
    async def model_node(state: StateV2) -> dict:
        
        result = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [result]}
    
    async def tools_router(state: StateV2) -> str:
        
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "tool_node"
        return END
    
    async def tool_node(state: StateV2) -> dict:
       
        tool_calls = state["messages"][-1].tool_calls
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if tool_name == "google_search":
                # Execute Google Search
                query = tool_args.get("query", "")
                try:
                    search_results = await search_tool.arun(query)
                except Exception as e:
                    search_results = f"Search error: {str(e)}"
                
                tool_message = ToolMessage(
                    content=str(search_results),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(tool_message)
        
        return {"messages": tool_messages}
    
    # Build the graph
    graph_builder = StateGraph(StateV2)
    graph_builder.add_node("model", model_node)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.set_entry_point("model")
    graph_builder.add_conditional_edges("model", tools_router)
    graph_builder.add_edge("tool_node", "model")
    
    return graph_builder.compile(checkpointer=memory_v2)


def initialize_v2_components():
    
    # Validate required environment variables
    required_vars = {
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "GOOGLE_CSE_ID": GOOGLE_CSE_ID,
        "GOOGLE_CLOUD_PROJECT": GOOGLE_CLOUD_PROJECT
    }
    
    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    # Initialize Google Search API Wrapper
    google_search = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
        k=4  # Number of results to return
    )
    
    # Create a LangChain Tool from Google Search
    search_tool = Tool(
        name="google_search",
        description=(
            "Search Google for recent results. Use this when you need current "
            "information, facts, news, or answers about any topic."
        ),
        func=google_search.run,
        coroutine=google_search.arun  # Async version
    )
    
    # Initialize Vertex AI Chat Model
    llm = ChatVertexAI(
        model_name="gemini-1.5-flash",  # or "gemini-1.5-pro" for more capability
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        temperature=0.7,
        max_output_tokens=2048,
        streaming=True
    )
    
    # Bind tools to the model
    tools = [search_tool]
    llm_with_tools = llm.bind_tools(tools=tools)
    
    # Build the graph (now this is safe because build_v2_graph is defined above)
    graph = build_v2_graph(llm_with_tools, search_tool)
    
    return llm_with_tools, search_tool, graph


# Initialize v2 components globally with proper error handling
try:
    llm_v2, search_tool_v2, graph_v2 = initialize_v2_components()
    print("✅ V2 components initialized successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not initialize v2 components: {e}")
    print("   Make sure GOOGLE_API_KEY, GOOGLE_CSE_ID, and GOOGLE_CLOUD_PROJECT are set")
    llm_v2, search_tool_v2, graph_v2 = None, None, None


def serialize_ai_message_chunk_v2(chunk) -> str:
    
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    raise TypeError(
        f"Object of type {type(chunk).__name__} is not correctly formatted "
        "for serialization"
    )


async def generate_v2_chat_responses(
    message: str,
    checkpoint_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    
    # Check if components are initialized
    if graph_v2 is None:
        yield (
            'data: {"type": "error", '
            '"message": "v2 components not initialized. '
            'Check environment variables."}\n\n'
        )
        return
    
    is_new_conversation = checkpoint_id is None
    
    # Set up configuration for LangGraph
    if is_new_conversation:
        new_checkpoint_id = str(uuid4())
        config = {"configurable": {"thread_id": new_checkpoint_id}}
        
        events = graph_v2.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
        
        # Send checkpoint ID to client
        yield f'data: {{"type": "checkpoint", "checkpoint_id": "{new_checkpoint_id}"}}\n\n'
    else:
        config = {"configurable": {"thread_id": checkpoint_id}}
        events = graph_v2.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
    
    # Stream events from the graph
    try:
        async for event in events:
            event_type = event["event"]
            
            # Stream content chunks
            if event_type == "on_chat_model_stream":
                chunk_content = serialize_ai_message_chunk_v2(event["data"]["chunk"])
                
                # Handle list content
                if isinstance(chunk_content, list):
                    chunk_content = " ".join(str(item) for item in chunk_content)
                
                # Escape content for JSON
                safe_content = (
                    str(chunk_content)
                    .replace('"', '\\"')
                    .replace("'", "\\'")
                    .replace("\n", "\\n")
                )
                yield f'data: {{"type": "content", "content": "{safe_content}"}}\n\n'
            
            # Detect search intent
            elif event_type == "on_chat_model_end":
                output = event["data"]["output"]
                tool_calls = getattr(output, "tool_calls", [])
                search_calls = [
                    call for call in tool_calls
                    if call["name"] == "google_search"
                ]
                
                if search_calls:
                    search_query = search_calls[0]["args"].get("query", "")
                    safe_query = (
                        search_query
                        .replace('"', '\\"')
                        .replace("'", "\\'")
                        .replace("\n", "\\n")
                    )
                    yield f'data: {{"type": "search_start", "query": "{safe_query}"}}\n\n'
            
            # Return search results
            elif event_type == "on_tool_end" and event["name"] == "google_search":
                output = event["data"]["output"]
                
                # Google Search returns a formatted string with snippets
                safe_output = str(output).replace('"', '\\"').replace("\n", "\\n")
                yield f'data: {{"type": "search_results", "results": "{safe_output}"}}\n\n'
    
    except Exception as e:
        # Handle streaming errors gracefully
        error_msg = str(e).replace('"', '\\"').replace("\n", "\\n")
        yield f'data: {{"type": "error", "message": "{error_msg}"}}\n\n'
    
    finally:
        # Always send end event
        yield 'data: {"type": "end"}\n\n'


# Export the main function and components for use in app.py
__all__ = [
    'generate_v2_chat_responses',
    'graph_v2',
    'memory_v2',
    'StateV2',
    'initialize_v2_components'
]
