"""
v2.py - AI Search API v2 with Vertex AI and Google Search
Uses ChatVertexAI for conversational AI and GoogleSearchAPIWrapper for web search
"""

from typing import TypedDict, Annotated, Optional, AsyncGenerator
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
import os
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
import json

load_dotenv()

# Load API Keys for v2
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Custom Search Engine ID
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")  # GCP Project ID
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")  # Vertex AI location

# Initialize memory saver for checkpointing (shared with v1)
memory_v2 = MemorySaver()

class StateV2(TypedDict):
    """State definition for v2 graph"""
    messages: Annotated[list, add_messages]


def initialize_v2_components():
    """
    Initialize Vertex AI model and Google Search tool
    Returns: tuple of (llm_with_tools, search_tool, graph)
    """
    
    # Initialize Google Search API Wrapper
    google_search = GoogleSearchAPIWrapper(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
        k=4  # Number of results to return
    )
    
    # Create a LangChain Tool from Google Search
    search_tool = Tool(
        name="google_search",
        description="Search Google for recent results. Use this when you need current information, facts, news, or answers about any topic.",
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
    
    # Build the graph
    graph = build_v2_graph(llm_with_tools, search_tool)
    
    return llm_with_tools, search_tool, graph


def build_v2_graph(llm_with_tools, search_tool):
    """
    Build LangGraph workflow for v2
    """
    
    async def model_node(state: StateV2):
        """Model node that invokes the LLM"""
        result = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [result]}
    
    async def tools_router(state: StateV2):
        """Router to decide if we need to call tools"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "tool_node"
        else:
            return END
    
    async def tool_node(state: StateV2):
        """Tool node that executes tool calls"""
        tool_calls = state["messages"][-1].tool_calls
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            if tool_name == "google_search":
                # Execute Google Search
                query = tool_args.get("query", "")
                search_results = await search_tool.arun(query)
                
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


# Initialize v2 components globally
try:
    llm_v2, search_tool_v2, graph_v2 = initialize_v2_components()
except Exception as e:
    print(f"Warning: Could not initialize v2 components: {e}")
    print("Make sure GOOGLE_API_KEY, GOOGLE_CSE_ID, and GOOGLE_CLOUD_PROJECT are set")
    llm_v2, search_tool_v2, graph_v2 = None, None, None


def serialize_ai_message_chunk_v2(chunk):
    """Serialize AI message chunks for streaming"""
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )


async def generate_v2_chat_responses(
    message: str, 
    checkpoint_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Generate streaming chat responses using Vertex AI and Google Search
    
    Args:
        message: User's message
        checkpoint_id: Optional checkpoint ID for conversation continuity
        
    Yields:
        Server-Sent Events (SSE) formatted strings
    """
    
    if graph_v2 is None:
        yield f"data: {{\"type\": \"error\", \"message\": \"v2 components not initialized\"}}\n\n"
        return
    
    is_new_conversation = checkpoint_id is None
    
    if is_new_conversation:
        new_checkpoint_id = str(uuid4())
        config = {"configurable": {"thread_id": new_checkpoint_id}}
        
        events = graph_v2.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
        
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {"configurable": {"thread_id": checkpoint_id}}
        events = graph_v2.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
    
    async for event in events:
        event_type = event["event"]
        
        # Stream content chunks
        if event_type == "on_chat_model_stream":
            chunk_content = serialize_ai_message_chunk_v2(event["data"]["chunk"])
            if isinstance(chunk_content, list):
                chunk_content = " ".join(str(item) for item in chunk_content)
            
            # Escape content for JSON
            safe_content = str(chunk_content).replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"
        
        # Detect search intent
        elif event_type == "on_chat_model_end":
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "google_search"]
            
            if search_calls:
                search_query = search_calls[0]["args"].get("query", "")
                safe_query = search_query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"
        
        # Return search results (Google Search returns text snippets, not URLs directly)
        elif event_type == "on_tool_end" and event["name"] == "google_search":
            output = event["data"]["output"]
            
            # Google Search returns a formatted string with snippets
            # We'll send it as search results metadata
            safe_output = str(output).replace('"', '\\"').replace("\n", "\\n")
            yield f"data: {{\"type\": \"search_results\", \"results\": \"{safe_output}\"}}\n\n"
    
    yield f"data: {{\"type\": \"end\"}}\n\n"


# Export the main function for use in app.py
__all__ = ['generate_v2_chat_responses', 'graph_v2', 'memory_v2']
