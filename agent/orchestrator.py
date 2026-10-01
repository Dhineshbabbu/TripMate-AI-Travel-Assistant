import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tool_node, tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools.agent_tools import search_destination_guide,get_weather_forecast

project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not configured")

llm = ChatGroq(model="openai/gpt-oss-20b",
                     api_key=GROQ_API_KEY)

# Bind the tools with llm
llm = llm.bind_tools([search_destination_guide,get_weather_forecast])

# Define the LLM node and Tool node
def llm_node(state:AgentState) -> str:
    user_query = state["messages"]

    response = llm.invoke(user_query)

    return { 
        "messages":[response] 
        }


tool_node = ToolNode([search_destination_guide,get_weather_forecast])

#Define the Stategraph Node and Edges

graph = StateGraph(AgentState)

graph.add_node(llm_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"llm_node")
graph.add_conditional_edges("llm_node",tools_condition)
graph.add_edge("tools","llm_node")
graph = graph.compile(checkpointer=MemorySaver())

if __name__ == "__main__":

    #Load the System promt
    SYSTEM_PROMPT = project_root / "prompt" / "system_prompt.txt"
    with open(SYSTEM_PROMPT, "r", encoding="utf-8") as f:
        system_prompt_content = f.read()

    config = {"configurable": {"thread_id":"673c5538-61c1-4d95-a71f-d5ed5ef9e5b9"}}

    #make agent to understand the system prompt
    graph.invoke({"messages": [SystemMessage(content=system_prompt_content)]}, config=config)

    response = graph.invoke(
        {
            "messages":[HumanMessage(content="What should I pack for Reykjavik?")]
        },
        config=config
    )

    # for event in events:
    #     for node,messages in event.items():
    #         print(f"Response from {node}")
    #         for diff_nodes,message in messages.items():
    #             print(f"Message form {diff_nodes} is {message}")

    print(response["messages"][-1].content)