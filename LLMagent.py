import os
import time
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",  # Groq supports multiple models
    temperature=0.7,
    api_key=groq_api_key,
    streaming=True,
)

# pre-built tool that fetches the latest web results.
search_tool = DuckDuckGoSearchRun()

# Wrapping it with @tool lets the agent call it during reasoning
@tool
def search_latest(query: str) -> str:
    """Search the web and return the latest results."""
    return search_tool.run(query)

# Create the agent
# create_agent() connects the LLM + tools.
agent = create_agent(
    model=llm,
    tools=[search_latest],
    system_prompt="You are a helpful assistant that provides the latest search results."
)

def stream_agent_response(user_input: str):
    # DIRECT LLM STREAM (real tokens)
    for chunk in llm.stream([HumanMessage(content=user_input)]):
        if chunk.content:
            for char in chunk.content:
                yield char
                time.sleep(0.01)

# Function used by FastAPI
# def run_agent(user_input: str) -> str:
#     result = agent.invoke({
#         "messages": [HumanMessage(content=user_input)]
#     })
#     return result["messages"][-1].content
