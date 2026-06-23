from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor,create_tool_calling_agent
import requests
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
load_dotenv()

# chat_model=ChatGroq(
#     model="openai/gpt-oss-120b"
# )
llm_obj=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V4-Pro")
chat_model=ChatHuggingFace(llm=llm_obj)
Weather_api = os.getenv("OPENWEATHER_API_KEY")

#search tool
search_tool=TavilySearch()



template = """
Answer the following questions as best you can.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
...
Thought: I now know the final answer
Final Answer: the answer

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)
# prompt=hub.pull("hwchase17/react")

@tool
def get_weather_data(input: str) -> dict:
    
    """tool of get weather using city """
    url=f"https://api.openweathermap.org/data/2.5/weather?q={input}&appid={Weather_api}&units=metric"
    response=requests.get(url)
    return response.json()

#agent calling
agent_calling=create_react_agent(
    llm=chat_model,
    tools=[search_tool,get_weather_data],
    prompt=prompt
)

# tool exucation
agent_exucation=AgentExecutor(
    agent=agent_calling,
    tools=[search_tool,get_weather_data],
    verbose=True
    # handle_parsing_errors=True  
)

# invoking agent

result=agent_exucation.invoke({"input":"what is weather of pune?"})
print(result)