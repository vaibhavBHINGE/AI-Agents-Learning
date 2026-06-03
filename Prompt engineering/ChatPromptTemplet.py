from langchain_core.messages import  HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import  ChatHuggingFace ,HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()

model=ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        model="google/gemma-4-31B-it"
    )
)

# model=ChatOllama(
#     model="qwen2.5:0.5b")

template = ChatPromptTemplate(
    [
        ("system", "You are expert in {domain}."),
        ("human", "tell me about {topic}")
    ]
)

templet=template.format_messages(domain="cricket", topic="Dusra ball")

print("Prompt: ",templet)
response=model.invoke(templet)
print("Response: ",response.content)