from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm_obj=HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",                                         
    timeout=300)

model=ChatHuggingFace(llm=llm_obj)

# model=ChatOllama(model="qwen2.5-00.5b",timeout=300)

#prompt template 1
prompt1=PromptTemplate(
    template="What is the capital of {country}?",
    input_variables=["country"]
)


# prompt template 2
prompt2=PromptTemplate(
    template="tell me more about this {capital}?",
    input_variables=["capital"]
)

parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'country': "China"})

print(result)