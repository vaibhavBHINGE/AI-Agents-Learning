from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import  RunnableBranch,RunnableLambda, RunnableSequence
from dotenv import load_dotenv
load_dotenv()

modelHuggingFace=ChatHuggingFace(llm=HuggingFaceEndpoint( model="deepseek-ai/DeepSeek-V4-Flash"))

strParser = StrOutputParser()

prompt1 = PromptTemplate(
    template="What is the country of player {name}, form he play in international matches?",
    input_variables=["name"]
)
prompt2 = PromptTemplate(
    template="tell more about the sport of this player {name}",
     input_variables=["name"]
     )

runnable_sequence = RunnableSequence(prompt1, modelHuggingFace ,strParser, prompt2, modelHuggingFace,strParser)

output = runnable_sequence.invoke({"name": "MS Dhoni"})

print(output)