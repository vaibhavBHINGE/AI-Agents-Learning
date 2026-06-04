from langchain_huggingface import  HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm_obj = HuggingFaceEndpoint( model="deepseek-ai/DeepSeek-V4-Flash")

model = ChatHuggingFace(llm=llm_obj)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="give me the achievements of cricket player {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="select top 5 of his achievements as a cricket player: {topic}.",
    input_variables=["topic"]
)

chain=prompt | model | prompt2 | model | parser
print(chain)
result=chain.invoke({"topic": "Ms Dhoni"})
print(result)
chain.get_graph().print_ascii()  # To visualize the chain structure