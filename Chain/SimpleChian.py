from langchain_huggingface import  HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm_obj = HuggingFaceEndpoint(model="openbmb/MiniCPM4.1-8B")

model = ChatHuggingFace(llm=llm_obj)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="list top 5 interesting facts about {topic}.",
    input_variables=["topic"]
)

chain=prompt | model | parser
print(chain)
print(prompt)
result=chain.invoke({"topic": "cricket"})
print(result)

chain.get_graph().print_ascii()  # To visualize the chain structure