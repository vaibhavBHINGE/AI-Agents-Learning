from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

llm_obj=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V4-Flash")

model=ChatHuggingFace(llm=llm_obj)  

class info(BaseModel):
    name:str=Field(description="name of the movie")
    ability:str=Field(description="ability of the character")
    girlfriend:str=Field(description="name of her girlfriend")


parser=PydanticOutputParser(pydantic_object=info)

#prompt template 1
prompt=PromptTemplate(
    template="tell me the movie name, ability and girlfriend name of this fictional character {character}?/n{format_instruction}", 
    input_variables=["character"],
    partial_variables={'format_instruction': parser.get_format_instructions()})

chain=prompt | model | parser
print(prompt)
result=chain.invoke({'character': "Hulk"})
print(result)