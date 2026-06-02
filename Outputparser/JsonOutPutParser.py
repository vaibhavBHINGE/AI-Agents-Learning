from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm_obj=HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash")

model=ChatHuggingFace(llm=llm_obj)  


parser=JsonOutputParser()
# Prompt template 1
prompt=PromptTemplate(
    template="tell me the movie name from {character} this fictional character?/n{format_instruction}", 
    input_variables=["character"],
    partial_variables={'format_instruction': parser.get_format_instructions()})

# prompt template 2

prompt2=PromptTemplate(
    template="tell me more about this {movie_name}?/n{format_instruction}",
    input_variables=["movie_name"],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain=prompt | model | parser | prompt2 | model | parser

result=chain.invoke({'character': "DR. STRANGE"})
print(result)