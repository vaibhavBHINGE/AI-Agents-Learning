from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm_obj=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash")

model=ChatHuggingFace(llm=llm_obj)  

schema=(
    ResponseSchema(name="Fact-1", description="decription of facet-1"),
    ResponseSchema(name="Fact-2", description="decription of facet-2")
    ResponseSchema(name="Fact-3", description="decription of facet-3"),
    ResponseSchema(name="Fact-4", description="description of facet-4")
)

parser=StructuredOutputParser.from_response_schemas(schema)
Prompt template 1
prompt=PromptTemplate(  
    template="tell me the movie name from {character} this fictional character?/n{format_instruction}", 
    input_variables=["character"],
    partial_variables={'format_instruction': parser.get_format_instructions()})

