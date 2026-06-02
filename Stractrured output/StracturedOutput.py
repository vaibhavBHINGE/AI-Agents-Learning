from dotenv import load_dotenv
load_dotenv(dotenv_path=r"F:\Gen AI\API calling\Ollama\.env")
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from pydantic import BaseModel, Field
llm_obj=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V4-Pro")
chat_model=ChatHuggingFace(llm=llm_obj)


# Schema for Pydantic 
class Response(BaseModel):
    sentiment: str = Field(description="The sentiment of the review")
    tone: str = Field(description="The tone of the review")
    summary: str = Field(description="A summary of the review")
    your_rating: int = Field(description="Your rating of the product")

model_Response=chat_model.with_structured_output(Response) 
print("model_response: ",model_Response)

query="""Not much comfortable for those who have wider finger area, after using this for months I am writing this review.
An yellow stain are appearing on the clothes which make this item less appealing so if you are able to manage it you can purchase this. Overall this item gives nice looks and will be comfortable for those who are having lean foot structure.
Not much heavy
Nice grip
Can pair with casuals."""
output=model_Response.invoke(query)

print(output)

