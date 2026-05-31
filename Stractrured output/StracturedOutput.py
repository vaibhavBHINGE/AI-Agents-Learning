from dotenv import load_dotenv
load_dotenv(dotenv_path=r"F:\Gen AI\API calling\Ollama\.env")
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
    # temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    # max_tokens=None,
    # timeout=None,
    # max_retries=2,
    # # other params...
)


# Schema for typeDict 
class Response(TypedDict):

    sentiment: Annotated[str, "The sentiment of the review"]
    tone: Annotated[str, "The tone of the review"]
    summery: Annotated[str, "A summary of the review"]
    your_rating: Annotated[int, "Your rating of the product"]

model_Response=model.with_structured_output(Response)
print("model_response: ",model_Response)
output=model_Response.invoke("""Not much comfortable for those who have wider finger area, after using this for months I am writing this review.
An yellow stain are appearing on the clothes which make this item less appealing so if you are able to manage it you can purchase this. Overall this item givs nice looks and will be comfortable for those who are having lean foot structure.
Not much heavy
Nice grip
Can pair with casuals.""")

print(output)

