# Here we will call a LLM api from it will be an open source LLM from HuggingFace

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # hugginFace lib from langchain we need to pip install it
 
from dotenv import load_dotenv  # to call an LLM model we need to have api key that is stored in .env file
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro") # calling model
chat_model=ChatHuggingFace(llm=llm)

output=chat_model.invoke("how many planets are in our solar system?")  # invoking/ passing message prompt to LLM model
print(output.content) # .content will print only content without any metadata 