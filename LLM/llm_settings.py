from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm_obj=HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Pro",    # to extract the model from HuggingFace we need api token that is stored in .env file we importing that using load_dotenv() function
    temperature=0.7, # controls the randomness of the model's responses. A higher temperature (e.g., 1.0) will make the model's output more diverse and creative, while a lower temperature (e.g., 0.2) will make it more focused and deterministic.
    max_new_tokens=400, # the maximum number of tokens to generate in the completion.
    top_p=0.9, # controls the diversity of the model's output by limiting the token selection to a subset of the most probable tokens. A higher top_p (e.g., 0.9) will allow for more diverse responses, while a lower top_p (e.g., 0.5) will make the output more focused and deterministic.
)
chat_model=ChatHuggingFace(llm=llm_obj)

prompt="How many universes are there?"
response=chat_model.invoke(prompt)

print(response.content)
