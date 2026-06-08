from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
load_dotenv()

model=ChatHuggingFace(llm=HuggingFaceEndpoint( model="deepseek-ai/DeepSeek-V4-Flash"))

loader = PyPDFLoader(r"C:\Users\vbhin\OneDrive\Desktop\Vaibhav_Bhinge.pdf")
Loaded_pdf = loader.load()

print(len(Loaded_pdf))
print(Loaded_pdf[0].page_content)
print(Loaded_pdf[0].metadata)   
print(Loaded_pdf)   
