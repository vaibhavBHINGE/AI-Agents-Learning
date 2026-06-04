from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

modelGroq=ChatGroq(model="openai/gpt-oss-20b")
modelGoogle=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
modelHuggingFace=ChatHuggingFace(llm=HuggingFaceEndpoint( model="deepseek-ai/DeepSeek-V4-Flash"))

parser = StrOutputParser()

prompt1 = PromptTemplate(template="give me the prime number between {start} and {end}.",input_variables=["start","end"])
prompt2 = PromptTemplate(template="give me the sum of all numbers between {start} and {end}.",input_variables=["start","end"])
prompt3 = PromptTemplate(template="give the addition between {start} and {end}.",input_variables=["start","end"])

parallel_chain = RunnableParallel({

    "start": prompt1 | modelGroq | parser,
    "end": prompt2 | modelGoogle | parser
      
    })

print("parallel_chain: ", parallel_chain)

merged_chain = prompt3 | modelHuggingFace | parser
print("merged_chain: ", merged_chain)
chain = parallel_chain | merged_chain
print("chain: ", chain)
result=chain.invoke({"start": 1, "end": 10})
print("result: ", result)

chain.get_graph().print_ascii()  # To visualize the chain structure