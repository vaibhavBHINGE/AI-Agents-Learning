from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import  RunnableBranch,RunnableLambda, RunnableSequence, RunnablePassthrough,RunnableParallel
from dotenv import load_dotenv
load_dotenv()

modelHuggingFace=ChatHuggingFace(llm=HuggingFaceEndpoint( model="deepseek-ai/DeepSeek-V4-Flash"))

parser=StrOutputParser()

def word_count(text):
    return len(text.split())

prompt=PromptTemplate.from_template("What is the capital of {Country}?")
prompt2=PromptTemplate.from_template("tell me 5 interesting facts about {capital}?")

runnablPass_Through=RunnableSequence(prompt,modelHuggingFace,parser,RunnableParallel({
    "capital": RunnableSequence(prompt2,modelHuggingFace,parser),
    "capetailOfCountry": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
    }
)
)

result=runnablPass_Through.invoke({"Country":"Russia"})
print(result)

runnablPass_Through.get_graph().print_ascii()