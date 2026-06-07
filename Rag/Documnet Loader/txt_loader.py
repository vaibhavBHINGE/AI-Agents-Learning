from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import  RunnableBranch,RunnableLambda
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()

modelHuggingFace=ChatHuggingFace(llm=HuggingFaceEndpoint( model="deepseek-ai/DeepSeek-V4-Flash"))

strParser = StrOutputParser()

document=TextLoader(r"C:\Users\vbhin\Documents\workspace-spring-tool-suite-4-4.21.0.RELEASE\.metadata\.plugins\org.eclipse.jdt.core\javaLikeNames.txt").load()

prompt=PromptTemplate(
    
    template="What is in: {document}",
    input_variables=["document"],
)

chain=prompt | modelHuggingFace | strParser

output=chain.invoke({"document": document[0].page_content})
print(output)