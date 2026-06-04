from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import  RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import  Literal
from dotenv import load_dotenv
load_dotenv()

modelHuggingFace=ChatHuggingFace(llm=HuggingFaceEndpoint( model="deepseek-ai/DeepSeek-V4-Flash"))

strParser = StrOutputParser()
class FeedBack(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']=Field(description="sentiment of the feedback")

pydanticParser = PydanticOutputParser(pydantic_object=FeedBack)

# this will be the example of conditional chain, here by using sentiment of the feedback we will decide the resonance of the chain

prompt1 = PromptTemplate(
    template="give me the sentiment of the feedback: {feedback}./n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={'format_instruction': pydanticParser.get_format_instructions()})

prompt2 = PromptTemplate(template="give me the appropriate response for the positive {feedback}.",input_variables=["feedback"])
prompt3 = PromptTemplate(template="give me the appropriate response for the negative {feedback}.",input_variables=["feedback"])
prompt4 = PromptTemplate(template="give me the appropriate response for the neutral {feedback}.",input_variables=["feedback"])


sentiment_chain = prompt1 | modelHuggingFace | pydanticParser


conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | modelHuggingFace | strParser),
    (lambda x: x.sentiment == 'negative', prompt3 | modelHuggingFace | strParser),
    (lambda x: x.sentiment == 'neutral', prompt4 | modelHuggingFace | strParser),
    # Default branch
    RunnableLambda(lambda x: "There is no positive, negative or neutral sentiment in the feedback.")
)

chain = sentiment_chain | conditional_chain

prompt= "there is scope of improvement"

output=chain.invoke({"feedback": prompt})

print("output: ", output)

chain.get_graph().print_ascii()  # To visualize the chain structure



