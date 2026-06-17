from langchain_huggingface import HuggingFaceEndpointEmbeddings , HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever,BM25Retriever,EnsembleRetriever
from dotenv import load_dotenv
load_dotenv()

embeddings_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"  
)

chat_model = ChatHuggingFace(llm=HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V4-Flash"))

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embeddings_model
)

# the following code is for multi query retrival to the defect from the similarity retrival 
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3})

multi_query_retriever = MultiQueryRetriever.from_llm(
    llm=chat_model,
    retriever= vectorstore.as_retriever(search_kwargs={"k": 5}))

query = "How to improve energy levels and maintain balance?"

similarity_results = similarity_retriever.invoke(query)
multi_query_results = multi_query_retriever.invoke(query)

print("Similarity Retriever Results:")
for i, doc in enumerate(similarity_results):
    print(f"\n--- Similarity Result {i+1} ---")
    print(f"Content: {doc.page_content}")

print("\nMulti-Query Retriever Results:")
for i, doc in enumerate(multi_query_results):
    print(f"\n--- Multi-Query Result {i+1} ---")
    print(f"Content: {doc.page_content}")