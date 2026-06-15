from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv  
load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    Document(page_content="The cat is on the table."),
    Document(page_content="The dog is in the garden."),
    Document(page_content="The bird is flying in the sky."),
    Document(page_content="The fish is swimming in the pond.")
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="my_collection"
)
query = "Where is the cat?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.invoke(query)        
print(retrieved_docs)

for i, retrieved_doc in enumerate(retrieved_docs):
    print(f"\n ---Result {i+1}---")
    print(retrieved_doc.page_content)