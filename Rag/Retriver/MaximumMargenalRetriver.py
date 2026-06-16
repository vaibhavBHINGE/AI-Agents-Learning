from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv  
load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    Document(page_content="The cat is on the table."),      
    Document(page_content="The dog is in the garden."),
    Document(page_content="The bird is flying in the sky."),
    Document(page_content="The fish is swimming in the pond."),
    Document(page_content="The table is on the cat."),
    Document(page_content="The garden is in the dog."),
    Document(page_content="The sky is flying in the bird."),    
    Document(page_content="The pond is swimming in the fish.")
]

vectorstore = FAISS.from_documents(
    documents=documents,    
    embedding=embeddings
)

query = "on the table"
retriever = vectorstore.as_retriever(
    search_type="mmr",  #---- SETTING SERCH TYPE IS MAXIMUM MARGINAL RETRIVAL
    search_kwargs={"k": 3,"lambda": 1}  # ----- here k is getting two results and lambda is the parameter that controls the diversity of the results. A higher lambda value will result in more diverse results, while a lower lambda value will result in more similar results.
    )

result=retriever.invoke(query)

for i, retrieved_doc in enumerate(result):
    print(f"\n ---Result {i+1}---")
    print(retrieved_doc.page_content)

