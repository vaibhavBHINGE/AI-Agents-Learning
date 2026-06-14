from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever( wiki_client=any, top_k_results=2, lang="en")

query = "does god exist?"

langchain_document_objects = retriever.invoke(query)

print(langchain_document_objects)