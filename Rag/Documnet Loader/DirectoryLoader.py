from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader


loader = DirectoryLoader(
    path=r"E:\Resume\All resume",
    glob="*.pdf",
    loader_cls=PyPDFLoader

)
Loaded_pdf = loader.lazy_load()

for i in Loaded_pdf:
    print(i.metadata)
    print(i)
