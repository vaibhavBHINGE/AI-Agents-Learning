from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# loader = PyPDFLoader(r"C:\Users\vbhin\OneDrive\Desktop\Vaibhav_Bhinge.pdf")
# document = loader.load()
document = "This is a sample document that we will split into chunks based on character count. The chunk size is set to 100 characters, and there is no overlap between the chunks. This allows us to process the document in smaller pieces, which can be useful for various applications such as natural language processing or machine learning tasks."
text_splitter = CharacterTextSplitter(
    
    chunk_size=10, chunk_overlap=0, separator=""
)

result = text_splitter.split_text(document)

print(result)