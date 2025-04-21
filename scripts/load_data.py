from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def load_and_split(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    return split_docs
