from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from load_data import load_and_split

# Load Data
docs = load_and_split("data/sample.txt")

# Create Vector Store
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# Save Database
vectorstore.persist()
print("Embeddings stored successfully!")
