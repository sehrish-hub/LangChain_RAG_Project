from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Load stored vectors
vectorstore = Chroma(persist_directory="embeddings", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

def get_relevant_docs(query):
    return retriever.get_relevant_documents(query)

query = "LangChain kya hai?"
docs = get_relevant_docs(query)
for doc in docs:
    print(doc.page_content)
