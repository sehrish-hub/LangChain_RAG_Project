import os
import atexit
from dotenv import load_dotenv

from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ Updated LangChain import
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma  # ✅ Corrected Chroma import

from langchain_google_genai import GoogleGenerativeAIEmbeddings  # ✅ Gemini Embeddings

# ✅ Load API Keys from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Ensure API Key is Set
if not GEMINI_API_KEY:
    raise ValueError("❌ Error: Missing GEMINI_API_KEY in .env file!")

# ✅ Get Available Gemini Models
available_models = [
    'gemini-1.5-pro', 'gemini-1.5-pro-002', 'gemini-1.5-flash', 'gemini-1.5-flash-002'
]
selected_model = "gemini-1.5-pro" if "gemini-1.5-pro" in available_models else available_models[0]
print(f"⚙️ Using Gemini Model: {selected_model}")

# ✅ Define GoogleGenerativeAIEmbeddings with the correct model & timeout
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY,
    request_timeout=180  # ⏳ Increased timeout to 180s
)

# ✅ Initialize Chroma Vector Store
print("🔍 Initializing Chroma Vector Store...")
vectorstore = Chroma(
    persist_directory="embeddings",
    embedding_function=gemini_embeddings
)
print("✅ Chroma Vector Store Initialized!")

# ✅ Check if vector store is empty and add sample data
if not vectorstore.get()['documents']:
    print("⚠️ Chroma is empty! Adding sample documents...")
    vectorstore.add_texts(["LangChain RAG retrieves documents using embeddings and LLMs."])
    print("✅ Sample documents added!")

retriever = vectorstore.as_retriever()

# ✅ Use Gemini-1.5-Pro as a fallback if Gemini-Pro is unavailable
try:
    llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=GEMINI_API_KEY)
except Exception:
    print("⚠️ Gemini model not found! Using gemini-1.5-flash as fallback.")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# ✅ Retry Mechanism for API failures
@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))  # 🔄 Retries 3 times with 10s delay
def generate_response(query):
    try:
        print(f"⚡ Query: {query}")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        response = qa_chain.invoke({"query": query})  # ✅ Updated to `.invoke()`
        print("✅ Response received!")  # Debug log
        return response["result"]
    except Exception as e:
        print(f"❌ Error in generate_response: {e}")
        raise e  # 🚨 Trigger retry if it fails

# ✅ Example Query
query = "LangChain RAG kaise kaam karta hai?"
response = generate_response(query)

print("📢 Final Response:")
print(response)

# ✅ Cleanup function to prevent gRPC timeout
def cleanup():
    global llm
    try:
        if llm:
            del llm  # ✅ Ensures a clean shutdown
            print("🔴 LLM and Retriever connections closed properly!")
    except NameError:
        pass  # If llm is already deleted, do nothing

# Register cleanup function
atexit.register(cleanup)
