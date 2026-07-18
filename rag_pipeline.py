import os
import boto3
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock

from dotenv import load_dotenv
load_dotenv()

# --- Initialization ---
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1")


llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="amazon.nova-lite-v1:0",
    model_kwargs={"temperature": 0.0}
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# --- Memory ---
session_history = {}

def get_session_history(session_id: str) -> list:
    if session_id not in session_history:
        session_history[session_id] = []
    return session_history[session_id]

# --- Ingestion Methods ---
def ingest_documentation(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        text = f.read()
    chunks = text_splitter.create_documents([text], metadatas=[{"source_type": "documentation", "file": file_path}])
    vectorstore.add_documents(chunks)
    return f"Successfully ingested {len(chunks)} documentation chunks."

def ingest_faqs(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        text = f.read()
    chunks = text_splitter.create_documents([text], metadatas=[{"source_type": "faq", "file": file_path}])
    vectorstore.add_documents(chunks)
    return f"Successfully ingested {len(chunks)} FAQ chunks."

def ingest_chats(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        full_chat_text = f.read()
    
    chat_prompt = f"""Extract all technical questions and their verified solutions from this chat log.
    Ignore casual talk.
    
    Chat Log:
    {full_chat_text}
    
    Format:
    Question: [Summary]
    Verified Answer: [Summary]"""
    
    response = llm.invoke(chat_prompt)
    extracted_content = response.content.strip()
    
    chat_doc = Document(page_content=extracted_content, metadata={"source_type": "chat_history", "file": file_path})
    vectorstore.add_documents([chat_doc])
    
    # We now return a dictionary containing both the status message and the LLM output
    return {
        "message": "Successfully curated and ingested chat history.",
        "extracted_data": extracted_content
    }

# --- Query & Chat Engine ---
def contextualize_query(session_id: str, new_question: str) -> str:
    history = get_session_history(session_id)
    if not history:
        return new_question
        
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    rewrite_prompt = f"""Given the following conversation history and a new user question, 
    rewrite the new question into a standalone search query.
    
    Chat History:
    {history_text}
    
    New Question: {new_question}
    Standalone Query:"""
    
    response = llm.invoke(rewrite_prompt)
    return response.content.strip()

def generate_chat_response(session_id: str, question: str) -> str:
    """Handles the full RAG cycle: rewrite, retrieve, and respond."""
    standalone_query = contextualize_query(session_id, question)
    retrieved_docs = retriever.invoke(standalone_query)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    history = get_session_history(session_id)
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    
    final_prompt = f"""You are a helpful engineering assistant. 
    Use the retrieved technical context and the conversation history to answer the user's question.
    If you don't know the answer based on the context, say so.
    
    Conversation History:
    {history_text}
    
    Retrieved Context:
    {context_text}
    
    User Question: {question}
    Answer:"""
    
    response = llm.invoke(final_prompt)
    bot_answer = response.content.strip()
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": bot_answer})
    
    return bot_answer