import os
from datetime import datetime
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
    model_kwargs={"temperature": 0.5}
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
# MMR (Maximal Marginal Relevance): retrieves diverse and relevant chunks.
# fetch_k=20 candidates are fetched, then narrowed to k=10 most relevant & diverse.
# This improves Hit Ratio, Precision, and Recall over plain similarity search.
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
)
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
    
    # 📝 We updated the prompt to explicitly ask for the solver's name
    chat_prompt = f"""Extract all technical questions and their verified solutions from this chat log.
    Ignore casual talk. Identify the name of the person who provided the solution and the timestamp when the solution was given.
    
    Chat Log:
    {full_chat_text}
    
    Format:
    Question: [Summary]
    Verified Answer: [Summary]
    Resolved By: [Name of the person]
    Date Resolved: [Timestamp of the solution]"""
    response = llm.invoke(chat_prompt)
    extracted_content = response.content.strip()
    
    chat_doc = Document(page_content=extracted_content, metadata={"source_type": "chat_history", "file": file_path})
    vectorstore.add_documents([chat_doc])
    
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
    rewrite_prompt = f"""You are a query rewriter for a RAG system.

Given a conversation history and a new follow-up question from the user, rewrite it as a clear standalone search query.

Rules:
- If the user is asking to elaborate or explain something already mentioned (e.g. "explain first", "tell me more", "why?"), rewrite it to specifically target the NEW aspect they want — not a repeat of what was already answered.
- If the user is asking about a new topic, write a standalone query for that topic.
- If the user's question is unrelated to the conversation (e.g. small talk, off-topic), return it as-is.
- NEVER produce a query that would retrieve the exact same information already given in the last assistant response.

Conversation History:
{history_text}

New User Question: {new_question}

Standalone Search Query (be specific and targeted):"""

    response = llm.invoke(rewrite_prompt)
    return response.content.strip()

def generate_chat_response(session_id: str, question: str) -> str:
    """Handles the full RAG cycle: rewrite, retrieve, and respond."""
    standalone_query = contextualize_query(session_id, question)
    retrieved_docs = retriever.invoke(standalone_query)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    history = get_session_history(session_id)
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    
    today = datetime.now().strftime("%A, %B %d, %Y")

    final_prompt = f"""You are a helpful and knowledgeable engineering assistant for a software team.
Today's date is: {today}

<instructions>
Answer the user's question using ONLY the facts found in the <context> section below.
- NEVER copy, quote, or reproduce the <context> block in your response. Use it silently as reference only.
- If the question is specific (e.g., "how to fix X"), give a direct concise answer using exact commands/values from context.
- If the question is broad or vague, summarize the relevant information — list known errors, issues, or topics found.
- For time-sensitive questions like "latest changes" or "recent updates", look for timestamps and summarize recent entries.
- Do NOT repeat information already given in <history>. Build on it or go deeper.
- If the user says "explain first" / "tell me more", elaborate on the FIRST point from your last answer with more detail.
- If the context has genuinely NO relevant information, say: "I couldn't find information about this in the available documentation. Could you be more specific?"
- Do NOT invent commands, configs, values, or technical details not present in context.
</instructions>

<history>
{history_text}
</history>

<context>
{context_text}
</context>

User Question: {question}
Answer:"""
    
    response = llm.invoke(final_prompt)
    bot_answer = response.content.strip()
    
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": bot_answer})
    
    return bot_answer