import os
import boto3
import operator
from typing import TypedDict, Literal, Annotated
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

# --- Initialization ---
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1"))

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0.0}
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# --- In-Memory Checkpointer ---
memory = MemorySaver()


# ============================================================
# LANGGRAPH: State + Nodes + Router
# ============================================================

# --- State Definition ---
class GraphState(TypedDict):
    session_id: str
    question: str
    standalone_query: str
    retrieved_context: str
    query_type: str                                # "greeting" or "technical"
    answer: str
    messages: Annotated[list, operator.add]         # auto-accumulates across invocations


# --- Node 1: Route the query ---
def route_query(state: GraphState) -> GraphState:
    """Uses the LLM to classify the user's question as greeting or technical."""
    question = state["question"]

    messages = state.get("messages", [])[-20:]
    classification_prompt = f"""Classify the following user message into exactly one category.

Categories:
- "greeting": casual talk, hello, hi, how are you, thanks, bye, personal introductions (like "I am...", "my name is..."), or any non-technical conversation.
- "technical": any question about engineering, infrastructure, systems, debugging, documentation, or technical knowledge.

If unsure, classify as "greeting".

User Message: {question}

Respond with ONLY one word: greeting or technical"""


    response = llm.invoke(classification_prompt)
    query_type = response.content.strip().lower()

    # Fallback: if LLM returns something unexpected, default to technical
    if query_type not in ("greeting", "technical"):
        query_type = "technical"

    return {"query_type": query_type}


# --- Node 2: Handle greeting ---
def handle_greeting(state: GraphState) -> GraphState:
    """Responds to casual/greeting messages without hitting the vector DB."""
    question = state["question"]

    greeting_prompt = f"""You are a friendly engineering assistant. 
The user sent a casual message. Respond warmly and briefly. 
Let them know you're here to help with technical questions.

IMPORTANT: If conversation history shows conflicting user information (like different names),
always trust the MOST RECENT message.

User Message: {question}
Response:"""

    response = llm.invoke(greeting_prompt)
    answer = response.content.strip()

    return {
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


# --- Node 3: Rewrite query for RAG ---
def rewrite_query(state: GraphState) -> GraphState:
    """Contextualizes the question using conversation history from MemorySaver."""
    question = state["question"]
    # Use last 20 messages (10 conversation turns):
    messages = state.get("messages", [])[-20:]

    if not messages:
        return {"standalone_query": question}

    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    )
    rewrite_prompt = f"""Given the following conversation history and a new user question, 
    rewrite the new question into a standalone search query.
    
    Chat History:
    {history_text}
    
    New Question: {question}
    Standalone Query:"""

    response = llm.invoke(rewrite_prompt)
    return {"standalone_query": response.content.strip()}


# --- Node 4: Retrieve documents ---
def retrieve_documents(state: GraphState) -> GraphState:
    """Fetches relevant documents from ChromaDB."""
    standalone_query = state["standalone_query"]
    retrieved_docs = retriever.invoke(standalone_query)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"retrieved_context": context_text}


# --- Node 5: Generate technical answer ---
def generate_answer(state: GraphState) -> GraphState:
    """Generates the final answer using retrieved context + history."""
    question = state["question"]
    context_text = state["retrieved_context"]
    messages = state.get("messages", [])

    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    )

    final_prompt = f"""You are a helpful engineering assistant. 
    Use the retrieved technical context and the conversation history to answer the user's question.
    If you don't know the answer based on the context, say so.
    
    IMPORTANT: If the user has provided conflicting information (like different names), 
    always trust the MOST RECENT message. Treat newer messages as corrections to older ones.
    
    Conversation History:
    {history_text}
    
    Retrieved Context:
    {context_text}
    
    User Question: {question}
    Answer:"""

    response = llm.invoke(final_prompt)
    answer = response.content.strip()

    # Append this turn's messages — operator.add accumulates them in checkpoint
    return {
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


# --- Conditional Edge: decide which path ---
def decide_route(state: GraphState) -> Literal["handle_greeting", "rewrite_query"]:
    """Routes to greeting handler or RAG pipeline based on classification."""
    if state["query_type"] == "greeting":
        return "handle_greeting"
    return "rewrite_query"


# ============================================================
# BUILD THE GRAPH
# ============================================================

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("route_query", route_query)
workflow.add_node("handle_greeting", handle_greeting)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_answer", generate_answer)

# Add edges
workflow.add_edge(START, "route_query")
workflow.add_conditional_edges("route_query", decide_route)
workflow.add_edge("handle_greeting", END)
workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile with in-memory checkpointer
graph = workflow.compile(checkpointer=memory)


# ============================================================
# PUBLIC API (called by main.py — no changes needed there)
# ============================================================

def generate_chat_response(session_id: str, question: str) -> str:
    """Handles the full cycle: route → (greet | RAG) → respond."""
    initial_state: GraphState = {
        "session_id": session_id,
        "question": question,
        "standalone_query": "",
        "retrieved_context": "",
        "query_type": "",
        "answer": "",
        "messages": [],
    }

    # thread_id ties this invocation to the session's checkpoint
    config = {"configurable": {"thread_id": session_id}}
    final_state = graph.invoke(initial_state, config)

    return final_state["answer"]


# ============================================================
# INGESTION METHODS (unchanged)
# ============================================================

def ingest_documentation(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        text = f.read()
    chunks = text_splitter.create_documents(
        [text], metadatas=[{"source_type": "documentation", "file": file_path}]
    )
    vectorstore.add_documents(chunks)
    return f"Successfully ingested {len(chunks)} documentation chunks."

def ingest_faqs(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        text = f.read()
    chunks = text_splitter.create_documents(
        [text], metadatas=[{"source_type": "faq", "file": file_path}]
    )
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

    chat_doc = Document(
        page_content=extracted_content,
        metadata={"source_type": "chat_history", "file": file_path},
    )
    vectorstore.add_documents([chat_doc])

    return {
        "message": "Successfully curated and ingested chat history.",
        "extracted_data": extracted_content,
    }
