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
    docs_relevant: str                             # "yes" or "no" — set by grader
    sources: list                                  # source file names from retrieved docs
    answer: str
    messages: Annotated[list, operator.add]         # auto-accumulates across invocations


# --- Node 1: Route the query ---
def route_query(state: GraphState) -> GraphState:
    """Uses the LLM to classify the user's question as greeting or technical."""
    question = state["question"]

    classification_prompt = f"""Classify the following user message into exactly one category.

Categories:
- "greeting": casual talk, hello, hi, how are you, thanks, bye, personal introductions (like "I am...", "my name is..."), or any non-technical conversation.
- "technical": any question about engineering, infrastructure, systems, debugging, documentation, or technical knowledge.

If unsure, classify as "greeting".

User Message: {question}

Respond with ONLY one word: greeting or technical"""

    response = llm.invoke(classification_prompt)
    query_type = response.content.strip().lower()

    if query_type not in ("greeting", "technical"):
        query_type = "technical"

    return {"query_type": query_type}


# --- Node 2: Handle greeting ---
def handle_greeting(state: GraphState) -> GraphState:
    """Responds to casual/greeting messages without hitting the vector DB."""
    question = state["question"]
    messages = state.get("messages", [])[-20:]

    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    ) if messages else "No prior conversation."

    greeting_prompt = f"""You are a friendly engineering assistant. 
The user sent a casual message. Respond warmly and briefly. 
Let them know you're here to help with technical questions.

IMPORTANT: If conversation history shows conflicting user information (like different names),
always trust the MOST RECENT message.

Conversation History:
{history_text}

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


# --- Node 4: Retrieve documents (now captures sources) ---
def retrieve_documents(state: GraphState) -> GraphState:
    """Fetches relevant documents from ChromaDB and extracts source metadata."""
    standalone_query = state["standalone_query"]
    retrieved_docs = retriever.invoke(standalone_query)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Extract unique source file names for attribution
    sources = list(set(
        os.path.basename(doc.metadata.get("file", "unknown source"))
        for doc in retrieved_docs
    ))

    return {"retrieved_context": context_text, "sources": sources}


# --- Node 5: Grade retrieved documents (NEW — Retrieval Grader) ---
def grade_documents(state: GraphState) -> GraphState:
    """LLM checks if the retrieved documents are relevant to the question."""
    question = state["question"]
    context_text = state["retrieved_context"]

    grading_prompt = f"""You are a relevance grader. Given a user question and retrieved documents, 
determine if the documents contain information relevant to answering the question.

User Question: {question}

Retrieved Documents:
{context_text}

Are these documents relevant to the question? Respond with ONLY one word: yes or no"""

    response = llm.invoke(grading_prompt)
    relevance = response.content.strip().lower()

    if relevance not in ("yes", "no"):
        relevance = "no"  # If uncertain, treat as not relevant (safer)

    return {"docs_relevant": relevance}


# --- Node 6: Generate technical answer (now includes sources) ---
def generate_answer(state: GraphState) -> GraphState:
    """Generates the final answer using retrieved context + history + source attribution."""
    question = state["question"]
    context_text = state["retrieved_context"]
    sources = state.get("sources", [])
    messages = state.get("messages", [])[-20:]

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

    # Append source attribution
    if sources:
        source_list = ", ".join(sources)
        answer += f"\n\n📎 **Sources:** {source_list}"

    return {
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


# --- Node 7: Fallback response (NEW — when docs aren't relevant) ---
def fallback_response(state: GraphState) -> GraphState:
    """Provides an honest response when retrieved docs are not relevant."""
    question = state["question"]

    answer = (
        "I couldn't find relevant information in the knowledge base to answer your question. "
        "This might be because:\n"
        "- The topic hasn't been added to the documentation yet\n"
        "- The question needs to be rephrased\n\n"
        "You can upload relevant documentation using the **📂 Update Knowledge Base** page, "
        "or try rephrasing your question."
    )

    return {
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


# --- Conditional Edge: greeting vs technical ---
def decide_route(state: GraphState) -> Literal["handle_greeting", "rewrite_query"]:
    """Routes to greeting handler or RAG pipeline based on classification."""
    if state["query_type"] == "greeting":
        return "handle_greeting"
    return "rewrite_query"


# --- Conditional Edge: relevant vs not relevant (NEW) ---
def decide_relevance(state: GraphState) -> Literal["generate_answer", "fallback_response"]:
    """Routes to answer generation or fallback based on document relevance."""
    if state["docs_relevant"] == "yes":
        return "generate_answer"
    return "fallback_response"


# ============================================================
# BUILD THE GRAPH
# ============================================================

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("route_query", route_query)
workflow.add_node("handle_greeting", handle_greeting)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("fallback_response", fallback_response)

# Add edges
workflow.add_edge(START, "route_query")
workflow.add_conditional_edges("route_query", decide_route)
workflow.add_edge("handle_greeting", END)
workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_relevance)
workflow.add_edge("generate_answer", END)
workflow.add_edge("fallback_response", END)

# Compile with in-memory checkpointer
graph = workflow.compile(checkpointer=memory)


# ============================================================
# PUBLIC API (called by main.py — no changes needed there)
# ============================================================

def generate_chat_response(session_id: str, question: str) -> str:
    """Handles the full cycle: route → (greet | RAG with grading) → respond."""
    initial_state: GraphState = {
        "session_id": session_id,
        "question": question,
        "standalone_query": "",
        "retrieved_context": "",
        "query_type": "",
        "docs_relevant": "",
        "sources": [],
        "answer": "",
        "messages": [],
    }

    config = {"configurable": {"thread_id": session_id}}
    final_state = graph.invoke(initial_state, config)

    return final_state["answer"]


# ============================================================
# INGESTION METHODS (unchanged)
# ============================================================

def ingest_documentation(file_path: str, original_filename: str = None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        text = f.read()

    source_name = original_filename or os.path.basename(file_path)
    chunks = text_splitter.create_documents(
        [text], metadatas=[{"source_type": "documentation", "file": source_name}]
    )
    vectorstore.add_documents(chunks)
    return f"Successfully ingested {len(chunks)} documentation chunks from {source_name}."

def ingest_faqs(file_path: str, original_filename: str = None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        text = f.read()

    source_name = original_filename or os.path.basename(file_path)
    chunks = text_splitter.create_documents(
        [text], metadatas=[{"source_type": "faq", "file": source_name}]
    )
    vectorstore.add_documents(chunks)
    return f"Successfully ingested {len(chunks)} FAQ chunks from {source_name}."

def ingest_chats(file_path: str, original_filename: str = None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        full_chat_text = f.read()

    source_name = original_filename or os.path.basename(file_path)

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
        metadata={"source_type": "chat_history", "file": source_name},
    )
    vectorstore.add_documents([chat_doc])

    return {
        "message": f"Successfully curated and ingested chat history from {source_name}.",
        "extracted_data": extracted_content,
    }

