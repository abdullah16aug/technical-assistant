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
from tavily import TavilyClient

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

memory = MemorySaver()


# ============================================================
# LANGGRAPH: State + Nodes + Router
# ============================================================

class GraphState(TypedDict):
    session_id: str
    question: str
    standalone_query: str
    retrieved_context: str
    query_type: str
    docs_relevant: str
    sources: list
    needs_clarification: bool
    clarification_question: str
    answer: str
    messages: Annotated[list, operator.add]    # accumulates conversation history


# --- Helper: build readable trace label from stream chunk ---
def _build_trace_label(node_name: str, state_update: dict) -> str:
    """Builds a human-readable trace label from node name and state update."""
    if node_name == "route_query":
        return f"route_query → {state_update.get('query_type', '')}"
    elif node_name == "planner":
        nc = state_update.get("needs_clarification", False)
        return f"planner → {'vague, asking for clarification' if nc else 'specific, proceeding to RAG'}"
    elif node_name == "ask_clarification":
        return "ask_clarification → waiting for user to clarify"
    elif node_name == "rewrite_query":
        sq = state_update.get("standalone_query", "")
        short = sq[:60] + "..." if len(sq) > 60 else sq
        return f"rewrite_query → '{short}'"
    elif node_name == "retrieve_documents":
        sources = state_update.get("sources", [])
        return f"retrieve_documents → {len(sources)} source(s) {sources}"
    elif node_name == "grade_documents":
        return f"grade_documents → docs relevant: {state_update.get('docs_relevant', '')}"
    elif node_name == "generate_answer":
        return "generate_answer → answered from knowledge base ✅"
    elif node_name == "web_search":
        answer = state_update.get("answer", "")
        if "couldn't find" in answer:
            return "web_search → no results from DuckDuckGo ❌"
        return "web_search → answered from DuckDuckGo 🌐"
    elif node_name == "handle_greeting":
        return "handle_greeting → casual response"
    return node_name


# --- Node 1: Route the query ---
def route_query(state: GraphState) -> GraphState:
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


# --- Node 3: Planner ---
def planner(state: GraphState) -> GraphState:
    question = state["question"]
    messages = state.get("messages", [])[-20:]

    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    ) if messages else "No prior conversation."

    planner_prompt = f"""You are an engineering assistant analyzing a user question.

Conversation History:
{history_text}

Current Question: {question}

Determine if this question is specific enough to search technical documentation,
OR if it is too vague and needs clarification.

A question is TOO VAGUE if:
- It mentions a general technology without specifying which one (e.g., "pipeline issue" without saying Airflow/Spark/Dataflow)
- It references "the error" or "the problem" without any context
- It says "how to fix it" without saying what "it" is
- The conversation history does NOT provide enough context to make it specific

A question is SPECIFIC ENOUGH if:
- It names the technology (e.g., BigQuery, Airflow, Kubernetes, Spark)
- It describes a concrete problem or scenario
- Prior conversation history already provides the missing context

Respond in this EXACT format (two lines only):
VERDICT: specific
CLARIFICATION: (leave blank if specific, or write one short clarifying question if vague)"""

    response = llm.invoke(planner_prompt)
    output = response.content.strip()

    needs_clarification = False
    clarification_question = ""

    for line in output.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().lower()
            needs_clarification = "vague" in verdict
        elif line.startswith("CLARIFICATION:"):
            clarification_question = line.replace("CLARIFICATION:", "").strip()

    if needs_clarification and not clarification_question:
        clarification_question = "Could you provide more details about your question?"

    return {
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
    }


# --- Node 4: Ask clarification ---
def ask_clarification(state: GraphState) -> GraphState:
    question = state["question"]
    clarification = state.get("clarification_question", "Could you clarify your question?")

    return {
        "answer": clarification,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": clarification},
        ],
    }


# --- Node 5: Rewrite query ---
def rewrite_query(state: GraphState) -> GraphState:
    question = state["question"]
    messages = state.get("messages", [])[-20:]

    if not messages:
        return {"standalone_query": question}

    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    )

    rewrite_prompt = f"""Given the conversation history and a new user question, rewrite the question as a short standalone search query.

IMPORTANT: Output ONLY the search query itself. No explanation, no sentences, no punctuation at the end. Just the search terms.

Chat History:
{history_text}

New Question: {question}
Standalone Query:"""

    response = llm.invoke(rewrite_prompt)
    standalone = response.content.strip()

    # Safety: if the LLM still returns a verbose answer, fall back to original question
    if len(standalone) > 150 or "\n" in standalone:
        standalone = question

    return {"standalone_query": standalone}


# --- Node 6: Retrieve documents ---
def retrieve_documents(state: GraphState) -> GraphState:
    standalone_query = state["standalone_query"]
    retrieved_docs = retriever.invoke(standalone_query)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    sources = list(set(
        os.path.basename(doc.metadata.get("file", "unknown source"))
        for doc in retrieved_docs
    ))

    return {"retrieved_context": context_text, "sources": sources}


# --- Node 7: Grade retrieved documents ---
def grade_documents(state: GraphState) -> GraphState:
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
        relevance = "no"

    return {"docs_relevant": relevance}


# --- Node 8: Generate answer from KB ---
def generate_answer(state: GraphState) -> GraphState:
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


# --- Node 9: Web Search fallback (Tavily) ---
def web_search(state: GraphState) -> GraphState:
    """Falls back to Tavily web search when KB has no relevant docs.
    Gracefully skips if TAVILY_API_KEY is not set or quota is exhausted.
    """
    question = state["question"]
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    # --- Skip if no API key configured ---
    if not tavily_key or tavily_key == "your_key_here":
        answer = (
            "I couldn't find relevant information in my knowledge base for this question. "
            "Web search is not configured yet (no TAVILY_API_KEY in .env). "
            "You can upload relevant documentation using the **📂 Update Knowledge Base** page."
        )
        return {
            "answer": answer,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    try:
        tavily = TavilyClient(api_key=tavily_key)
        response = tavily.search(
            query=question,
            max_results=3,
            search_depth="basic",   # "advanced" for deeper search (uses 2 credits)
        )
        results = response.get("results", [])

        if not results:
            answer = (
                "I couldn't find relevant information in the knowledge base or on the web. "
                "Please try rephrasing your question or upload relevant documentation."
            )
            return {
                "answer": answer,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            }

        # Tavily returns: [{title, url, content, score}, ...]
        web_context = "\n\n".join([
            f"Title: {r.get('title', '')}\nSource: {r.get('url', '')}\n{r.get('content', '')}"
            for r in results
        ])

        web_prompt = f"""You are a helpful engineering assistant.
The user's question was not found in the internal knowledge base, so here are web search results.
Use these results to answer the question. Be concise and accurate.
Only use information from the provided search results.

User Question: {question}

Web Search Results:
{web_context}

Answer:"""

        response = llm.invoke(web_prompt)
        answer = response.content.strip()

        web_sources = "\n".join([f"- [{r.get('title', r.get('url', ''))}]({r.get('url', '')})" for r in results])
        answer += f"\n\n🌐 **Web Sources:**\n{web_sources}"

    except Exception as e:
        err_str = str(e).lower()
        if "quota" in err_str or "rate" in err_str or "limit" in err_str or "429" in err_str:
            answer = (
                "I couldn't find the answer in the knowledge base, and the Tavily web search quota has been reached. "
                "Please try again later or upload relevant documentation using the **📂 Update Knowledge Base** page."
            )
        else:
            answer = (
                "I couldn't find relevant information in the knowledge base. "
                f"Web search encountered an error: {str(e)}\n\n"
                "You can upload relevant documentation using the **📂 Update Knowledge Base** page."
            )

    return {
        "answer": answer,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


# --- Conditional Edges ---
def decide_route(state: GraphState) -> Literal["handle_greeting", "planner"]:
    if state["query_type"] == "greeting":
        return "handle_greeting"
    return "planner"


def decide_after_planner(state: GraphState) -> Literal["ask_clarification", "rewrite_query"]:
    if state.get("needs_clarification"):
        return "ask_clarification"
    return "rewrite_query"


def decide_relevance(state: GraphState) -> Literal["generate_answer", "web_search"]:
    if state["docs_relevant"] == "yes":
        return "generate_answer"
    return "web_search"


# ============================================================
# BUILD THE GRAPH
# ============================================================

workflow = StateGraph(GraphState)

workflow.add_node("route_query", route_query)
workflow.add_node("handle_greeting", handle_greeting)
workflow.add_node("planner", planner)
workflow.add_node("ask_clarification", ask_clarification)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("web_search", web_search)

workflow.add_edge(START, "route_query")
workflow.add_conditional_edges("route_query", decide_route)
workflow.add_edge("handle_greeting", END)
workflow.add_conditional_edges("planner", decide_after_planner)
workflow.add_edge("ask_clarification", END)
workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_relevance)
workflow.add_edge("generate_answer", END)
workflow.add_edge("web_search", END)

graph = workflow.compile(checkpointer=memory)


# ============================================================
# PUBLIC API
# ============================================================

def _make_initial_state(session_id: str, question: str) -> GraphState:
    return {
        "session_id": session_id,
        "question": question,
        "standalone_query": "",
        "retrieved_context": "",
        "query_type": "",
        "docs_relevant": "",
        "sources": [],
        "needs_clarification": False,
        "clarification_question": "",
        "answer": "",
        "messages": [],
    }


def generate_chat_response(session_id: str, question: str) -> dict:
    """Non-streaming version. Returns {answer, trace}."""
    config = {"configurable": {"thread_id": session_id}}
    current_trace = []
    final_answer = ""

    for chunk in graph.stream(_make_initial_state(session_id, question), config, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            current_trace.append(_build_trace_label(node_name, state_update))
            if state_update.get("answer"):
                final_answer = state_update["answer"]

    return {"answer": final_answer, "trace": current_trace}


def stream_chat_response(session_id: str, question: str):
    """Generator for SSE streaming. Yields dicts: {trace_step, answer, done}."""
    config = {"configurable": {"thread_id": session_id}}
    final_answer = ""

    for chunk in graph.stream(_make_initial_state(session_id, question), config, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            label = _build_trace_label(node_name, state_update)
            if state_update.get("answer"):
                final_answer = state_update["answer"]
            yield {"trace_step": label, "answer": "", "done": False}

    yield {"trace_step": "", "answer": final_answer, "done": True}


# ============================================================
# INGESTION METHODS
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
