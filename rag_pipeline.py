import os
import re
import random
import boto3
import operator
from typing import TypedDict, Literal, Annotated, List, Any
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

from dotenv import load_dotenv
load_dotenv()

# --- Initialization ---
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
)

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    model_kwargs={"temperature": 0.0}
)
embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan-embed-text-v2:0"
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

memory = MemorySaver()


# ============================================================
# PYDANTIC STRUCTURED OUTPUT SCHEMAS
# ============================================================

class IntentPlan(BaseModel):
    """Classify intent and plan next step."""
    query_type: Literal["greeting", "technical"]
    needs_clarification: bool
    clarification_question: str = ""


class RelevanceGrade(BaseModel):
    """Grade whether retrieved documents are relevant to the user question."""
    relevant: bool


_llm_intent = llm.with_structured_output(IntentPlan)
_llm_grade = llm.with_structured_output(RelevanceGrade)


# ============================================================
# LANGGRAPH: State + Nodes + Router
# ============================================================

class GraphState(TypedDict):
    session_id: str
    question: str
    retrieved_context: str
    query_type: str
    docs_relevant: str
    sources: list
    needs_clarification: bool
    clarification_question: str
    clarification_rounds: int
    answer: str
    messages: Annotated[list, operator.add]


# --- Fast greeting regex ---
_SIMPLE_GREETINGS = re.compile(
    r"^(hi|hello|hey|hiya|yo|sup|morning|evening|afternoon|thanks|thank you|thanks!|bye|goodbye|see ya|how are you|howdy)([\s\.,!]*)$",
    re.IGNORECASE
)


# --- Trace label generator ---
def _build_trace_label(node_name: str, state_update: dict) -> str:
    if node_name == "classify_and_plan":
        qt = state_update.get("query_type", "")
        nc = state_update.get("needs_clarification", False)
        if qt == "greeting":
            return "classify_and_plan → greeting"
        elif nc:
            return "classify_and_plan → technical, vague → asking for clarification"
        else:
            return "classify_and_plan → technical → retrieving from Knowledge Base 🔍"
    elif node_name == "ask_clarification":
        return "ask_clarification → waiting for user to clarify"
    elif node_name == "handle_greeting":
        return "handle_greeting → casual response"
    elif node_name == "retrieve_documents":
        sources = state_update.get("sources", [])
        return f"retrieve_documents → retrieved {len(sources)} source(s) {sources} 📚"
    elif node_name == "grade_documents":
        rel = state_update.get("docs_relevant", "")
        if rel == "yes":
            return "grade_documents → chunks ARE relevant ✅ (using internal KB)"
        return "grade_documents → chunks NOT relevant ❌ (falling back to Tavily web search)"
    elif node_name == "generate_answer":
        return "generate_answer → synthesized response from internal KB 🧠"
    elif node_name == "web_search":
        answer = state_update.get("answer", "")
        if "couldn't find" in answer or "disabled" in answer:
            return "web_search → no results from Tavily ❌"
        return "web_search → answered from Tavily Web Search 🌐"
    return node_name


# --- Shared node helpers ---
def _format_history(messages: list, limit: int = 20) -> str:
    lines = []
    for msg in messages[-limit:]:
        if isinstance(msg, BaseMessage):
            lines.append(f"{msg.type.capitalize()}: {msg.content}")
        elif isinstance(msg, dict):
            lines.append(f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}")
    return "\n".join(lines) if lines else "No prior conversation."


def _respond(question: str, answer: str, **extra) -> GraphState:
    return {
        "answer": answer,
        "messages": [
            HumanMessage(content=question),
            AIMessage(content=answer),
        ],
        **extra,
    }


# ============================================================
# NODES
# ============================================================

def classify_and_plan(state: GraphState) -> GraphState:
    question = state["question"].strip()
    
    if _SIMPLE_GREETINGS.match(question):
        return {
            "query_type": "greeting",
            "needs_clarification": False,
            "clarification_question": "",
        }

    history_text = _format_history(state.get("messages", []))

    prompt = f"""You are an engineering assistant classifier and planner.

Conversation History:
{history_text}

User Message: {question}

Instructions:
1. Set query_type to "greeting" ONLY for standard salutations, small talk, or sign-offs (e.g. "hi", "hello", "thanks", "bye", "good morning").
   Set query_type to "technical" for:
   - Any engineering, infrastructure, DevOps, debugging, documentation, or product knowledge question.
   - Any question about conversation history, previous questions, user details, memory, or past context.

2. If query_type is "technical": set needs_clarification=true ONLY if it is truly impossible to search — e.g. user says "fix it" or "the error" with ZERO context and chat history provides none either.
   IMPORTANT: Questions about conversation history, memory, user details, or broad queries are NOT vague. Always set needs_clarification=false for these.

3. If needs_clarification=true: set clarification_question to one short, specific question."""

    result: IntentPlan = _llm_intent.invoke(prompt)
    if result.needs_clarification and state.get("clarification_rounds", 0) >= 1:
        return {
            "query_type": result.query_type,
            "needs_clarification": False,
            "clarification_question": "",
        }
    
    return {
        "query_type": result.query_type,
        "needs_clarification": result.needs_clarification,
        "clarification_question": result.clarification_question,
    }


def handle_greeting(state: GraphState) -> GraphState:
    question = state["question"]
    messages = state.get("messages", [])

    if not messages and _SIMPLE_GREETINGS.match(question.strip()):
        responses = [
            "Hello! I'm your engineering assistant. What technical challenge are we tackling today?",
            "Hi there! How can I help you with our infrastructure or systems today?",
            "Hey! Ready to troubleshoot or search some docs? Let me know what you need."
        ]
        answer = random.choice(responses)
    else:
        history_text = _format_history(messages)
        greeting_prompt = f"""You are a friendly engineering assistant.
The user sent a greeting or casual message. Respond warmly, concisely, and naturally.

Conversation History:
{history_text}

User Message: {question}
Response:"""

        response = llm.invoke(greeting_prompt)
        answer = response.content.strip()

    return _respond(question, answer)


def ask_clarification(state: GraphState) -> GraphState:
    question = state["question"]
    clarification = state.get("clarification_question", "Could you clarify your question?")
    return _respond(question, clarification, clarification_rounds=state.get("clarification_rounds", 0) + 1)


def retrieve_documents(state: GraphState) -> GraphState:
    question = state["question"]
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join([f"[Source: {os.path.basename(doc.metadata.get('file', 'unknown'))}]\n{doc.page_content}" for doc in retrieved_docs])

    sources = list(set(
        os.path.basename(doc.metadata.get("file", "unknown source"))
        for doc in retrieved_docs
    ))

    return {"retrieved_context": context_text, "sources": sources}


def grade_documents(state: GraphState) -> GraphState:
    context_text = state.get("retrieved_context", "")

    if len(context_text.strip()) < 50:
        return {"docs_relevant": "no"}

    question = state["question"]
    grading_prompt = f"""You are a relevance grader for an engineering knowledge base.

Determine if the retrieved documents contain information that helps answer the user's question.
Be fair — if the docs have relevant information, mark as relevant=true. Otherwise mark as relevant=false.

User Question: {question}

Retrieved Documents:
{context_text}"""

    result: RelevanceGrade = _llm_grade.invoke(grading_prompt)
    return {"docs_relevant": "yes" if result.relevant else "no"}


def generate_answer(state: GraphState) -> GraphState:
    question = state["question"]
    context_text = state["retrieved_context"]
    sources = state.get("sources", [])
    history_text = _format_history(state.get("messages", []))

    prompt = f"""You are a helpful engineering assistant.
Use the retrieved technical context and the conversation history to answer the user's question.
If you don't know the answer based on the context, say so.

Conversation History:
{history_text}

Retrieved Internal Context:
{context_text}

User Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    if sources:
        answer += f"\n\n📎 **Sources:** {', '.join(sources)}"

    return _respond(question, answer)


def web_search(state: GraphState) -> GraphState:
    """Invoked ONLY when internal retrieved chunks are NOT relevant."""
    question = state["question"]
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    if not tavily_key or tavily_key == "your_key_here":
        answer = (
            "I couldn't find relevant information in my internal knowledge base for this question, "
            "and Tavily web search is not configured (no TAVILY_API_KEY in .env). "
            "You can upload relevant documentation using the **📂 Update Knowledge Base** page."
        )
        return _respond(question, answer)

    try:
        tavily = TavilyClient(api_key=tavily_key)
        response = tavily.search(query=question, max_results=3, search_depth="basic")
        results = response.get("results", [])

        if not results:
            answer = (
                "I couldn't find relevant information in the internal knowledge base or on the web. "
                "Please try rephrasing your question or upload relevant documentation."
            )
            return _respond(question, answer)

        web_context = "\n\n".join([
            f"Title: {r.get('title', '')}\nSource: {r.get('url', '')}\n{r.get('content', '')}"
            for r in results
        ])

        web_prompt = f"""You are a helpful engineering assistant.
The user's question was not found in the internal knowledge base, so here are web search results from Tavily.
Use these results to answer the question concisely and accurately.

User Question: {question}

Web Search Results:
{web_context}

Answer:"""

        response = llm.invoke(web_prompt)
        answer = response.content.strip()

        web_sources = "\n".join([
            f"- [{r.get('title', r.get('url', ''))}]({r.get('url', '')})"
            for r in results
        ])
        answer += f"\n\n🌐 **Web Sources:**\n{web_sources}"

    except Exception as e:
        answer = f"I couldn't find relevant internal docs, and web search encountered an error: {str(e)}"

    return _respond(question, answer)


# ============================================================
# CONDITIONAL EDGES
# ============================================================

def decide_after_classify(state: GraphState) -> Literal["handle_greeting", "ask_clarification", "retrieve_documents"]:
    if state["query_type"] == "greeting":
        return "handle_greeting"
    if state.get("needs_clarification"):
        return "ask_clarification"
    return "retrieve_documents"


def decide_relevance(state: GraphState) -> Literal["generate_answer", "web_search"]:
    if state["docs_relevant"] == "yes":
        return "generate_answer"
    return "web_search"


# ============================================================
# BUILD THE GRAPH
# ============================================================

workflow = StateGraph(GraphState)

workflow.add_node("classify_and_plan", classify_and_plan)
workflow.add_node("handle_greeting", handle_greeting)
workflow.add_node("ask_clarification", ask_clarification)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("web_search", web_search)

workflow.add_edge(START, "classify_and_plan")
workflow.add_conditional_edges("classify_and_plan", decide_after_classify)
workflow.add_edge("handle_greeting", END)
workflow.add_edge("ask_clarification", END)
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
        "retrieved_context": "",
        "query_type": "",
        "docs_relevant": "",
        "sources": [],
        "needs_clarification": False,
        "clarification_question": "",
        "answer": "",
        "messages": [],
    }


def _run_graph(session_id: str, question: str):
    """Runs the graph once, yielding (trace_label, state_update) for each node as it completes."""
    config = {"configurable": {"thread_id": session_id}}
    for chunk in graph.stream(_make_initial_state(session_id, question), config, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            yield _build_trace_label(node_name, state_update), state_update


def generate_chat_response(session_id: str, question: str) -> dict:
    """Non-streaming version. Returns {answer, trace}."""
    trace = []
    final_answer = ""

    for label, state_update in _run_graph(session_id, question):
        trace.append(label)
        if state_update.get("answer"):
            final_answer = state_update["answer"]

    return {"answer": final_answer, "trace": trace}


def stream_chat_response(session_id: str, question: str):
    """Generator for SSE streaming. Yields dicts: {trace_step, token, answer, done}."""
    final_answer = ""

    for label, state_update in _run_graph(session_id, question):
        if state_update.get("answer"):
            final_answer = state_update["answer"]
        yield {"trace_step": label, "token": "", "done": False}

    for char_chunk in [final_answer[i:i+15] for i in range(0, len(final_answer), 15)]:
        yield {"trace_step": "", "token": char_chunk, "done": False}

    yield {"trace_step": "", "token": "", "done": True, "answer": final_answer}


# ============================================================
# INGESTION METHODS
# ============================================================

def _ingest_text_file(file_path: str, source_type: str, original_filename: str = None) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, "r") as f:
        text = f.read()
    source_name = original_filename or os.path.basename(file_path)
    chunks = text_splitter.create_documents(
        [text], metadatas=[{"source_type": source_type, "file": source_name}]
    )
    vectorstore.add_documents(chunks)
    return f"Successfully ingested {len(chunks)} {source_type} chunks from {source_name}."


def ingest_documentation(file_path: str, original_filename: str = None):
    return _ingest_text_file(file_path, "documentation", original_filename)


def ingest_faqs(file_path: str, original_filename: str = None):
    return _ingest_text_file(file_path, "faq", original_filename)

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