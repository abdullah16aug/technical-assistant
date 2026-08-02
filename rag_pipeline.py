import os
import re
import random
import boto3
import operator
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel
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
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    model_kwargs={"temperature": 0.0}
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    """Classify intent and plan next step — replaces route_query + planner in one LLM call."""
    query_type: Literal["greeting", "technical"]
    needs_clarification: bool
    clarification_question: str = ""


class RelevanceGrade(BaseModel):
    """Grade whether retrieved documents are relevant to the question."""
    relevant: bool


# Structured output LLMs — bound once at module load, reused per call
_llm_intent = llm.with_structured_output(IntentPlan)
_llm_grade = llm.with_structured_output(RelevanceGrade)


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
    clarification_rounds: int 
    answer_prompt: str                          # built by generate_answer; streamed externally
    answer: str
    messages: Annotated[list, operator.add]     # accumulates conversation history


# --- Fast greeting regex (zero LLM cost for simple greetings) ---
_SIMPLE_GREETINGS = re.compile(
    r"^(hi|hello|hey|hiya|yo|sup|morning|evening|afternoon|thanks|thank you|thanks!|bye|goodbye|see ya|how are you|howdy)([\s\.,!]*)$",
    re.IGNORECASE
)


# --- Helper: build readable trace label from stream chunk ---
def _build_trace_label(node_name: str, state_update: dict) -> str:
    """Builds a human-readable trace label from node name and state update."""
    if node_name == "classify_and_plan":
        qt = state_update.get("query_type", "")
        nc = state_update.get("needs_clarification", False)
        if qt == "greeting":
            return "classify_and_plan → greeting"
        elif nc:
            return "classify_and_plan → technical, vague → asking for clarification"
        else:
            return "classify_and_plan → technical, specific → proceeding to RAG"
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
        return "generate_answer → streaming answer... 🔄"
    elif node_name == "web_search":
        answer = state_update.get("answer", "")
        if "couldn't find" in answer:
            return "web_search → no results from Tavily ❌"
        return "web_search → answered from Tavily 🌐"
    elif node_name == "handle_greeting":
        return "handle_greeting → casual response"
    return node_name


# ============================================================
# NODES
# ============================================================

# --- Node 1: classify_and_plan — MERGED route_query + planner (1 LLM call instead of 2) ---
def classify_and_plan(state: GraphState) -> GraphState:
    """Single structured-output LLM call that classifies query type AND decides
    if clarification is needed. Uses Pydantic tool calling — no string parsing.
    Fast-path: simple greetings matched by regex before any Bedrock call.
    """
    question = state["question"].strip()
    
    # Fast path: regex match for common single-phrase greetings (zero LLM cost)
    if _SIMPLE_GREETINGS.match(question):
        return {
            "query_type": "greeting",
            "needs_clarification": False,
            "clarification_question": "",
        }

    messages = state.get("messages", [])[-20:]
    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    ) if messages else "No prior conversation."

    prompt = f"""You are an engineering assistant classifier and planner.

Conversation History:
{history_text}

User Message: {question}

Instructions:
1. Set query_type to "greeting" for: casual talk, hello, hi, how are you, thanks, bye, personal introductions, or any non-technical conversation.
   Set query_type to "technical" for: any engineering, infrastructure, DevOps, debugging, documentation, or product knowledge question.

2. If query_type is "technical": set needs_clarification=true ONLY if it is truly impossible to search — e.g. user says "fix it" or "the error" with ZERO context and chat history provides none either.
   IMPORTANT: Broad questions like "tell me about the product", "what are common challenges", "retrieve FAQs", "what do we use X for" are NOT vague. Always set needs_clarification=false for these.

3. If needs_clarification=true: set clarification_question to one short, specific question."""

    result: IntentPlan = _llm_intent.invoke(prompt)
    # Hard cap: never ask more than once per conversation thread
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


# --- Node 2: Handle greeting (zero LLM cost — pre-canned responses) ---
def handle_greeting(state: GraphState) -> GraphState:
    question = state["question"].lower()

    if "thank" in question:
        responses = [
            "You're very welcome! Let me know if you need help with anything else.",
            "No problem at all! Any other technical issues I can look into?",
            "Happy to help! What's next?"
        ]
    elif "bye" in question or "see ya" in question or "goodbye" in question:
        responses = [
            "Goodbye! Have a great day.",
            "Catch you later! Let me know when you need help again.",
            "Bye! I'll be here if anything breaks."
        ]
    else:
        responses = [
            "Hello! I'm your engineering assistant. What technical challenge are we tackling today?",
            "Hi there! How can I help you with our infrastructure or systems today?",
            "Hey! Ready to troubleshoot or search some docs? Let me know what you need."
        ]

    answer = random.choice(responses)
    return {
        "answer": answer,
        "messages": [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer},
        ],
    }


# --- Node 3: Ask clarification ---
def ask_clarification(state: GraphState) -> GraphState:
    question = state["question"]
    clarification = state.get("clarification_question", "Could you clarify your question?")

    return {
        "answer": clarification,
        "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": clarification},
        ],
    }


# --- Node 4: Rewrite query ---
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

    # Safety: if the LLM returns verbose text, fall back to original question
    if len(standalone) > 150 or "\n" in standalone:
        standalone = question

    return {"standalone_query": standalone}


# --- Node 5: Retrieve documents ---
def retrieve_documents(state: GraphState) -> GraphState:
    standalone_query = state["standalone_query"]
    retrieved_docs = retriever.invoke(standalone_query)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    sources = list(set(
        os.path.basename(doc.metadata.get("file", "unknown source"))
        for doc in retrieved_docs
    ))

    return {"retrieved_context": context_text, "sources": sources}


# --- Node 6: Grade retrieved documents ---
# Uses structured output (tool calling) for reliable boolean — no fragile string parsing.
# Pre-filter heuristic: skip LLM entirely when context is empty.
def grade_documents(state: GraphState) -> GraphState:
    context_text = state["retrieved_context"]

    # Fast heuristic: empty retrieval is irrelevant (saves one full LLM round-trip)
    if len(context_text.strip()) < 50:
        return {"docs_relevant": "no"}

    question = state["question"]
    grading_prompt = f"""You are a relevance grader for an engineering knowledge base.

Determine if the retrieved documents contain information that helps answer the user's question.
Be generous — if the docs have ANY relevant information, mark as relevant=true.

User Question: {question}

Retrieved Documents:
{context_text}"""

    result: RelevanceGrade = _llm_grade.invoke(grading_prompt)
    return {"docs_relevant": "yes" if result.relevant else "no"}


# --- Node 7: Build answer prompt (LLM call deferred to streaming layer) ---
# This node does NOT call the LLM. It builds and stores the prompt in state.
# stream_chat_response() streams the tokens; generate_chat_response() calls LLM directly.
def generate_answer(state: GraphState) -> GraphState:
    question = state["question"]
    context_text = state["retrieved_context"]
    sources = state.get("sources", [])
    messages = state.get("messages", [])[-20:]

    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    )

    prompt = f"""You are a helpful engineering assistant.
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

    # Store prompt + sources in state; the actual LLM call happens outside the graph
    return {
        "answer_prompt": prompt,
        "sources": sources,
    }


# --- Node 8: Web Search fallback (Tavily) ---
def web_search(state: GraphState) -> GraphState:
    """Falls back to Tavily web search when KB has no relevant docs."""
    question = state["question"]
    tavily_key = os.getenv("TAVILY_API_KEY", "")

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
            search_depth="basic",
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

        web_sources = "\n".join([
            f"- [{r.get('title', r.get('url', ''))}]({r.get('url', '')})"
            for r in results
        ])
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


# ============================================================
# CONDITIONAL EDGES
# ============================================================

def decide_after_classify(state: GraphState) -> Literal["handle_greeting", "ask_clarification", "rewrite_query"]:
    """Single routing function — replaces the old decide_route + decide_after_planner pair."""
    if state["query_type"] == "greeting":
        return "handle_greeting"
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

workflow.add_node("classify_and_plan", classify_and_plan)
workflow.add_node("handle_greeting", handle_greeting)
workflow.add_node("ask_clarification", ask_clarification)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("web_search", web_search)

workflow.add_edge(START, "classify_and_plan")
workflow.add_conditional_edges("classify_and_plan", decide_after_classify)
workflow.add_edge("handle_greeting", END)
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
        "answer_prompt": "",
        "answer": "",
        "messages": [],
    }


def generate_chat_response(session_id: str, question: str) -> dict:
    """Non-streaming version. Returns {answer, trace}."""
    config = {"configurable": {"thread_id": session_id}}
    current_trace = []
    final_answer = ""
    answer_prompt = ""
    sources = []

    for chunk in graph.stream(_make_initial_state(session_id, question), config, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            current_trace.append(_build_trace_label(node_name, state_update))
            if node_name == "generate_answer":
                answer_prompt = state_update.get("answer_prompt", "")
                sources = state_update.get("sources", [])
            elif state_update.get("answer"):
                final_answer = state_update["answer"]

    # generate_answer path: call LLM directly (no streaming in this code path)
    if answer_prompt:
        response = llm.invoke(answer_prompt)
        final_answer = response.content.strip()
        if sources:
            final_answer += f"\n\n📎 **Sources:** {', '.join(sources)}"
        graph.update_state(config, {
            "answer": final_answer,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_answer},
            ],
        })

    return {"answer": final_answer, "trace": current_trace}


def stream_chat_response(session_id: str, question: str):
    """Generator for SSE streaming. Yields dicts: {trace_step, token, answer, done}.

    Two-phase streaming:
      Phase 1 — Stream graph node execution labels in real-time.
      Phase 2 — Stream generate_answer LLM output token-by-token via llm.stream().

    Non-KB paths (greeting, clarification, web_search) skip Phase 2 since their
    answers are already complete when the node finishes.
    """
    config = {"configurable": {"thread_id": session_id}}
    answer_prompt = ""
    sources = []
    final_answer = ""

    # --- Phase 1: Trace streaming ---
    for chunk in graph.stream(_make_initial_state(session_id, question), config, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            label = _build_trace_label(node_name, state_update)
            if node_name == "generate_answer":
                answer_prompt = state_update.get("answer_prompt", "")
                sources = state_update.get("sources", [])
            elif state_update.get("answer"):
                final_answer = state_update["answer"]
            yield {"trace_step": label, "token": "", "done": False}

    # --- Phase 2: Token streaming (generate_answer path only) ---
    if answer_prompt:
        for token_chunk in llm.stream(answer_prompt):
            token = token_chunk.content
            if token:
                final_answer += token
                yield {"trace_step": "", "token": token, "done": False}

        if sources:
            source_text = f"\n\n📎 **Sources:** {', '.join(sources)}"
            final_answer += source_text
            yield {"trace_step": "", "token": source_text, "done": False}

        # Persist the streamed answer to MemorySaver for conversation continuity
        graph.update_state(config, {
            "answer": final_answer,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_answer},
            ],
        })

    yield {"trace_step": "", "token": "", "done": True, "answer": final_answer}


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
