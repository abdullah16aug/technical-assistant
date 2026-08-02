# 🤖 Engineering RAG Assistant

An **Agentic RAG (Retrieval-Augmented Generation)** chatbot for engineering teams. It answers technical questions from your team's internal documentation, FAQs, and past chat logs — and falls back to live web search when the knowledge base doesn't have the answer. Built with **LangGraph**, **FastAPI**, **Streamlit**, **ChromaDB**, and **Amazon Bedrock**.

---

## 🧠 What We Built — The Full Picture

This is not a simple Q&A bot. It is a **multi-node agentic system** where every query passes through a decision-making graph before an answer is produced. Each node in the graph has a specific role:

```
User Question
      │
      ▼
┌─────────────────┐
│   route_query   │  ← LLM classifies: "greeting" or "technical"
└────────┬────────┘
         │
    ┌────┴──────┐
    ▼            ▼
greeting      technical
    │            │
    ▼            ▼
┌──────────┐  ┌─────────┐
│ handle_  │  │ planner │  ← Is the question specific enough to search?
│ greeting │  └────┬────┘
└────┬─────┘       │
     │         ┌───┴────────────┐
     │         ▼                ▼
     │    too vague          specific
     │         │                │
     │         ▼                ▼
     │  ┌──────────────┐  ┌──────────────┐
     │  │ask_clarific- │  │rewrite_query │  ← Contextualizes with chat history
     │  │ation         │  └──────┬───────┘
     │  └──────┬───────┘         ▼
     │         │          ┌──────────────────┐
     │         │          │retrieve_documents │  ← ChromaDB vector similarity search
     │         │          └──────┬───────────┘
     │         │                 ▼
     │         │          ┌──────────────┐
     │         │          │grade_documents│  ← Are docs relevant? (LLM grader)
     │         │          └──────┬───────┘
     │         │            ┌────┴──────┐
     │         │            ▼            ▼
     │         │        relevant      not relevant
     │         │            │            │
     │         │            ▼            ▼
     │         │    ┌───────────────┐  ┌──────────┐
     │         │    │generate_answer│  │web_search│  ← Tavily live web search
     │         │    └───────┬───────┘  └────┬─────┘
     ▼         ▼            ▼               ▼
    END       END          END             END
```

---

## 🔩 Detailed Breakdown of Every Component

### 1. LangGraph — The Brain / Orchestrator

**What it is:** LangGraph is a library built on top of LangChain for building stateful, multi-step workflows using a graph of nodes and edges.

**Why we use it instead of a simple chain:**
- A plain LangChain chain runs steps in a fixed order: retrieve → generate. No decisions, no branching.
- LangGraph lets us build **conditional routing**: if the question is a greeting, skip the vector DB entirely; if docs aren't relevant, search the web instead.
- It manages **state** (`GraphState`) across all nodes, so every node can read what any previous node wrote.

**How it works in our system:**
- We define a `GraphState` TypedDict with fields like `question`, `answer`, `docs_relevant`, `messages`, etc.
- Each **node** is a Python function that reads from state and returns a partial update.
- **Conditional edges** are functions that return the name of the next node to run.
- `MemorySaver` checkpointer persists conversation history across turns per `thread_id` (session).

**Alternatives:**
- **LangChain LCEL (plain chains):** Simpler but no branching or decision-making.
- **CrewAI:** Better for multi-agent teams, overkill for a single-agent RAG system.
- **AutoGen:** Microsoft's framework, heavy setup for this use case.

---

### 2. `GraphState` — Shared Memory Between Nodes

```python
class GraphState(TypedDict):
    session_id: str
    question: str               # Raw user input
    standalone_query: str       # Rewritten query for vector search
    retrieved_context: str      # Text from matched ChromaDB chunks
    query_type: str             # "greeting" or "technical"
    docs_relevant: str          # "yes" or "no" from grader
    sources: list               # Filenames of matched docs
    needs_clarification: bool   # Set by planner
    clarification_question: str # Bot's follow-up question
    answer: str                 # Final answer sent to user
    messages: Annotated[list, operator.add]  # Auto-accumulating chat history
```

**Why `Annotated[list, operator.add]` for messages?**
LangGraph merges state updates using reducer functions. Without `operator.add`, each node's return would **replace** the messages list. With it, each turn's messages get **appended** to the existing list automatically — giving us persistent memory across turns.

---

### 3. Node: `route_query` — The Gatekeeper

**What it does:** Sends the question to Claude Haiku and asks it to classify as `"greeting"` or `"technical"` in one word.

**Why this matters:** Without this, every "hi" or "thanks" would trigger a full RAG pipeline — wasting vector DB calls and making responses feel robotic.

**What happens after:**
- `greeting` → `handle_greeting` (casual, warm response, no DB hit)
- `technical` → `planner`

---

### 4. Node: `planner` — Ambiguity Detector

**What it does:** Reads the question + last 20 messages of conversation history and decides if the question is **specific enough to search**, or too vague to be useful.

**Example of what it catches:**
- ❌ `"pipeline is failing"` → vague (which pipeline? Airflow? Spark? Dataflow?)
- ✅ `"Airflow DAG failing with Cloud SQL connection timeout"` → specific

**Why this matters:** Without a planner, a vague question like "how do I fix it?" would retrieve random docs and generate a hallucinated or irrelevant answer. The planner catches this early and asks the user to clarify.

**Multi-turn awareness:** Because the planner reads `messages` (conversation history), if a user first said "Airflow" and then asks "what's the retry config?", the planner sees the context and marks it as specific — no re-asking.

**Alternatives:**
- Rule-based keyword detection (fragile, doesn't understand context)
- Always ask for clarification (annoying UX)
- Skip it entirely (random bad answers)

---

### 5. Node: `rewrite_query` — Query Contextualizer

**What it does:** Takes the user's question + conversation history and rewrites it as a **standalone search query** for ChromaDB.

**Why this matters:**
- Users ask follow-up questions: `"What about the timeout config?"` — this makes no sense without context.
- The rewriter turns it into: `"Airflow Cloud SQL connection timeout configuration"` — a complete search phrase.

**Safety check:** If the LLM returns a verbose paragraph (it sometimes does), we fall back to the original question (anything >150 chars or multi-line is rejected).

---

### 6. ChromaDB — The Vector Database

**What it is:** ChromaDB is a local, embedded vector database that stores text chunks as mathematical vectors (embeddings) and retrieves the most similar chunks given a query.

**Why we use it instead of a traditional database:**
- Traditional DBs (SQL, MongoDB) match by exact keywords. They can't understand meaning.
- ChromaDB matches by **semantic similarity** — "connection refused" will match "cannot connect to host" even though no keywords overlap.

**How ingestion works:**
1. Text is split into 500-character chunks with 50-char overlap (prevents cutting off mid-sentence).
2. Each chunk is embedded using `all-MiniLM-L6-v2` (a 384-dimension vector).
3. Chunks + metadata (filename, source type) are stored persistently in `./chroma_db/`.

**How retrieval works:**
1. The standalone query is embedded into the same vector space.
2. ChromaDB finds the top-3 most similar chunks (cosine similarity).
3. Those chunks become the `retrieved_context` passed to the LLM.

**Alternatives:**
- **Pinecone:** Managed cloud vector DB, needs API key and internet.
- **Weaviate:** Open-source but heavier to self-host.
- **pgvector:** PostgreSQL extension for vectors, good if you already use Postgres.
- **FAISS:** Facebook's library, fast but no metadata filtering or persistence out-of-box.

**Why ChromaDB here:** Zero-config, fully local, no external dependencies, persists to disk automatically.

---

### 7. HuggingFace Embeddings — `all-MiniLM-L6-v2`

**What it does:** Converts text into a 384-dimensional vector that captures semantic meaning. "Cloud SQL connection error" and "database connectivity failure" will be close together in this vector space.

**Why this model specifically:**
- Small (80MB) — loads fast, runs on CPU.
- High quality for its size — consistently ranked top-5 on MTEB (Massive Text Embedding Benchmark) for retrieval tasks.
- Free, no API key needed.

**Alternatives:**
- `text-embedding-3-small` (OpenAI) — better quality, costs money per token.
- `embed-english-v3.0` (Cohere) — excellent quality, API key required.
- `bge-large-en-v1.5` — better accuracy, ~4x larger and slower.

---

### 8. Node: `grade_documents` — The Relevance Gatekeeper

**What it does:** After retrieving 3 chunks from ChromaDB, sends them to Claude Haiku with a prompt asking: "Are these documents relevant to the question? Answer yes or no."

**Why this matters (prevents hallucination):**
- Without a grader, the LLM receives irrelevant chunks and tries to answer anyway — making up plausible-sounding but wrong answers.
- With the grader, if ChromaDB returns GCP firewall rules for a question about Kubernetes CPU limits, the grader catches the mismatch and routes to web search instead.

**What happens after:**
- `yes` → `generate_answer` (answer from your knowledge base)
- `no` → `web_search` (Tavily fallback)

---

### 9. Claude Haiku via Amazon Bedrock — The LLM

**What it is:** `anthropic.claude-3-haiku-20240307-v1:0` is Anthropic's smallest, fastest Claude model. It's accessed via AWS Bedrock — Amazon's managed LLM API gateway.

**Why Bedrock specifically:**
- If your team already uses AWS, IAM handles auth — no separate API keys.
- Bedrock gives access to multiple model providers (Anthropic, Meta, Mistral, Amazon Nova) through one API.
- Data stays in your AWS region (no data going to Anthropic's servers directly).

**Why Claude Haiku and not a bigger model:**
- The pipeline makes 4-5 LLM calls per message (classifier, planner, rewriter, grader, generator). A bigger model would be 5-10x slower and more expensive.
- Haiku is fast enough for real-time chat and smart enough for classification and grading tasks.

**Alternatives:**
- `amazon.nova-lite-v1:0` (Amazon Nova Lite) — cheaper, similar speed.
- `meta.llama3-8b-instruct-v1:0` (Llama 3) — fully open-source weights.
- `gpt-4o-mini` (OpenAI) — similar capability/price, different API.

---

### 10. Node: `web_search` — Tavily Fallback

**What it does:** When the knowledge base has no relevant documents, queries Tavily's search API, gets 3 clean result snippets, and generates an answer with the LLM using those results as context.

**Why Tavily and not Google/Bing:**
- Tavily is specifically built for LLM agents — it returns clean, pre-parsed text snippets instead of raw HTML.
- No rate limiting or bot blocking (unlike DuckDuckGo).
- Results include `title`, `url`, and `content` (already extracted, no parsing needed).
- Free tier: 1,000 searches/month.

**Why we use the original question for web search (not the rewritten standalone_query):**
- The standalone query is optimized for internal ChromaDB search — sometimes over-simplified.
- The original question has natural phrasing that search engines understand better.

**Graceful degradation:**
- If `TAVILY_API_KEY` is not set → honest message explaining web search is not configured.
- If quota is exhausted → specific message about rate limiting.
- If network error → generic error message with fallback guidance.

**Alternatives:**
- DuckDuckGo (`duckduckgo-search`) — free, no API key, but inconsistent and rate-limited.
- Google Search API (via SerpAPI) — reliable, costs $50+/month for 5,000 searches.
- Bing Search API — $7/1000 queries, more reliable than DuckDuckGo.

---

### 11. `MemorySaver` — Conversation Memory

**What it does:** LangGraph's built-in in-memory checkpointer. After each graph run, it saves the entire `GraphState` (including the `messages` list) keyed by `thread_id` (your session ID). On the next run, it loads the saved state and merges it with the new initial state.

**Why this gives us persistent memory:**
- `messages: Annotated[list, operator.add]` + MemorySaver = the bot remembers your name, context, and previous questions within the same browser session.
- If you say "my name is Rahul" in turn 1, the bot will address you as Rahul in turn 5.

**Limitation:** In-memory only. If the FastAPI server restarts, conversation history is lost. For production, replace with `SqliteSaver` or `PostgresSaver`.

---

### 12. FastAPI — The Backend API

**What it does:** Exposes the LangGraph pipeline as HTTP endpoints that the Streamlit frontend calls.

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Non-streaming: returns full `{answer, trace}` JSON |
| `POST` | `/chat/stream` | **SSE streaming**: yields trace steps live as graph executes |
| `POST` | `/ingest/documentation` | Ingest a `.md` documentation file |
| `POST` | `/ingest/faq` | Ingest a `.md` FAQ file |
| `POST` | `/ingest/chats` | Ingest chat logs (LLM extracts Q&A pairs) |

**Why we have both `/chat` and `/chat/stream`:**
- `/chat` returns everything at once after the full graph completes (~5-10 seconds).
- `/chat/stream` uses **Server-Sent Events (SSE)** — each node completion sends a `data: {...}` event to the client immediately. Streamlit consumes these to build the live trace panel.

**Why FastAPI and not Flask:**
- FastAPI has native `async` support, Pydantic validation, and automatic Swagger docs at `/docs`.
- `StreamingResponse` makes SSE trivial to implement.
- ~2-3x faster than Flask for I/O-bound workloads.

---

### 13. Streamlit — The Chat Frontend

**What it does:** Renders the chat UI with message history, the live streaming trace panel, and the knowledge base upload form.

**The streaming trace panel:**
```python
with st.status("🔍 Analyzing your question...", expanded=True) as status:
    with requests.post("/chat/stream", stream=True) as response:
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                data = json.loads(line[6:])
                if data["trace_step"]:
                    st.write(f"✅ {data['trace_step']}")   # live update!
                if data["done"]:
                    bot_answer = data["answer"]
    status.update(label="✅ Done", state="complete")
```

Each node's completion event arrives as an SSE line → Streamlit writes it to `st.status()` → user sees the trace building up in real time while waiting.

**Chat history ingestion (LLM extraction):**
When a `.txt` chat log is uploaded, the LLM doesn't just chunk it — it reads the entire chat, extracts only the **technical Q&A pairs**, and stores that curated content. This means casual greetings, off-topic messages, and repeated content are automatically filtered out.

---

## 🗺️ Complete Data Flow

```
User types: "Our Airflow DAG failed. Cloud SQL connection timeout."
                │
                ▼ POST /chat/stream (SSE)
                │
         ┌──────────────────────────────────────────────────────┐
         │                  LangGraph Graph                      │
         │                                                        │
         │  route_query ──────────────────────────► "technical"  │
         │       │                                               │
         │  planner ─────────────────────────────► "specific"    │
         │       │                                               │
         │  rewrite_query ────────────────────► "Airflow DAG     │
         │       │                              Cloud SQL timeout"│
         │       ▼                                               │
         │  retrieve_documents ───────────► top-3 chunks from    │
         │       │                          gcp.md, gcp.txt       │
         │       ▼                                               │
         │  grade_documents ──────────────────────────► "yes"    │
         │       │                                               │
         │  generate_answer ──► LLM + context + history → answer │
         └──────────────────────────────────────────────────────┘
                │
                ▼ SSE events stream to Streamlit
                │  data: {"trace_step": "route_query → technical"}
                │  data: {"trace_step": "planner → specific"}
                │  data: {"trace_step": "retrieve_documents → ..."}
                │  data: {"done": true, "answer": "Check if Cloud SQL Proxy..."}
                │
                ▼ Streamlit renders:
                   ✅ route_query → technical
                   ✅ planner → specific, proceeding to RAG
                   ✅ rewrite_query → 'Airflow DAG Cloud SQL timeout'
                   ✅ retrieve_documents → 3 source(s) ['gcp.md', 'gcp.txt']
                   ✅ grade_documents → docs relevant: yes
                   ✅ generate_answer → answered from knowledge base ✅

                   [Answer displayed]
                   📎 Sources: gcp.md, gcp.txt
```

---

## 🏗️ Project Structure

```
technical-assistant/
├── app.py              # Streamlit frontend — chat UI, live trace panel, file upload
├── main.py             # FastAPI backend — REST + SSE endpoints, Pydantic models
├── rag_pipeline.py     # Core agentic RAG engine (LangGraph graph, all 9 nodes)
├── inspect_db.py       # Utility to inspect ChromaDB chunk count and run test queries
├── pyproject.toml      # Dependencies managed by uv
├── .env                # Secrets: AWS credentials, TAVILY_API_KEY (not committed)
├── chroma_db/          # ChromaDB persistent vector storage (auto-created)
└── data/
    ├── docs/           # Technical documentation (.md)
    ├── faqs/           # FAQ files (.md)
    └── chats/          # Team chat history exports (.txt)
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **LLM** | Claude Haiku via AWS Bedrock | Fast, cheap, smart enough for classification + generation |
| **Agentic Orchestration** | LangGraph | Conditional routing, stateful nodes, multi-turn memory |
| **Vector Store** | ChromaDB | Local, zero-config, semantic similarity search |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Small, fast, high-quality, free |
| **Web Search** | Tavily API | LLM-optimized search, clean results, no rate limiting |
| **Backend** | FastAPI | Async, SSE streaming, automatic Swagger docs |
| **Frontend** | Streamlit | Rapid UI, native chat components, SSE consumption |
| **Memory** | LangGraph MemorySaver | In-memory per-session conversation history |
| **Package Manager** | uv | Fast Python package management |

---

## 📋 Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- AWS credentials with access to Amazon Bedrock (`ap-south-1` or your region)
- Tavily API key — free at [tavily.com](https://tavily.com) (1,000 searches/month)

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd technical-assistant
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-south-1

TAVILY_API_KEY=your_tavily_key
```

---

## ▶️ Running the Application

You need **two terminal windows**.

### Terminal 1 — FastAPI Backend

```bash
uv run uvicorn main:app --reload --port 8000
```

API available at: `http://127.0.0.1:8000`
Swagger docs at: `http://127.0.0.1:8000/docs`

### Terminal 2 — Streamlit Frontend

```bash
uv run streamlit run app.py
```

UI opens at: `http://localhost:8501`

> **Always start the FastAPI backend before Streamlit.**

---

## 📂 Populating the Knowledge Base

Use the **"📂 Update Knowledge Base"** page in the Streamlit sidebar:

| Category | Format | What happens |
|----------|--------|-------------|
| **Documentation** | `.md` | Split into chunks → embedded → stored in ChromaDB |
| **FAQs** | `.md` | Same as documentation |
| **Chat History** | Any text | LLM reads the full chat → extracts only technical Q&A pairs → stored as a single curated document |

---

## 📡 API Reference

### `POST /chat` — Non-streaming

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "question": "How do I fix Airflow connection timeout?"}'
```

Response:
```json
{
  "answer": "Check if the Cloud SQL Proxy is running...",
  "trace": [
    "route_query → technical",
    "planner → specific, proceeding to RAG",
    "retrieve_documents → 3 source(s) ['gcp.md']",
    "grade_documents → docs relevant: yes",
    "generate_answer → answered from knowledge base ✅"
  ]
}
```

### `POST /chat/stream` — SSE Streaming

Each node completion yields a `data:` event:
```
data: {"trace_step": "route_query → technical", "answer": "", "done": false}
data: {"trace_step": "planner → specific, proceeding to RAG", "answer": "", "done": false}
...
data: {"trace_step": "", "answer": "Check if the Cloud SQL Proxy...", "done": true}
```

---

## 🔍 Inspecting the Vector Database

```bash
uv run python inspect_db.py
```

Shows total chunk count and lets you run a test similarity query interactively.

---

## 🗺️ Architecture Decision: Why Agentic RAG vs Plain RAG?

| | Plain RAG | Agentic RAG (this system) |
|--|-----------|--------------------------|
| Routing | Always retrieves | Routes greetings away from DB |
| Ambiguity | Retrieves random docs | Planner asks for clarification |
| Bad retrievals | Hallucinates | Grader detects and routes to web search |
| Memory | Stateless | MemorySaver persists across turns |
| Trace | Black box | Full node execution trace visible in UI |
| Web fallback | ❌ | ✅ Tavily search when KB has no answer |
