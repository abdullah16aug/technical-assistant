# 🤖 TeamAssistant — Production Agentic RAG & Knowledge Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![AWS Bedrock](https://img.shields.io/badge/LLM-AWS%20Bedrock-yellow.svg)](https://aws.amazon.com/bedrock/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)

**TeamAssistant** is an enterprise-ready, stateful **Agentic Retrieval-Augmented Generation (RAG)** platform engineered to assist technical teams with product documentation, FAQs, and incident resolution. Built on a modular **LangGraph** state graph, it combines dynamic intent classification, multi-turn dialogue planning, semantic vector search, corrective document grading (Self-RAG / Corrective RAG), live web search fallbacks, and real-time two-phase SSE token streaming.

---

## 📌 Table of Contents
- [Problem Statement & Solution](#-problem-statement--solution)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [LangGraph State Machine Workflow](#-langgraph-state-machine-workflow)
- [Project Directory Structure](#-project-directory-structure)
- [Tech Stack & Justification](#-tech-stack--justification)
- [End-to-End Execution Flow](#-end-to-end-execution-flow)
- [Knowledge Ingestion Engine](#-knowledge-ingestion-engine)
- [API Reference](#-api-reference)
- [Getting Started](#-getting-started)
- [Future Engineering Roadmap](#-future-engineering-roadmap)

---

## 💡 Problem Statement & Solution

### The Challenge
Standard RAG pipelines (Naive RAG) suffer from critical production limitations:
1. **High Latency & Costs:** Every query (even a simple "hello") triggers expensive vector database lookups and full LLM generation loops.
2. **Hallucinations on Out-of-Domain Queries:** When the vector DB retrieves irrelevant context, LLMs generate plausible but incorrect answers.
3. **Ambiguous User Queries:** Queries like *"how do I fix the error?"* lack context and cause Naive RAG systems to fetch random irrelevant documents.
4. **Lack of Conversational Context:** Naive chains lack state management, failing to resolve co-references in multi-turn dialogues (*"How do I configure it?"*).

### The Solution
**TeamAssistant** addresses these vulnerabilities using an **Agentic RAG State Graph**:
- **Intent Consolidation & Fast-Pathing:** Single-pass structured output classification (`classify_and_plan`) with regex fast-pathing to eliminate redundant LLM hops.
- **Ambiguity Guardrails:** Detects vague queries before hitting the vector DB and asks targeted clarifying questions.
- **Corrective RAG & Web Fallback:** Evaluates document relevance heuristically and via LLM grading (`grade_documents`). Automatically falls back to Tavily live web search if internal knowledge is missing.
- **Two-Phase SSE Token Streaming:** Streams node execution traces in Phase 1, followed by word-by-word LLM token generation in Phase 2 for a seamless user experience.

---

## ✨ Key Features

- 🧠 **Agentic State Machine:** Multi-step graph powered by LangGraph with `MemorySaver` checkpointer for stateful multi-turn history.
- ⚡ **Low-Latency Architecture:** Consolidated classification node using Pydantic structured output (`with_structured_output`), reducing Bedrock round-trips by 50%.
- 🛡️ **Self-Corrective Quality Control:** Automated relevance grading prevents context pollution and hallucinated responses.
- 🌐 **Graceful External Web Search:** Seamless integration with Tavily AI Search for out-of-domain technical queries.
- 📥 **Multi-Format Technical Knowledge Ingestion:** Automated chunking and LLM-assisted curation of Markdown docs, FAQs, and raw chat exports.
- 🚀 **Asynchronous Two-Phase Streaming API:** SSE-powered FastAPI backend delivering real-time execution trace badges + LLM token streaming to Streamlit UI.

---

## 🏗️ System Architecture

```
                               ┌──────────────────────────┐
                               │   Streamlit Frontend     │
                               │   (Chat UI & Streaming)  │
                               └────────────┬─────────────┘
                                            │ HTTP POST /chat/stream (SSE)
                                            ▼
                               ┌──────────────────────────┐
                               │   FastAPI Backend API    │
                               │  (Async SSE Controller)  │
                               └────────────┬─────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 LangGraph StateGraph                                   │
│                                                                                        │
│     ┌─────────────────────┐                                                            │
│     │  classify_and_plan  │ ──► [Greeting] ──► handle_greeting ──► END                │
│     └──────────┬──────────┘                                                            │
│                │                                                                       │
│                ├──────────────► [Vague Query] ──► ask_clarification ──► END            │
│                │                                                                       │
│                ▼ [Specific Technical Query]                                            │
│     ┌─────────────────────┐                                                            │
│     │    rewrite_query    │ (Contextualizes multi-turn standalone terms)               │
│     └──────────┬──────────┘                                                            │
│                ▼                                                                       │
│     ┌─────────────────────┐      ┌──────────────────────────┐                          │
│     │ retrieve_documents  │ ───► │ ChromaDB Vector Storage  │                          │
│     └──────────┬──────────┘      └──────────────────────────┘                          │
│                ▼                                                                       │
│     ┌─────────────────────┐                                                            │
│     │   grade_documents   │                                                            │
│     └──────────┬──────────┘                                                            │
│                │                                                                       │
│                ├──────────────► [Relevant: YES] ──► generate_answer (KB Context)       │
│                │                                          │                            │
│                ▼ [Relevant: NO / Empty Context]           ▼                            │
│     ┌─────────────────────┐                           ┌──────┐                         │
│     │     web_search      │ ────────────────────────► │ END  │                         │
│     │   (Tavily Fallback) │                           └──────┘                         │
│     └─────────────────────┘                                                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 LangGraph State Machine Workflow

The graph operates over a central state schema defined in `rag_pipeline.py`:

```python
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
    answer_prompt: str
    answer: str
    messages: Annotated[list, operator.add]
```

### Node Execution Responsibilities

| Node Name | Input State Keys | Output State Updates | Architectural Purpose |
|---|---|---|---|
| `classify_and_plan` | `question`, `messages` | `query_type`, `needs_clarification`, `clarification_question` | Single-pass classification & ambiguity detection via Pydantic structured output. |
| `handle_greeting` | `question`, `messages` | `answer`, `messages` | Generates personalized responses using LLM history or regex fast-path. |
| `ask_clarification` | `question`, `clarification_question` | `answer`, `clarification_rounds`, `messages` | Prompts user for missing details; bounded by hard cap. |
| `rewrite_query` | `question`, `messages` | `standalone_query` | Transforms conversational queries into standalone search terms. |
| `retrieve_documents` | `standalone_query` | `retrieved_context`, `sources` | Performs MMR similarity search against local ChromaDB instance. |
| `grade_documents` | `question`, `retrieved_context` | `docs_relevant` | Self-RAG relevance check with empty-context fast heuristic. |
| `generate_answer` | `question`, `retrieved_context`, `messages` | `answer_prompt`, `sources` | Formulates final synthesis prompt for external token streaming. |
| `web_search` | `question` | `answer`, `messages` | Fallback search engine via Tavily API when KB coverage fails. |

---

## 📁 Project Directory Structure

```
technical-assistant/
├── app.py              # Streamlit Web UI (Chat layout, streaming handler, KB management)
├── main.py             # FastAPI REST Server (Asynchronous SSE & HTTP endpoints)
├── rag_pipeline.py     # Core LangGraph pipeline, node logic, and vector store operations
├── inspect_db.py       # Developer utility for querying & auditing ChromaDB collections
├── pyproject.toml      # Dependency & package configuration (managed via uv)
├── .env                # Environment variables (AWS Bedrock region, Tavily API keys)
├── chroma_db/          # Persistent local ChromaDB database directory
└── data/
    ├── docs/           # Technical product documentation (.md)
    ├── faqs/           # Product FAQ reference files (.md)
    └── chats/          # Slack/Teams raw incident chat logs (.txt)
```

---

## 🛠️ Tech Stack & Justification

| Technology | Selection Rationale | Production Alternatives Considered |
|---|---|---|
| **LangGraph** | Provides explicit state graphs, conditional branching, and checkpoint persistence required for agentic loops. | **LangChain LCEL** (No state graphs), **CrewAI** (Heavy multi-agent overhead). |
| **AWS Bedrock (Claude Haiku)** | Enterprise security, high throughput, low latency (~300ms per call), low cost. | **OpenAI Direct API** (Data privacy limits), **Local Llama-3** (Infra overhead). |
| **ChromaDB** | Embedded, zero-config vector store with native metadata filtering and persistence. | **Pinecone** (SaaS lock-in & cost), **Milvus** (Overkill for single-node deployment). |
| **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) | High semantic accuracy (384-dim), fast CPU inference, $0 operational cost. | **OpenAI text-embedding-3** (Network latency + per-token charge). |
| **FastAPI** | Native `async` support, SSE stream response capabilities, automatic Pydantic schema validation. | **Flask** (Synchronous, lacks native SSE stream handling). |
| **Streamlit** | Rapid Python-native UI prototyping with native chat UI components and reactive session state. | **React / Next.js** (High frontend development overhead). |
| **Tavily AI Search** | Tailored specifically for LLM search agents; returns clean pre-parsed content snippets. | **DuckDuckGo** (Frequent rate limits), **Google Custom Search** (High cost). |

---

## ⚡ End-to-End Execution Flow

### 1. Two-Phase SSE Streaming Request Cycle
When a user submits a question in Streamlit:
1. **HTTP Connection:** Streamlit issues a `POST` request to `/chat/stream` with `session_id` and `question`.
2. **Phase 1 — Graph Event Streaming:** As LangGraph executes nodes (`classify_and_plan` → `rewrite_query` → `retrieve_documents` → `grade_documents`), FastAPI streams trace events immediately to Streamlit, rendering live progress updates inside an `st.status()` container.
3. **Phase 2 — Token Streaming:** When `generate_answer` completes, `stream_chat_response()` invokes `llm.stream(answer_prompt)` and yields tokens directly to Streamlit, which updates a live markdown text placeholder token-by-token.

---

## 📥 Knowledge Ingestion Engine

The system supports 3 distinct ingestion modes:
1. **Technical Documentation (`ingest_documentation`):** Uses `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 50) and tags vectors with `source_type="documentation"`.
2. **Product FAQs (`ingest_faqs`):** Splitted and indexed with metadata `source_type="faq"`.
3. **Raw Team Chat Logs (`ingest_chats`):** Pre-processed via LLM structured extraction to transform noisy incident conversations into structured `Question:` and `Verified Answer:` pairs prior to vector indexing.

---

## 🔌 API Reference

### POST `/chat/stream`
Processes a message through the LangGraph engine and streams execution state and tokens using Server-Sent Events (SSE).

**Request Body:**
```json
{
  "session_id": "user_session_123",
  "question": "How do I configure the timeout for GCP Cloud SQL?"
}
```

**SSE Event Response Format:**
```http
data: {"trace_step": "classify_and_plan → technical, specific", "token": "", "done": false}
data: {"trace_step": "rewrite_query → 'GCP Cloud SQL timeout configuration'", "token": "", "done": false}
data: {"trace_step": "retrieve_documents → 2 source(s) ['gcp.md']", "token": "", "done": false}
data: {"trace_step": "generate_answer → streaming answer... 🔄", "token": "", "done": false}
data: {"trace_step": "", "token": "To", "done": false}
data: {"trace_step": "", "token": " configure", "done": false}
...
data: {"trace_step": "", "token": "", "done": true, "answer": "To configure..."}
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- AWS Account configured with Bedrock access (`ap-south-1` recommended)
- Tavily Search API key

### Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/technical-assistant.git
   cd technical-assistant
   ```

2. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   AWS_DEFAULT_REGION=ap-south-1
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   TAVILY_API_KEY=tvly-your_tavily_key
   ```

3. **Install Dependencies:**
   ```bash
   uv sync
   ```

4. **Launch Application Servers:**
   - **Backend API Server (FastAPI):**
     ```bash
     uv run uvicorn main:app --reload --port 8000
     ```
   - **Frontend UI Server (Streamlit):**
     ```bash
     uv run streamlit run app.py
     ```

---

## 🛣️ Future Engineering Roadmap

- [ ] **SQL Checkpointer Migration:** Replace `MemorySaver` with `SqliteSaver` / `PostgresSaver` for cross-session persistent storage.
- [ ] **Self-Reflection & Hallucination Guard:** Integrate an additional node to verify answer consistency against context before output rendering.
- [ ] **Hybrid Search:** Combine ChromaDB dense vector search with sparse BM25 keyword matching using Reciprocal Rank Fusion (RRF).
- [ ] **Human-in-the-Loop Approval:** Add LangGraph `interrupt` hooks for high-risk system commands or ticket creations.
M reads the full chat → extracts only technical Q&A pairs → stored as a single curated document |

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
