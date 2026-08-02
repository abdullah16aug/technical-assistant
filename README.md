# README.md — Copy-paste this into your README

```markdown
# 🤖 Engineering RAG Assistant

An intelligent technical support chatbot powered by **LangGraph**, **FastAPI**, **Streamlit**, and **ChromaDB**. It uses Retrieval-Augmented Generation (RAG) to answer engineering questions from your team's documentation, FAQs, and past chat logs.

## 🧠 How It Works

The assistant uses a **LangGraph StateGraph** to route incoming queries:

```
User Query
    │
    ▼
┌──────────────┐
│  route_query  │ ← LLM classifies: "greeting" or "technical"
└──────┬───────┘
       │
  ┌────┴─────┐
  ▼          ▼
greeting   technical
  │          │
  ▼          ▼
┌────────┐  ┌──────────────┐
│ handle │  │ rewrite_query │ ← Contextualizes with chat history
│greeting│  └──────┬───────┘
└───┬────┘         ▼
    │        ┌─────────────────┐
    │        │retrieve_documents│ ← ChromaDB similarity search
    │        └──────┬──────────┘
    │               ▼
    │        ┌───────────────┐
    │        │generate_answer │ ← LLM response with context
    │        └──────┬────────┘
    ▼               ▼
   END             END
```

- **Greetings** (hi, thanks, bye) get a friendly response without hitting the vector DB.
- **Technical queries** go through the full RAG pipeline: rewrite → retrieve → generate.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Amazon Nova Lite (via AWS Bedrock) |
| **Orchestration** | LangGraph |
| **Vector Store** | ChromaDB |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Backend API** | FastAPI |
| **Frontend** | Streamlit |
| **Package Manager** | uv |

## 📋 Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **AWS credentials** configured with access to Amazon Bedrock (`us-east-1`)
  ```bash
  aws configure
  ```

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

Create a `.env` file in the project root (or update the existing one):

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

## ▶️ Running the Application

You need **two terminal windows** — one for the backend, one for the frontend.

### Terminal 1: Start the FastAPI Backend

```bash
uv run uvicorn main:app --reload --port 8000
```

The API server will be available at: **http://127.0.0.1:8000**

You can verify it's running by visiting: http://127.0.0.1:8000/docs (Swagger UI)

### Terminal 2: Start the Streamlit Frontend

```bash
uv run streamlit run app.py
```

The UI will open automatically at: **http://localhost:8501**

> **Important:** Always start the FastAPI backend **before** the Streamlit frontend. The frontend makes HTTP calls to the backend on port 8000.

## 📂 Populating the Knowledge Base

Use the **"📂 Update Knowledge Base"** page in the Streamlit sidebar to upload files:

| Data Category | File Format | Description |
|--------------|-------------|-------------|
| **Documentation** | `.md` | Technical docs, runbooks, system architecture notes |
| **FAQs** | `.md` | Frequently asked questions in markdown format |
| **Chat History** | Any text | Raw team chat logs — the LLM extracts Q&A pairs automatically |

Sample data files can be placed in:
```
data/
├── docs/       # Technical documentation (.md)
├── faqs/       # FAQ files (.md)
└── chats/      # Chat history exports
```

## 🔍 Inspecting the Vector Database

To check what's stored in ChromaDB and test similarity search:

```bash
uv run python inspect_db.py
```

This will show total chunk count and let you run a test query interactively.

## 📁 Project Structure

```
technical-assistant/
├── app.py              # Streamlit frontend (chat UI + file upload)
├── main.py             # FastAPI backend (REST API endpoints)
├── rag_pipeline.py     # Core RAG engine with LangGraph query routing
├── inspect_db.py       # Utility to inspect ChromaDB contents
├── pyproject.toml      # Project dependencies (managed by uv)
├── .env                # AWS credentials (not committed)
├── chroma_db/          # ChromaDB persistent storage
├── data/               # Sample data files
│   ├── docs/           #   Technical documentation
│   ├── faqs/           #   FAQ files
│   └── chats/          #   Chat history logs
└── workspace.ipynb     # Development notebook
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a question, get a RAG-powered answer |
| `POST` | `/ingest/documentation` | Ingest a markdown documentation file |
| `POST` | `/ingest/faq` | Ingest a markdown FAQ file |
| `POST` | `/ingest/chats` | Ingest chat logs (LLM extracts Q&A pairs) |

### Example: Chat request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session", "question": "How do I restart the nginx service?"}'
```
```
