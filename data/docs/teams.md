# TeamAssistant Product Knowledge Base

## Product Overview

**TeamAssistant** is an AI-powered enterprise knowledge assistant designed to help engineering teams quickly retrieve information from technical documentation, FAQs, and historical Microsoft Teams conversations.

The application uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses while minimizing hallucinations.

---

# Product Features

## AI Knowledge Search

* Semantic search across internal documentation
* Natural language question answering
* Context-aware responses
* Source citation support

---

## Multi-Source Retrieval

The assistant retrieves knowledge from:

* Technical Documentation
* Frequently Asked Questions
* Microsoft Teams Chat History

The system combines information from multiple sources before generating the final response.

---

## Conversation Memory

The assistant supports conversational context.

Example:

User:
How do I optimize Airflow?

User:
What about Spark?

The assistant understands that the follow-up question is related to the previous discussion.

---

## Web Search Fallback

If relevant information is unavailable in the internal knowledge base:

1. Search the internal knowledge base
2. Evaluate document relevance
3. Perform web search if necessary
4. Generate an answer with web sources

---

# System Architecture

Components

* Streamlit UI
* LangGraph Workflow
* LangChain
* AWS Bedrock
* Amazon Titan Embeddings
* ChromaDB
* DuckDuckGo (Web Fallback)

---

# Authentication

Authentication uses Microsoft Entra ID.

User Roles

* Administrator
* Data Engineer
* Viewer

Permissions

Administrator

* Manage users
* Upload documents
* Delete documents
* Configure settings

Data Engineer

* Ask questions
* Upload documents
* View chat history

Viewer

* Read-only access

---

# Document Management

Supported Formats

* PDF
* Markdown
* Text
* Word Documents

Maximum upload size

* 20 MB per file

Uploaded documents are automatically:

* Chunked
* Embedded
* Indexed
* Stored in ChromaDB

---

# Knowledge Ingestion Pipeline

1. Upload document
2. Extract text
3. Split into chunks
4. Generate embeddings
5. Store vectors
6. Ready for retrieval

---

# Search Pipeline

1. Receive user question
2. Rewrite query
3. Retrieve relevant documents
4. Grade document relevance
5. Generate answer
6. Return sources

---

# Agent Workflow

Execution Flow

START

↓

Intent Analysis

↓

Planner

↓

Query Rewriting

↓

Knowledge Retrieval

↓

Document Grading

↓

Response Generation

↓

END

---

# API Endpoints

### POST /ask

Submit a user question.

Request

```json
{
  "question": "How does authentication work?"
}
```

Response

```json
{
  "answer": "...",
  "sources": [
    "docs.md",
    "faq.md"
  ]
}
```

---

### POST /upload

Uploads new documents into the knowledge base.

Supported file types

* pdf
* md
* txt
* docx

---

### GET /health

Returns application health status.

Response

```json
{
    "status":"healthy"
}
```

---

# Performance

Average Response Time

* Knowledge Base Search: < 1 second
* End-to-End Response: 2–4 seconds

Maximum Context Chunks

* Top 5 retrieved chunks

Embedding Model

* Amazon Titan Text Embeddings

Generation Model

* Claude via AWS Bedrock

---

# Security

* IAM-based access control
* HTTPS encryption
* Role-based authorization
* Secrets stored securely
* No credentials stored in source code

---

# Troubleshooting

## No answer found

Possible reasons

* Document not uploaded
* Incorrect query
* Low document relevance

Solution

* Upload latest documentation
* Rephrase the question
* Enable web search fallback

---

## Slow responses

Possible causes

* Large document uploads
* High retrieval latency
* External web search

Solution

* Optimize chunk size
* Reduce Top-K
* Cache embeddings

---

# Product Limitations

* Images are not indexed
* Scanned PDFs require OCR
* Internet search depends on external providers
* Supports English documentation only

---

# Future Enhancements

* Microsoft Teams API integration
* SharePoint integration
* Jira integration
* Confluence integration
* Long-term memory
* Multi-agent orchestration
* Self-RAG
* Corrective RAG
* Human approval workflow
