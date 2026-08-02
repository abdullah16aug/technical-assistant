import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import rag_pipeline
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Engineering RAG API", version="1.0")


class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    trace: List[str] = []

class IngestionRequest(BaseModel):
    file_path: str
    original_filename: Optional[str] = None

class IngestionResponse(BaseModel):
    status: str
    message: str
    extracted_data: Optional[str] = None


# Non-streaming chat (kept for compatibility)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = rag_pipeline.generate_chat_response(request.session_id, request.question)
        return ChatResponse(answer=result["answer"], trace=result.get("trace", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Streaming chat — used by Streamlit for live trace panel
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    def generate():
        for event in rag_pipeline.stream_chat_response(request.session_id, request.question):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/ingest/documentation", response_model=IngestionResponse)
async def api_ingest_documentation(request: IngestionRequest):
    try:
        result = rag_pipeline.ingest_documentation(request.file_path, request.original_filename)
        return IngestionResponse(status="success", message=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/faq", response_model=IngestionResponse)
async def api_ingest_faq(request: IngestionRequest):
    try:
        result = rag_pipeline.ingest_faqs(request.file_path, request.original_filename)
        return IngestionResponse(status="success", message=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/chats", response_model=IngestionResponse)
async def api_ingest_chats(request: IngestionRequest):
    try:
        result = rag_pipeline.ingest_chats(request.file_path, request.original_filename)
        return IngestionResponse(
            status="success",
            message=result["message"],
            extracted_data=result["extracted_data"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
