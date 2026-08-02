

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rag_pipeline  # Importing our engine
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Engineering RAG API", version="1.0")


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str

class IngestionRequest(BaseModel):
    file_path: str
    original_filename: Optional[str] = None
class IngestionResponse(BaseModel):
    status: str
    message: str
    extracted_data: Optional[str] = None

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        answer = rag_pipeline.generate_chat_response(request.session_id, request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/documentation", response_model=IngestionResponse)
async def api_ingest_documentation(request: IngestionRequest):
    try:
        result = rag_pipeline.ingest_documentation(request.file_path,request.original_filename)
        return IngestionResponse(status="success", message=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/faq", response_model=IngestionResponse)
async def api_ingest_faq(request: IngestionRequest):
    try:
        result = rag_pipeline.ingest_faqs(request.file_path,request.original_filename)
        return IngestionResponse(status="success", message=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/chats", response_model=IngestionResponse)
async def api_ingest_chats(request: IngestionRequest):
    try:
        # result ab ek dictionary hai: {"message": "...", "extracted_data": "..."}
        result = rag_pipeline.ingest_chats(request.file_path,request.original_filename)
        
        # Hum dictionary se values nikal kar response model mein match kar rahe hain
        return IngestionResponse(
            status="success",
            message=result["message"],
            extracted_data=result["extracted_data"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))