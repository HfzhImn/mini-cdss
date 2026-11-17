import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.rag import RAGResult, answer_query

router = APIRouter(prefix="/api/v1")

class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=3, max_length=500)
    mode: Literal["icd10", "drug", "auto"] = "auto"
    chapter_id: str | None = Field(default=None, min_length=1, max_length=10)
    group_id: str | None = Field(default=None, min_length=1, max_length=10)


class QueryMetadata(BaseModel):
    model: str
    provider: str
    latency_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    retrieval_count: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, object]]
    metadata: QueryMetadata


@router.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest, request: Request):
    if req.mode == "drug" and (req.chapter_id or req.group_id):
        raise HTTPException(
            status_code=422,
            detail="chapter_id dan group_id hanya berlaku untuk mode icd10",
        )

    # Ambil dependencies yang di-inject dari app state
    retriever = request.app.state.retriever
    llm = request.app.state.llm
    
    # Jalankan fungsi sinkron di thread pool agar tidak memblokir event loop FastAPI
    try:
        result: RAGResult = await asyncio.to_thread(
            answer_query,
            retriever,
            llm,
            req.query.strip(),
            req.mode,
            req.chapter_id,
            req.group_id,
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("RAG query failed")
        raise HTTPException(status_code=502, detail="Layanan RAG gagal memproses permintaan") from exc
    
    # Konversi hasil RAGResult menjadi dictionary untuk JSON response
    return result.to_api_dict()
