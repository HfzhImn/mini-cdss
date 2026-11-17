from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from api.routes import query
from src.llm_clients import get_client
from src.retrieval import Retriever
from src.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resource berat SEKALI saat server start
    if not settings.chroma_db_dir.exists():
        raise RuntimeError(
            f"Chroma index belum tersedia di {settings.chroma_db_dir}. "
            "Jalankan 'python -m src.build_index' terlebih dahulu."
        )
    chroma = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
    collections = {
        collection if isinstance(collection, str) else collection.name
        for collection in chroma.list_collections()
    }
    required_collections = {
        settings.icd10_collection,
        settings.drug_interaction_collection,
    }
    missing_collections = required_collections - collections
    if missing_collections:
        missing = ", ".join(sorted(missing_collections))
        raise RuntimeError(
            f"Collection ChromaDB belum tersedia: {missing}. "
            "Jalankan 'python -m src.build_index' terlebih dahulu."
        )
    embedder = SentenceTransformer(settings.embedding_model)
    app.state.retriever = Retriever(chroma, embedder)
    app.state.llm = get_client(settings.default_provider)
    yield
    # Cleanup bisa ditambahkan di sini kalau diperlukan nanti

app = FastAPI(
    title="Mini-CDSS API",
    description="API untuk asisten klinis pencarian kode ICD-10 dan interaksi obat",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrasi Router
app.include_router(query.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Mini-CDSS API is running"}
