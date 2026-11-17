"""
cli.py
------
TUJUAN: cara TERCEPAT untuk kamu ngetes apakah pipeline RAG jalan dengan
benar, TANPA perlu bangun UI dulu. Jalankan ini duluan sebelum lanjut ke
FastAPI/Next.js -- kalau CLI ini sudah menjawab dengan benar, berarti
"otak"-nya (retrieval + LLM) sudah beres, tinggal bungkus jadi API.

CARA PAKAI:
    export GROQ_API_KEY=xxx   (atau taruh di file .env)
    python3 src/cli.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

from src import rag
from src.llm_clients import get_client
from src.retrieval import Retriever
from src.settings import settings


def main():
    print("Memuat embedding model & index... (sekali di awal, agak lama)")
    embedder = SentenceTransformer(settings.embedding_model)
    client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
    retriever = Retriever(client, embedder)

    provider = settings.default_provider
    llm = get_client(provider)
    print(f"Pakai provider: {provider}\n")

    print("=== Mini-CDSS CLI ===")
    print("Ketik pertanyaan (contoh: 'kode ICD-10 untuk diare akut pada anak')")
    print("Ketik 'obat: <pertanyaan>' untuk mode drug interaction")
    print("Ketik 'exit' untuk keluar\n")

    while True:
        query = input("> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        if query.lower().startswith("obat:"):
            result = rag.answer_drug_interaction_query(retriever, llm, query[5:].strip())
        else:
            result = rag.answer_icd10_query(retriever, llm, query)

        print(f"\n{result.answer}\n")
        print(
            f"[model: {result.llm.model} | latency: {result.llm.latency_ms:.0f}ms | "
            f"cost: ${result.llm.estimated_cost_usd:.6f} | dokumen dipakai: {result.retrieval_count}]\n"
        )


if __name__ == "__main__":
    main()
