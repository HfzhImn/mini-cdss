"""
rag.py
------
TUJUAN:
Ini yang menyatukan semua bagian: retrieval.py (cari data relevan) +
llm_clients.py (generate jawaban) -> jadi satu alur RAG yang GROUNDED.

KONSEP PALING PENTING DI FILE INI: "grounding" untuk mencegah hallucination.
Sistem prompt di bawah secara eksplisit MELARANG model menjawab dari
"pengetahuan umum"-nya sendiri -- model WAJIB hanya memakai potongan
konteks yang kita berikan dari hasil retrieval. Kalau jawabannya tidak ada
di konteks, model harus bilang "tidak ditemukan", bukan mengarang.

Ini penting BANGET di domain kesehatan (dan ini yang nanti diukur di
eval framework Hari 2 sebagai "hallucination rate") -- LLM yang mengarang
kode ICD-10 atau info interaksi obat yang salah bisa berbahaya kalau
dipakai di dunia nyata.
"""

from dataclasses import dataclass

from src.llm_clients import LLMClient, LLMResponse
from src.retrieval import RetrievedDoc, Retriever

SYSTEM_PROMPT = """Kamu adalah asisten klinis yang membantu tenaga medis mencari kode ICD-10 \
dan informasi interaksi obat.

ATURAN WAJIB (pelanggaran terhadap aturan ini dianggap jawaban salah):
1. HANYA gunakan informasi dari bagian "KONTEKS" di bawah. JANGAN PERNAH \
menjawab dari pengetahuan umummu sendiri, walau kamu yakin tahu jawabannya.
2. Kalau jawaban yang diminta TIDAK ADA di dalam KONTEKS, kamu WAJIB menjawab \
persis: "Tidak ditemukan informasi yang relevan dalam basis data." Jangan menebak \
atau mengarang kode/nama obat yang tidak ada di konteks.
3. Setiap kali kamu menyebut kode ICD-10 atau nama obat, itu HARUS persis sama \
dengan yang tertulis di KONTEKS -- jangan mengubah, menyingkat, atau menggabungkan.
4. Jawab dalam Bahasa Indonesia yang jelas, singkat, dan sertakan sumber \
(kode/nama yang kamu kutip dari konteks) di akhir jawaban.
5. Ini BUKAN pengganti penilaian klinis dokter -- kamu hanya alat bantu pencarian."""


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    llm: LLMResponse
    retrieval_count: int

    def to_api_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "metadata": {
                "model": self.llm.model,
                "provider": getattr(self.llm, "provider", "unknown"),
                "latency_ms": getattr(self.llm, "latency_ms", 0.0),
                "cost_usd": getattr(self.llm, "estimated_cost_usd", 0.0),
                "input_tokens": getattr(self.llm, "input_tokens", 0),
                "output_tokens": getattr(self.llm, "output_tokens", 0),
                "retrieval_count": self.retrieval_count,
            }
        }


def _format_context(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(tidak ada dokumen relevan ditemukan)"
    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{i}] {doc.text} (skor relevansi: {doc.rerank_score:.2f})")
    return "\n".join(lines)


def answer_icd10_query(
    retriever: Retriever,
    llm: LLMClient,
    query: str,
    chapter_id: str | None = None,
    group_id: str | None = None,
) -> RAGResult:
    docs = retriever.search_icd10(query, chapter_id=chapter_id, group_id=group_id)
    context = _format_context(docs)
    user_prompt = f"KONTEKS:\n{context}\n\nPERTANYAAN: {query}"
    llm_response = llm.generate(SYSTEM_PROMPT, user_prompt)
    return RAGResult(
        answer=llm_response.text,
        sources=[d.metadata for d in docs],
        llm=llm_response,
        retrieval_count=len(docs),
    )


def answer_drug_interaction_query(retriever: Retriever, llm: LLMClient, query: str) -> RAGResult:
    docs = retriever.search_drug_interactions(query)
    context = _format_context(docs)
    user_prompt = f"KONTEKS:\n{context}\n\nPERTANYAAN: {query}"
    llm_response = llm.generate(SYSTEM_PROMPT, user_prompt)
    return RAGResult(
        answer=llm_response.text,
        sources=[d.metadata for d in docs],
        llm=llm_response,
        retrieval_count=len(docs),
    )


def answer_query(
    retriever: Retriever,
    llm: LLMClient,
    query: str,
    mode: str = "auto",
    chapter_id: str | None = None,
    group_id: str | None = None,
) -> RAGResult:
    """
    Unified entry point untuk FastAPI.
    mode: "icd10" | "drug" | "auto" (auto-detect dari prefix 'obat:')
    """
    if mode == "drug" or (mode == "auto" and query.lower().startswith("obat:")):
        clean_query = query[5:].strip() if (mode == "auto" and query.lower().startswith("obat:")) else query
        return answer_drug_interaction_query(retriever, llm, clean_query)
    return answer_icd10_query(retriever, llm, query, chapter_id, group_id)
