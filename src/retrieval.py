"""
retrieval.py
------------
TUJUAN:
Ini implementasi pola RAG retrieval 3 tahap:

  1. HYBRID RETRIEVE: cari kandidat dari 2 metode independen --
     semantic search (embedding + cosine similarity di ChromaDB, bagus
     menangkap kemiripan makna) DAN BM25 (lexical/keyword matching, bagus
     menangkap istilah eksak). Hasil keduanya digabung pakai Reciprocal
     Rank Fusion (lihat fusion.py untuk penjelasan lengkap kenapa RRF,
     bukan gabungkan skor mentah).

  2. RERANK: dari hasil fusion itu, pakai cross-encoder untuk menilai
     ULANG relevansi tiap dokumen terhadap query secara lebih teliti
     (model ini melihat query & dokumen BERSAMAAN, bukan membandingkan
     2 vector terpisah -- lebih akurat tapi lebih lambat, makanya cuma
     dipakai untuk kandidat hasil retrieval, bukan seluruh dataset).
     Ambil top-N hasil rerank untuk masuk ke prompt LLM.

Kenapa ini penting buat interview: JD eksplisit sebut "reranking" sebagai
skill yang dicari, dan hybrid search (bukan semantic search polos) adalah
pola yang dipakai RAG production sungguhan -- banyak portfolio project
berhenti di semantic-search-plus-rerank saja; menambahkan BM25 dan bisa
menjelaskan KAPAN lexical search unggul dari semantic (istilah/kode eksak)
adalah sinyal paham trade-off, bukan cuma ngikutin tutorial paling umum.
"""

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from src.fusion import bm25_rank, build_bm25_index, reciprocal_rank_fusion
from src.logging_config import get_logger
from src.settings import settings

logger = get_logger(__name__)


@dataclass
class RetrievedDoc:
    text: str
    metadata: dict
    rerank_score: float


class _CollectionIndex:
    """
    Menyimpan corpus penuh 1 collection ChromaDB dalam bentuk yang bisa
    dipakai BM25 (butuh semua dokumen di memori, beda dari ChromaDB yang
    query on-demand). `ids` menjaga urutan tetap supaya index posisi di
    corpus ini bisa dipetakan balik ke ID asli ChromaDB dengan konsisten
    -- ini yang menyatukan hasil semantic search (dari ChromaDB, merujuk
    ID) dengan hasil BM25 (dari corpus lokal ini, merujuk posisi list).
    """

    def __init__(self, collection):
        # `include` sengaja tidak minta embeddings -- kita cuma butuh teks
        # & metadata di sini, embeddings tetap tersimpan di ChromaDB sendiri
        # dan dipakai lewat collection.query(), bukan dimuat ulang manual.
        raw = collection.get(include=["documents", "metadatas"])
        # ChromaDB tidak menjamin urutan hasil .get() sama dengan urutan
        # insert -- urutkan eksplisit by ID (integer) supaya index posisi
        # di sini deterministic & reproducible antar-run.
        order = sorted(range(len(raw["ids"])), key=lambda i: int(raw["ids"][i]))
        self.ids = [raw["ids"][i] for i in order]
        self.documents = [raw["documents"][i] for i in order]
        self.metadatas = [raw["metadatas"][i] for i in order]
        self.id_to_position = {id_: pos for pos, id_ in enumerate(self.ids)}
        self.bm25 = build_bm25_index(self.documents)
        logger.info(f"BM25 index dibangun dari {len(self.documents)} dokumen")


class Retriever:
    def __init__(self, chroma_client, embedder):
        """
        chroma_client: instance chromadb.PersistentClient yang SUDAH dibuat
                        di build_index.py (kita reuse, bukan bikin baru)
        embedder: instance SentenceTransformer yang sama dipakai saat indexing
                        (WAJIB sama modelnya -- query dan dokumen harus di-embed
                        dengan model identik supaya vector-nya sebanding)
        """
        self.client = chroma_client
        self.embedder = embedder
        self.icd10_col = self.client.get_collection(settings.icd10_collection)
        self.drug_col = self.client.get_collection(settings.drug_interaction_collection)

        logger.info("Memuat corpus & membangun BM25 index untuk hybrid search...")
        self._icd10_index = _CollectionIndex(self.icd10_col)
        self._drug_index = _CollectionIndex(self.drug_col)

        logger.info(f"Loading reranker: {settings.reranker_model}")
        self.reranker = CrossEncoder(settings.reranker_model)

    def _semantic_rank(
        self,
        collection,
        corpus_index: _CollectionIndex,
        query: str,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[int]:
        """Return posisi (bukan ID mentah) hasil semantic search, terurut
        relevansi -- posisi ini merujuk ke corpus_index.documents/.ids,
        supaya bisa langsung di-fusion-kan dengan hasil BM25."""
        query_embedding = self.embedder.encode([query]).tolist()
        query_args = {"query_embeddings": query_embedding, "n_results": top_k}
        if filters:
            if len(filters) == 1:
                query_args["where"] = filters
            else:
                query_args["where"] = {"$and": [{key: value} for key, value in filters.items()]}
        results = collection.query(**query_args)
        result_ids = results["ids"][0]
        # Beberapa ID hasil query bisa saja tidak ada di corpus_index kalau
        # terjadi race condition (dokumen dihapus di antara index dibangun
        # & query dijalankan) -- filter defensif, jangan crash karena itu.
        return [corpus_index.id_to_position[id_] for id_ in result_ids if id_ in corpus_index.id_to_position]

    def _hybrid_search(
        self,
        collection,
        corpus_index: _CollectionIndex,
        query: str,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[tuple]:
        allowed_positions = None
        if filters:
            allowed_positions = {
                position
                for position, metadata in enumerate(corpus_index.metadatas)
                if all(metadata.get(key) == value for key, value in filters.items())
            }
            if not allowed_positions:
                return []

        semantic_ranking = self._semantic_rank(collection, corpus_index, query, top_k, filters)
        bm25_ranking = bm25_rank(
            corpus_index.bm25,
            query,
            len(corpus_index.documents),
            top_k,
            allowed_positions,
        )
        fused_positions = reciprocal_rank_fusion([semantic_ranking, bm25_ranking])[:top_k]
        return [(corpus_index.documents[pos], corpus_index.metadatas[pos]) for pos in fused_positions]

    def _rerank(self, query: str, candidates: list, top_n: int) -> list[RetrievedDoc]:
        if not candidates:
            return []
        pairs = [(query, doc_text) for doc_text, _ in candidates]
        scores = self.reranker.predict(pairs)
        scored = [
            RetrievedDoc(text=doc_text, metadata=meta, rerank_score=float(score))
            for (doc_text, meta), score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda d: d.rerank_score, reverse=True)
        return scored[:top_n]

    def search_icd10(
        self,
        query: str,
        chapter_id: str | None = None,
        group_id: str | None = None,
    ) -> list[RetrievedDoc]:
        filters = {
            key: value
            for key, value in {"chapter_id": chapter_id, "group_id": group_id}.items()
            if value is not None
        }
        candidates = self._hybrid_search(
            self.icd10_col,
            self._icd10_index,
            query,
            settings.retrieve_top_k,
            filters or None,
        )
        return self._rerank(query, candidates, settings.rerank_top_n)

    def search_drug_interactions(self, query: str) -> list[RetrievedDoc]:
        candidates = self._hybrid_search(self.drug_col, self._drug_index, query, settings.retrieve_top_k)
        return self._rerank(query, candidates, settings.rerank_top_n)
