"""
build_index.py
---------------
TUJUAN:
Ini "otak" dari fase indexing RAG. Kita ambil data terstruktur (CSV) dan
ubah jadi vector embeddings yang disimpan di ChromaDB, supaya nanti bisa
dicari secara SEMANTIK (bukan cuma keyword matching).

Kenapa 2 collection terpisah (bukan 1 gabungan)?
Karena ICD-10 dan drug interaction itu 2 DOMAIN PERTANYAAN yang berbeda
("apa kode diagnosisnya?" vs "apakah 2 obat ini aman dikombinasi?").
Memisahkan collection membuat retrieval lebih presisi -- kita bisa
query collection yang tepat sesuai intent, bukan mengaduk semuanya jadi
satu ruang vector yang bising. Di CDSS asli, ini setara dengan 2 modul
berbeda yang disebut di job description.

CARA JALANKAN:
    python3 src/build_index.py

Ini HANYA perlu dijalankan ulang kalau data sumbernya berubah -- hasilnya
disimpan persisten di folder data/chroma_db/, jadi tidak perlu re-embed
setiap kali aplikasi start.
"""

import time

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.settings import settings


def add_in_batches(collection, ids, embeddings, documents, metadatas, batch_size=5000):
    """Add documents in chunks below ChromaDB's maximum batch size."""
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def build_icd10_index(client, embedder):
    print(f"\n[ICD-10] Memuat data dari {settings.icd10_csv}")
    df = pd.read_csv(settings.icd10_csv, dtype=str)
    print(f"[ICD-10] {len(df)} kode dimuat")

    # Hapus collection lama kalau ada, supaya build ini idempotent
    # (bisa dijalankan berkali-kali tanpa duplikasi data)
    try:
        client.delete_collection(settings.icd10_collection)
    except Exception:
        pass
    collection = client.create_collection(
        name=settings.icd10_collection,
        metadata={"hnsw:space": "cosine"},  # cosine similarity, standar untuk sentence embeddings
    )

    # Dokumen yang di-embed = kode + deskripsi, supaya query bisa match
    # baik dari kode ("A00.9") maupun dari gejala/nama penyakit
    documents = [f"{row.icd10_code}: {row.description}" for row in df.itertuples()]

    print("[ICD-10] Membuat embeddings (ini yang paling makan waktu)...")
    t0 = time.time()
    embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=64)
    print(f"[ICD-10] Selesai dalam {time.time() - t0:.1f}s")

    add_in_batches(
        collection,
        ids=[str(i) for i in range(len(df))],
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=[
            {
                "icd10_code": row.icd10_code,
                "description": row.description,
                "chapter_id": row.chapter_id,
                "group_id": row.group_id,
            }
            for row in df.itertuples()
        ],
    )
    print(f"[ICD-10] {collection.count()} dokumen ter-index di ChromaDB")


def build_drug_interaction_index(client, embedder):
    print(f"\n[Drug Interaction] Memuat data dari {settings.drug_interactions_csv}")
    df = pd.read_csv(settings.drug_interactions_csv)
    print(f"[Drug Interaction] {len(df)} interaksi dimuat")

    try:
        client.delete_collection(settings.drug_interaction_collection)
    except Exception:
        pass
    collection = client.create_collection(
        name=settings.drug_interaction_collection,
        metadata={"hnsw:space": "cosine"},
    )

    documents = df["search_text"].tolist()

    print("[Drug Interaction] Membuat embeddings...")
    embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=64)

    add_in_batches(
        collection,
        ids=[str(i) for i in range(len(df))],
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=[
            {
                "drug_a": row.drug_a,
                "drug_b": row.drug_b,
                "severity": row.severity,
                "mechanism": row.mechanism,
                "description": row.description,
            }
            for row in df.itertuples()
        ],
    )
    print(f"[Drug Interaction] {collection.count()} dokumen ter-index di ChromaDB")


def main():
    print(f"Loading embedding model: {settings.embedding_model} (download pertama kali agak lama, lalu di-cache)")
    embedder = SentenceTransformer(settings.embedding_model)
    # SentenceTransformer otomatis pilih GPU (CUDA) kalau tersedia, fallback ke
    # CPU kalau tidak -- baris ini cuma melaporkan pilihan itu ke layar supaya
    # kamu tahu PASTI device mana yang beneran dipakai, bukan menebak-nebak.
    print(f"Device yang dipakai: {embedder.device}")

    settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))

    build_icd10_index(client, embedder)
    build_drug_interaction_index(client, embedder)

    print("\n✅ Index selesai dibangun. Collections tersimpan persisten di:", settings.chroma_db_dir)


if __name__ == "__main__":
    main()
