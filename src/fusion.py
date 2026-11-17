"""
fusion.py
---------
TUJUAN:
Implementasi HYBRID SEARCH: menggabungkan hasil BM25 (lexical/keyword
matching) dengan semantic search (embedding similarity) SEBELUM reranking.

KENAPA HYBRID, BUKAN SEMANTIC SAJA:
Semantic search bagus menangkap KEMIRIPAN MAKNA ("radang usus" bisa match
"gastroenteritis" walau beda kata sama sekali), tapi kadang KALAH presisi
untuk pencarian ISTILAH SPESIFIK/EKSAK -- misal query yang menyebut nama
obat atau istilah medis persis kata-per-kata. BM25 (cocokkan kata kunci
literal) justru lebih tajam di kasus itu karena tidak "melebur" istilah
penting ke ruang vector yang lebih kabur. Kombinasi keduanya = recall
semantik yang luas + presisi leksikal untuk istilah eksak.

KENAPA RECIPROCAL RANK FUSION (RRF), BUKAN GABUNGKAN SKOR MENTAH:
Skor BM25 (bisa 0 sampai puluhan) dan skor cosine similarity ChromaDB
(0 sampai 1) ada di SKALA BERBEDA TOTAL -- menjumlahkan mentah-mentah
tidak masuk akal, salah satu metode akan selalu mendominasi cuma karena
rentang angkanya lebih besar, bukan karena benar-benar lebih relevan.
RRF menghindari masalah itu dengan HANYA memakai RANKING (posisi ke-1,
ke-2, dst) dari tiap metode, bukan skor mentahnya -- otomatis adil
menggabungkan metode manapun tanpa perlu kalibrasi skala.

Formula RRF standar: score(d) = sum_per_metode( 1 / (k + rank(d)) ),
k=60 adalah konstanta yang dipakai luas di literatur information
retrieval (termasuk paper asli RRF oleh Cormack et al., 2009) -- bukan
angka sembarang yang saya pilih sendiri.

Semua fungsi di sini SENGAJA murni (pure functions, tidak menyentuh
ChromaDB/API apapun) supaya bisa di-unit-test cepat & terisolasi --
lihat tests/test_fusion.py.
"""

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Tokenisasi sederhana: lowercase + ambil rangkaian alfanumerik.

    Sengaja MINIMAL (bukan NLP tokenizer canggih dengan stemming/stopword
    removal) -- untuk domain istilah medis & nama obat, tokenizer yang
    terlalu agresif bisa merusak istilah penting (mis. stemming 'diabetes'
    jadi 'diabet' tidak menambah nilai apapun di konteks ini, malah
    berisiko salah gabung dengan istilah lain yang kebetulan mirip).
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(documents: list[str]) -> BM25Okapi:
    tokenized_corpus = [tokenize(doc) for doc in documents]
    return BM25Okapi(tokenized_corpus)


def bm25_rank(
    bm25_index: BM25Okapi,
    query: str,
    corpus_size: int,
    top_k: int,
    allowed_positions: set[int] | None = None,
) -> list[int]:
    """
    Return indeks dokumen (posisi di corpus, BUKAN dokumennya langsung)
    terurut dari skor BM25 tertinggi. Index dipakai supaya caller bisa
    fusion-kan dengan hasil metode lain yang merujuk corpus yang sama.
    """
    scores = bm25_index.get_scores(tokenize(query))
    positions = allowed_positions if allowed_positions is not None else range(corpus_size)
    ranked = sorted(positions, key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """
    Gabungkan beberapa ranking (masing-masing list index dokumen, terurut
    dari paling relevan) jadi 1 ranking gabungan pakai RRF.

    Dokumen yang muncul di BEBERAPA ranking mendapat skor RRF lebih tinggi
    (kontribusi dari tiap metode dijumlahkan) -- ini yang membuat dokumen
    yang "disepakati" relevan oleh semantic DAN lexical search naik ke atas,
    persis perilaku yang kita inginkan dari hybrid search.

    Return: list index dokumen terurut dari skor RRF gabungan tertinggi
    ke terendah. Index yang tidak muncul di ranked_lists manapun tidak
    akan muncul di hasil (RRF cuma menggabungkan yang sudah di-retrieve).
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_idx in enumerate(ranked_list):
            # rank+1 supaya posisi teratas dapat rank=1, bukan rank=0
            # (rank=0 akan bikin pembagi jadi k+0=k untuk SEMUA metode di
            # posisi puncak, kurang standar dibanding literatur RRF asli)
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
