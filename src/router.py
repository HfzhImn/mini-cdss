"""
router.py
---------
TUJUAN:
Ini implementasi konkret dari "model routing" yang disebut JD di bagian
cost optimization ("Optimize prompts for cost via ... model routing").

IDE DASARNYA: tidak semua query butuh model paling pintar (dan paling mahal).
Query lookup sederhana ("kode ICD-10 untuk kolera") bisa dijawab model kecil
& cepat dengan akurasi sama baiknya dengan model besar -- jadi kenapa bayar
lebih mahal? Sebaliknya, query yang butuh sintesis/perbandingan kompleks
("bandingkan interaksi obat A dengan B dan C, mana yang paling berisiko")
lebih aman diarahkan ke model yang lebih kuat.

HEURISTIK yang dipakai di sini SENGAJA sederhana (panjang query + kata kunci
kompleksitas) -- ini portfolio project, bukan sistem routing berbasis ML
terpisah. Yang penting recruiter lihat kamu PAHAM KONSEPNYA dan bisa
menjelaskan trade-off-nya, bukan langsung over-engineer classifier terpisah
di hari ke-2 dari 3 hari yang kamu punya.

Kalau mau dikembangkan lebih lanjut (bukan untuk sekarang, tapi bagus
disebut saat interview): heuristik ini bisa diganti classifier kecil yang
dilatih dari data eval (mis. prediksi "butuh model besar atau tidak"
berdasar hasil eval Hari 2 -- ini contoh bagus untuk section "kalau
dilanjutkan" di README).
"""

import re
from dataclasses import dataclass

# Kata kunci yang menandakan query butuh reasoning lebih dalam --
# perbandingan, sintesis multi-item, atau pertanyaan "kenapa/bagaimana"
COMPLEXITY_KEYWORDS = [
    "bandingkan", "kombinasi", "kenapa", "mengapa", "bagaimana jika",
    "compare", "why", "explain", "what if", "risiko tertinggi", "paling aman",
]

SIMPLE_QUERY_MAX_WORDS = 12


@dataclass
class RoutingDecision:
    provider: str
    model: str
    reason: str


def route_query(query: str) -> RoutingDecision:
    word_count = len(query.split())
    has_complexity_keyword = any(kw in query.lower() for kw in COMPLEXITY_KEYWORDS)
    has_multiple_entities = len(re.findall(r"\bdan\b|\band\b|,", query.lower())) >= 2

    if has_complexity_keyword or has_multiple_entities:
        return RoutingDecision(
            provider="groq",
            model="llama-3.3-70b-versatile",
            reason="Query mengandung indikasi perbandingan/sintesis multi-entitas -> model lebih besar",
        )

    if word_count <= SIMPLE_QUERY_MAX_WORDS:
        return RoutingDecision(
            provider="groq",
            model="llama-3.1-8b-instant",
            reason=f"Query pendek & sederhana ({word_count} kata) -> model kecil & cepat cukup",
        )

    return RoutingDecision(
        provider="gemini",
        model="gemini-2.0-flash",
        reason="Query medium-length tanpa indikasi kompleksitas tinggi -> model menengah",
    )


if __name__ == "__main__":
    # Test cepat tanpa perlu API key -- murni tes logic routing
    test_queries = [
        "kode ICD-10 untuk kolera",
        "apakah warfarin aman dengan ibuprofen?",
        "bandingkan risiko kombinasi warfarin dengan aspirin dan ibuprofen, mana yang paling berbahaya?",
        "kenapa ACE inhibitor tidak boleh dikombinasi dengan spironolactone?",
    ]
    for q in test_queries:
        decision = route_query(q)
        print(f"Query: {q!r}\n  -> {decision.provider}/{decision.model}\n  alasan: {decision.reason}\n")
