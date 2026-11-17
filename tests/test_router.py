"""
test_router.py
---------------
TUJUAN:
Formalisasi test manual yang kemarin cuma dijalankan sekali di terminal
(lihat riwayat "python3 router.py" yang printnya kita baca manual). Bedanya
test formal vs itu:
  - Ada ASSERTION eksplisit, bukan cuma print yang kita baca sekilas lalu
    percaya "kelihatannya benar"
  - Jalan otomatis tiap kali ada perubahan kode (lewat `pytest`), bukan
    perlu diingat-ingat untuk dicoba manual lagi
  - Kalau nanti heuristik routing diubah, test ini LANGSUNG kasih tau
    kalau ada regresi (query yang tadinya ke-routing benar jadi salah)
"""

import pytest

from router import COMPLEXITY_KEYWORDS, SIMPLE_QUERY_MAX_WORDS, route_query


class TestRouteQuery:
    def test_short_simple_query_routes_to_small_model(self):
        decision = route_query("kode ICD-10 untuk kolera")
        assert decision.provider == "groq"
        assert decision.model == "llama-3.1-8b-instant"

    def test_query_with_comparison_keyword_routes_to_large_model(self):
        decision = route_query(
            "bandingkan risiko kombinasi warfarin dengan aspirin dan ibuprofen"
        )
        assert decision.model == "llama-3.3-70b-versatile"

    def test_query_with_why_keyword_routes_to_large_model(self):
        decision = route_query("kenapa ACE inhibitor tidak boleh dengan spironolactone?")
        assert decision.model == "llama-3.3-70b-versatile"

    def test_medium_length_query_without_complexity_routes_to_medium_model(self):
        # Query lebih dari SIMPLE_QUERY_MAX_WORDS kata, tapi tidak ada kata
        # kunci kompleksitas -- harus jatuh ke tier menengah (Gemini), bukan
        # tier kecil ATAU tier besar.
        long_simple_query = " ".join(["kata"] * (SIMPLE_QUERY_MAX_WORDS + 5))
        decision = route_query(long_simple_query)
        assert decision.provider == "gemini"

    @pytest.mark.parametrize("keyword", COMPLEXITY_KEYWORDS)
    def test_every_declared_complexity_keyword_actually_triggers_routing(self, keyword):
        # Ini test penting: memastikan SETIAP kata kunci yang didaftarkan di
        # COMPLEXITY_KEYWORDS benar-benar mengubah keputusan routing. Kalau
        # nanti ada yang nambah kata kunci baru ke daftar itu tapi salah
        # ketik/logic-nya tidak nyambung, test ini akan gagal dan ketahuan.
        query = f"tolong {keyword} beberapa hal untuk saya"
        decision = route_query(query)
        assert decision.model == "llama-3.3-70b-versatile", (
            f"Keyword {keyword!r} terdaftar di COMPLEXITY_KEYWORDS tapi "
            f"tidak memicu routing ke model besar"
        )

    def test_empty_query_does_not_crash(self):
        # Edge case: query kosong harus tetap menghasilkan keputusan valid,
        # bukan exception. Ini jenis test yang sering dilewatkan kalau cuma
        # testing manual dengan contoh query yang "masuk akal" saja.
        decision = route_query("")
        assert decision.provider in ("groq", "gemini")

    def test_routing_decision_always_includes_reason(self):
        # Kontrak penting: setiap keputusan routing HARUS punya alasan
        # yang bisa dibaca manusia -- ini yang bikin routing bisa di-debug
        # dan dijelaskan ke orang lain (mis. saat interview), bukan cuma
        # "black box" yang entah kenapa milih model tertentu.
        decision = route_query("test query apapun")
        assert decision.reason
        assert isinstance(decision.reason, str)
