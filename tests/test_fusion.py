"""
test_fusion.py
----------------
TUJUAN:
Menguji logic hybrid search (BM25 + Reciprocal Rank Fusion) secara
terisolasi -- semua fungsi di fusion.py sengaja pure (tidak menyentuh
ChromaDB/embedding model), jadi test ini cepat dan tidak butuh API key.

Fokus test di sini: apakah RRF benar-benar melakukan apa yang diklaim
di docstring-nya -- dokumen yang muncul di beberapa ranking sekaligus
harus naik ke atas, dan fusion harus tetap masuk akal walau salah satu
metode kosong (mis. BM25 tidak menemukan kecocokan sama sekali).
"""

from fusion import bm25_rank, build_bm25_index, reciprocal_rank_fusion, tokenize


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("Kolera Tidak Spesifik") == ["kolera", "tidak", "spesifik"]

    def test_strips_punctuation(self):
        assert tokenize("ICD-10: A00.9!") == ["icd", "10", "a00", "9"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []


class TestBm25Rank:
    def setup_method(self):
        self.corpus = [
            "Cholera unspecified infection",
            "Type 2 diabetes mellitus without complications",
            "Acute upper respiratory infection unspecified",
            "Essential primary hypertension",
        ]
        self.index = build_bm25_index(self.corpus)

    def test_exact_keyword_match_ranks_first(self):
        ranked = bm25_rank(self.index, "diabetes mellitus", corpus_size=len(self.corpus), top_k=4)
        assert ranked[0] == 1  # index dokumen "Type 2 diabetes mellitus..."

    def test_respects_top_k_limit(self):
        ranked = bm25_rank(self.index, "infection", corpus_size=len(self.corpus), top_k=2)
        assert len(ranked) == 2

    def test_no_keyword_match_still_returns_full_ranking(self):
        # BM25 tetap mengembalikan semua dokumen (dengan skor 0 untuk yang
        # tidak match sama sekali) -- bukan list kosong. Ini penting supaya
        # RRF di tahap selanjutnya tetap punya kandidat untuk digabung.
        ranked = bm25_rank(self.index, "xyz_kata_yang_tidak_ada", corpus_size=len(self.corpus), top_k=4)
        assert len(ranked) == 4

    def test_allowed_positions_limit_ranking(self):
        ranked = bm25_rank(
            self.index,
            "infection",
            corpus_size=len(self.corpus),
            top_k=4,
            allowed_positions={1, 3},
        )
        assert set(ranked) <= {1, 3}


class TestReciprocalRankFusion:
    def test_document_ranked_high_by_both_methods_wins(self):
        semantic_ranking = [2, 0, 1, 3]
        bm25_ranking = [2, 1, 0, 3]
        fused = reciprocal_rank_fusion([semantic_ranking, bm25_ranking])
        # Dokumen index 2 rank #1 di KEDUA metode -- harus jadi #1 di hasil fusion
        assert fused[0] == 2

    def test_document_only_in_one_method_still_included(self):
        semantic_ranking = [0, 1]
        bm25_ranking = [2, 3]
        fused = reciprocal_rank_fusion([semantic_ranking, bm25_ranking])
        assert set(fused) == {0, 1, 2, 3}

    def test_single_method_fusion_preserves_original_order(self):
        # Fusion dengan cuma 1 metode (list of 1 ranking) harus menghasilkan
        # urutan yang SAMA PERSIS dengan input -- sanity check paling dasar.
        ranking = [3, 1, 4, 0, 2]
        fused = reciprocal_rank_fusion([ranking])
        assert fused == ranking

    def test_empty_rankings_return_empty_result(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_agreement_beats_single_strong_ranking(self):
        # Kasus penting yang membuktikan NILAI hybrid search: dokumen yang
        # rank #2 di KEDUA metode harus mengalahkan dokumen yang rank #1
        # di HANYA SATU metode dan tidak muncul sama sekali di metode lain
        # -- konsensus dari 2 metode > sinyal kuat dari 1 metode saja.
        semantic_ranking = [5, 1, 2]  # doc 5 rank #1, doc 1 rank #2
        bm25_ranking = [1, 9]  # doc 1 rank #1 (tapi doc 5 tidak muncul sama sekali)
        fused = reciprocal_rank_fusion([semantic_ranking, bm25_ranking])
        assert fused[0] == 1  # doc 1: muncul di kedua metode, menang dari doc 5
