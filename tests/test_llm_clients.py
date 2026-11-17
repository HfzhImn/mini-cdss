"""
test_llm_clients.py
---------------------
TUJUAN:
Menguji bagian llm_clients.py yang bisa dites TANPA API key -- yaitu
matematika estimasi cost (murni perhitungan angka dari published pricing)
dan helper `_extract_text` (parsing response, tidak butuh network).

Kita SENGAJA tidak menguji `_call()` di sini karena itu butuh API key
sungguhan dan koneksi ke provider asli -- itu ranahnya integration test /
manual run (lihat eval_runner.py), bukan unit test yang harus cepat &
jalan di CI tanpa secret apapun.
"""

from types import SimpleNamespace

from llm_clients import ClaudeClient, GeminiClient, GroqClient, _extract_text


class TestCostEstimation:
    def test_groq_cost_scales_linearly_with_tokens(self):
        client = GroqClient.__new__(GroqClient)  # skip __init__ (butuh API key)
        client.model = "llama-3.1-8b-instant"
        cost_1k = client.estimate_cost(input_tokens=1000, output_tokens=0)
        cost_2k = client.estimate_cost(input_tokens=2000, output_tokens=0)
        # Cost harus linear -- 2x token = 2x cost. Kalau tidak, ada bug
        # matematika (mis. salah pakai token count kumulatif).
        assert cost_2k == cost_1k * 2

    def test_groq_zero_tokens_costs_zero(self):
        client = GroqClient.__new__(GroqClient)
        client.model = "llama-3.1-8b-instant"
        assert client.estimate_cost(0, 0) == 0.0

    def test_unknown_model_defaults_to_zero_cost_not_crash(self):
        # Kalau model belum terdaftar di PRICING dict, harus fallback ke 0
        # dengan tenang, BUKAN KeyError yang bikin seluruh eval run crash
        # di tengah jalan gara-gara 1 model yang belum di-price.
        client = GroqClient.__new__(GroqClient)
        client.model = "model-yang-belum-terdaftar"
        cost = client.estimate_cost(1000, 1000)
        assert cost == 0.0

    def test_output_tokens_priced_differently_from_input(self):
        client = GroqClient.__new__(GroqClient)
        client.model = "llama-3.1-8b-instant"
        cost_input_only = client.estimate_cost(input_tokens=1000, output_tokens=0)
        cost_output_only = client.estimate_cost(input_tokens=0, output_tokens=1000)
        # Groq (dan semua provider LLM) selalu pricing output lebih mahal
        # dari input -- kalau angkanya sama, kemungkinan besar tertukar
        # field-nya di PRICING dict.
        assert cost_output_only > cost_input_only

    def test_gemini_cost_estimation_runs(self):
        client = GeminiClient.__new__(GeminiClient)
        client.model = "gemini-2.0-flash"
        cost = client.estimate_cost(1000, 500)
        assert cost > 0

    def test_claude_cache_read_cheaper_than_normal_input(self):
        # Ini yang membuktikan cost benefit dari prompt caching (Hari 2)
        # secara matematis, bukan cuma klaim di README
        p = ClaudeClient.PRICING["claude-haiku-4-5"]
        assert p["cache_read"] < p["input"]


class TestExtractText:
    def test_extracts_single_text_block(self):
        blocks = [SimpleNamespace(type="text", text="Halo dunia")]
        assert _extract_text(blocks) == "Halo dunia"

    def test_skips_non_text_blocks(self):
        # Ini test paling penting -- membuktikan fix untuk isu "thinking
        # block sebelum text block" (ditemukan lewat mypy) benar-benar
        # berfungsi, bukan cuma komentar kosong.
        blocks = [
            SimpleNamespace(type="thinking", text="proses berpikir internal..."),
            SimpleNamespace(type="text", text="Ini jawaban final"),
        ]
        assert _extract_text(blocks) == "Ini jawaban final"

    def test_joins_multiple_text_blocks(self):
        blocks = [
            SimpleNamespace(type="text", text="Bagian pertama. "),
            SimpleNamespace(type="text", text="Bagian kedua."),
        ]
        assert _extract_text(blocks) == "Bagian pertama. Bagian kedua."

    def test_empty_content_returns_empty_string(self):
        assert _extract_text([]) == ""
