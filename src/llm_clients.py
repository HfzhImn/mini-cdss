"""
llm_clients.py
--------------
TUJUAN:
JD minta pengalaman dengan Claude, Gemini, Groq, DAN OpenAI. Kalau kode kita
manggil masing-masing provider dengan cara berbeda-beda tersebar di banyak
tempat, itu susah di-maintain dan susah dibandingkan saat eval. Solusinya:
1 INTERFACE seragam (`LLMClient.generate(...)`) yang di-implementasi
berbeda-beda per provider di belakang layar. Ini disebut "adapter pattern".

Manfaat konkret untuk proyekmu:
  - `rag.py` (orchestrator) tidak perlu tahu/peduli provider mana yang
    dipakai -- tinggal panggil `.generate()`
  - Model routing (Hari 2) jadi 1 baris kode: tinggal pilih instance client
    mana yang dipanggil berdasarkan kompleksitas query
  - Eval framework (Hari 2) bisa loop semua provider dengan kode yang SAMA

SEMUA API key diambil dari environment variable (lihat .env.example),
JANGAN PERNAH hardcode API key di kode -- ini bukan cuma soal keamanan,
tapi juga best practice yang akan dinilai kalau recruiter baca repo-mu.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.logging_config import get_logger
from src.retry import with_retry
from src.settings import settings

logger = get_logger(__name__)


def _extract_text(content_blocks) -> str:
    """
    Gabungkan semua blok bertipe 'text' dari response Claude.

    KENAPA TIDAK cukup `content[0].text`: model Claude generasi terbaru bisa
    mengirim blok 'thinking' SEBELUM blok 'text' di response yang sama --
    kalau kita asumsikan index 0 selalu text, kode ini bisa pecah (AttributeError,
    karena blok thinking tidak punya .text) begitu Anthropic mengubah default
    behavior modelnya. Filter eksplisit by `type` lebih tahan terhadap
    perubahan seperti itu di masa depan.
    """
    return "".join(block.text for block in content_blocks if block.type == "text")


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    # Estimasi biaya dalam USD, dihitung dari published pricing tiap provider.
    # Lihat PRICING di bawah -- WAJIB di-update manual kalau harga provider berubah,
    # karena tidak ada API yang mengembalikan cost secara real-time.
    estimated_cost_usd: float


class LLMClient(ABC):
    """Interface seragam yang wajib diimplementasi tiap provider."""

    provider_name: str = "unknown"
    model: str  # di-set oleh __init__ tiap subclass -- dideklarasikan di sini
    # supaya type checker (mypy) tahu atribut ini memang bagian dari kontrak
    # LLMClient, bukan cuma "muncul entah dari mana" di subclass masing-masing.

    @abstractmethod
    def _call(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens). Diimplementasi per-provider."""
        raise NotImplementedError

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        raise NotImplementedError

    @with_retry(max_attempts=3, base_delay=1.0)
    def _call_with_retry(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        return self._call(system_prompt, user_prompt)

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        t0 = time.perf_counter()
        text, in_tok, out_tok = self._call_with_retry(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000
        cost = self.estimate_cost(in_tok, out_tok)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )


# ---------------------------------------------------------------------------
# GROQ -- cepat, gratis untuk volume rendah-menengah, model open-source saja.
# Cocok jadi default untuk query sederhana/lookup cepat.
# ---------------------------------------------------------------------------
class GroqClient(LLMClient):
    provider_name = "groq"

    # Harga per 1M token (USD), diverifikasi Agustus 2026 dari pricing resmi Groq.
    # Model gratis tetap kita hitung "seolah berbayar" di sini supaya angka
    # perbandingan cost antar-model tetap adil & bermakna untuk portfolio,
    # bukan cuma "$0 karena kebetulan masih di free tier".
    PRICING = {
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    }

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        from groq import Groq

        self.model = model
        self.client = Groq(api_key=settings.require_api_key("groq"))

    def _call(self, system_prompt, user_prompt):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # rendah -- kita mau jawaban konsisten & grounded, bukan kreatif
            timeout=30.0,
        )
        text = resp.choices[0].message.content
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens

    def estimate_cost(self, input_tokens, output_tokens):
        p = self.PRICING.get(self.model, {"input": 0, "output": 0})
        return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]


# ---------------------------------------------------------------------------
# GEMINI -- context besar, rate limit harian tinggi di free tier.
# ---------------------------------------------------------------------------
class GeminiClient(LLMClient):
    provider_name = "gemini"

    PRICING = {
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    }

    def __init__(self, model: str = "gemini-2.0-flash"):
        from google import genai

        self.model = model
        self.client = genai.Client(
            api_key=settings.require_api_key("gemini"),
            http_options={'timeout': 30.0}
        )

    def _call(self, system_prompt, user_prompt):
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config={"system_instruction": system_prompt, "temperature": 0.1},
        )
        text = resp.text
        # Gemini API mengembalikan usage_metadata untuk hitung token
        in_tok = resp.usage_metadata.prompt_token_count
        out_tok = resp.usage_metadata.candidates_token_count
        return text, in_tok, out_tok

    def estimate_cost(self, input_tokens, output_tokens):
        p = self.PRICING.get(self.model, {"input": 0, "output": 0})
        return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]


# ---------------------------------------------------------------------------
# CLAUDE -- dipakai khusus untuk demo prompt caching & tool use (JD minta ini
# eksplisit sebagai bonus point). HEMAT pemakaian -- trial credit cuma ~$5.
# ---------------------------------------------------------------------------
class ClaudeClient(LLMClient):
    provider_name = "claude"

    PRICING = {
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
        "claude-haiku-4-5": {"input": 0.80, "output": 4.0, "cache_write": 1.0, "cache_read": 0.08},
    }

    def __init__(self, model: str = "claude-haiku-4-5"):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=settings.require_api_key("claude"))

    def _call(self, system_prompt, user_prompt):
        # CATATAN PENTING: parameter `temperature` SENGAJA tidak dikirim di sini.
        # Anthropic Python SDK v1.0+ (rilis Agustus 2026) MENGHAPUS temperature,
        # top_p, top_k dari signature method messages.create() untuk model-model
        # baru (Sonnet 5, Opus 4.7+, dst) -- bukan cuma dianggap deprecated di
        # sisi API, tapi jadi TypeError langsung di level Python kalau tetap
        # dikirim. Ini ketauan dari mypy, bukan dari baca changelog -- persis
        # jenis bug yang lolos kalau cuma smoke-test manual tanpa type checking.
        # Untuk use case kita (jawaban grounded/faktual, bukan kreatif), ini
        # tidak masalah -- kita memang tidak butuh kontrol sampling granular.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            timeout=30.0,
        )
        text = _extract_text(resp.content)
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    def generate_with_cache(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        Versi khusus untuk DEMO PROMPT CACHING (Hari 2).
        `cache_control` menandai system_prompt supaya Anthropic menyimpan versi
        ter-proses-nya di sisi server. Panggilan BERIKUTNYA dengan system_prompt
        IDENTIK akan kena tarif cache_read yang jauh lebih murah daripada
        tarif input normal -- ini yang nanti kita ukur & bandingkan.
        """
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
            timeout=30.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = _extract_text(resp.content)
        cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
        p = self.PRICING[self.model]
        cost = (
            (resp.usage.input_tokens / 1_000_000) * p["input"]
            + (cache_write / 1_000_000) * p["cache_write"]
            + (cache_read / 1_000_000) * p["cache_read"]
            + (resp.usage.output_tokens / 1_000_000) * p["output"]
        )
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider_name,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )

    def estimate_cost(self, input_tokens, output_tokens):
        p = self.PRICING.get(self.model, {"input": 0, "output": 0})
        return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]


def get_client(provider: str) -> LLMClient:
    """Factory function -- titik masuk tunggal untuk bikin client apapun."""
    if provider == "groq":
        return GroqClient()
    if provider == "gemini":
        return GeminiClient()
    if provider == "claude":
        return ClaudeClient()
    raise ValueError(f"Provider tidak dikenal: {provider}")
