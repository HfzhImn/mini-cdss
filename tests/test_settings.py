"""
test_settings.py
------------------
TUJUAN:
Menguji bagian PALING PENTING dari settings.py: pesan error yang jelas
saat API key belum diset. Ini yang membedakan pengalaman debugging
"KeyError: 'GROQ_API_KEY'" yang membingungkan (versi lama) dengan pesan
error yang langsung kasih tau HARUS ngapain (versi settings.py sekarang).
"""

import pytest
from pydantic import ValidationError

from settings import Settings


class TestRequireApiKey:
    def test_raises_clear_error_when_key_missing(self):
        s = Settings(groq_api_key=None, gemini_api_key=None, anthropic_api_key=None)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            s.require_api_key("groq")

    def test_returns_key_when_present(self):
        s = Settings(groq_api_key="fake-key-123")
        assert s.require_api_key("groq") == "fake-key-123"

    def test_error_message_names_the_env_var_to_fix(self):
        # Kontrak penting: pesan error HARUS menyebut nama environment
        # variable yang perlu diisi, supaya orang yang baca error tidak
        # perlu buka kode sumber untuk tahu apa yang harus dilakukan.
        s = Settings(anthropic_api_key=None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            s.require_api_key("claude")

    @pytest.mark.parametrize(
        "provider,env_var",
        [("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY"), ("claude", "ANTHROPIC_API_KEY")],
    )
    def test_every_provider_has_matching_env_var_in_error(self, provider, env_var):
        s = Settings(groq_api_key=None, gemini_api_key=None, anthropic_api_key=None)
        with pytest.raises(ValueError, match=env_var):
            s.require_api_key(provider)


class TestSettingsDefaults:
    def test_retrieve_top_k_must_be_positive(self):
        # Field(gt=0) di settings.py harus benar-benar menegakkan validasi
        # ini -- kalau tidak, orang bisa taruh 0 atau angka negatif di .env
        # dan errornya baru muncul jauh di dalam retrieval.py, susah dilacak
        # balik ke akar masalahnya.
        with pytest.raises(ValidationError):
            Settings(retrieve_top_k=0)

    def test_rerank_top_n_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(rerank_top_n=-1)

    def test_default_provider_is_groq(self):
        s = Settings()
        assert s.default_provider == "groq"
