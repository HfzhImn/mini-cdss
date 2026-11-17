"""
settings.py
------------
TUJUAN:
Menggantikan config.py versi lama (konstanta Python biasa) dengan
pydantic-settings. Bedanya penting:

  1. VALIDASI OTOMATIS -- kalau RETRIEVE_TOP_K diisi string bukan angka
     di .env, ini akan gagal LANGSUNG saat startup dengan pesan error
     jelas, bukan error aneh di tengah runtime saat angkanya dipakai.

  2. SATU SUMBER KEBENARAN UNTUK API KEY -- sebelumnya llm_clients.py
     langsung baca os.environ["GROQ_API_KEY"] di beberapa tempat
     berbeda. Kalau key belum diset, errornya KeyError yang membingungkan.
     Sekarang, semua lewat objek `settings` -- kalau key tidak ada,
     pesannya jelas menyebut variabel mana yang kurang.

  3. GAMPANG DI-OVERRIDE UNTUK TESTING -- di file test, kita bisa
     bikin instance Settings() baru dengan nilai custom tanpa perlu
     mengubah environment variable asli.

CARA PAKAI di module lain:
    from settings import settings
    print(settings.embedding_model)
    print(settings.icd10_csv)
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # jangan error kalau .env punya variabel lain yang tidak dipakai di sini
    )

    # --- Path data ---
    icd10_csv: Path = BASE_DIR / "data" / "processed" / "icd10_subset.csv"
    drug_interactions_csv: Path = BASE_DIR / "data" / "processed" / "drug_interactions.csv"
    chroma_db_dir: Path = BASE_DIR / "data" / "chroma_db"

    # --- Model ---
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- ChromaDB collections ---
    icd10_collection: str = "icd10_codes"
    drug_interaction_collection: str = "drug_interactions"

    # --- Retrieval params ---
    retrieve_top_k: int = Field(default=15, gt=0)
    rerank_top_n: int = Field(default=4, gt=0)

    # --- API keys (opsional di level Settings -- validasi "wajib ada"
    # dilakukan di titik pemakaian oleh masing-masing LLM client, supaya
    # orang yang cuma mau pakai Groq tidak dipaksa isi API key Gemini/Claude
    # juga cuma untuk menjalankan aplikasi) ---
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None

    default_provider: str = "groq"
    cors_origins: str = "http://localhost:3000"

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def require_api_key(self, provider: str) -> str:
        """Ambil API key provider tertentu, error jelas kalau belum diset."""
        key_map = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "claude": self.anthropic_api_key,
        }
        env_var_map = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
        }
        value = key_map.get(provider)
        if not value:
            raise ValueError(
                f"API key untuk provider '{provider}' belum diset. "
                f"Isi {env_var_map[provider]} di file .env kamu "
                f"(lihat .env.example)."
            )
        return value


settings = Settings()
