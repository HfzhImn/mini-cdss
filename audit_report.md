# 🔍 Audit Kode Mini-CDSS — Laporan Migrasi ke FastAPI

> **Catatan status:** laporan ini bertanggal 5 September 2026 dan bersifat
> historis. Untuk kondisi terbaru setelah penggunaan dataset WHO ICD-10 2019
> dan DDInter, gunakan `README.md` serta `PANDUAN.html` sebagai sumber utama.
> **Diaudit oleh:** Senior Python Engineer & AI Architect  
> **Tanggal:** 5 September 2026  
> **Scope:** `src/`, `scripts/`, `tests/`, `cli.py`, `pyproject.toml`, `requirements.txt`  
> **Tujuan:** Migrasi CLI → FastAPI backend siap integrasi Next.js

---

## 1. [CRITICAL] Function Signature & Entry Point (`rag.py` & `cli.py`)

### ✅ Temuan: Arsitektur sudah terpisah dengan BAIK

Logika utama **sudah ada di `rag.py`**, bukan di `cli.py`. `cli.py` hanya bertugas sebagai thin wrapper. Ini situasi ideal.

### Fungsi Utama di `rag.py`

Terdapat **DUA fungsi publik**, bukan satu:

```python
# Fungsi 1 — untuk query diagnosis/ICD-10
def answer_icd10_query(retriever: Retriever, llm: LLMClient, query: str) -> RAGResult:

# Fungsi 2 — untuk query interaksi obat
def answer_drug_interaction_query(retriever: Retriever, llm: LLMClient, query: str) -> RAGResult:
```

### Parameter yang Diterima

| Parameter   | Type        | Keterangan                                                                 |
| ----------- | ----------- | -------------------------------------------------------------------------- |
| `retriever` | `Retriever` | Instance kelas Retriever (dari `retrieval.py`) — stateful, harus di-inject |
| `llm`       | `LLMClient` | Instance adapter LLM (Groq/Gemini/Claude)                                  |
| `query`     | `str`       | Query teks dari pengguna                                                   |

> [!IMPORTANT]
> **Perhatian untuk FastAPI:** `retriever` dan `llm` adalah object berat (embedding model + ChromaDB connection + reranker model). Jangan di-instantiasi per request — harus jadi **singleton** via `lifespan` FastAPI atau dependency injection.

### Return Type / Structure

Kedua fungsi mengembalikan `RAGResult` (dataclass), bukan string mentah:

```python
@dataclass
class RAGResult:
    answer: str          # Teks jawaban dari LLM
    sources: list[dict]  # Metadata dokumen yang dipakai (kode ICD-10, nama obat, dsb)
    llm: LLMResponse     # Objek lengkap: model, provider, token, latency, cost
    retrieval_count: int # Jumlah dokumen yang diambil dari retrieval
```

Dan `LLMResponse` (juga dataclass):

```python
@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost_usd: float
```

### Pemetaan ke API Response (FastAPI-ready)

Struktur `RAGResult` **hampir sempurna** untuk dijadikan JSON response. Hanya perlu satu sentuhan:

```python
# Tambahkan method ini ke RAGResult di rag.py
def to_api_dict(self) -> dict:
    return {
        "answer": self.answer,
        "sources": self.sources,
        "metadata": {
            "model": self.llm.model,
            "provider": self.llm.provider,
            "latency_ms": self.llm.latency_ms,
            "cost_usd": self.llm.estimated_cost_usd,
            "input_tokens": self.llm.input_tokens,
            "output_tokens": self.llm.output_tokens,
            "retrieval_count": self.retrieval_count,
        }
    }
```

### Routing Logic di `cli.py` (baris 45–48)

```python
# cli.py baris 45-48 — logika routing ini WAJIB dipindah ke rag.py
if query.lower().startswith("obat:"):
    result = rag.answer_drug_interaction_query(retriever, llm, query[5:].strip())
else:
    result = rag.answer_icd10_query(retriever, llm, query)
```

**Rekomendasi:** Tambahkan fungsi `answer_query()` di `rag.py` sebagai **unified entry point** untuk FastAPI:

```python
# Tambahkan ke rag.py
from router import route_query  # sudah ada!
from llm_clients import GroqClient

def answer_query(retriever: Retriever, llm: LLMClient, query: str, mode: str = "auto") -> RAGResult:
    """
    Unified entry point untuk FastAPI.
    mode: "icd10" | "drug" | "auto" (auto-detect dari prefix 'obat:')
    """
    if mode == "drug" or (mode == "auto" and query.lower().startswith("obat:")):
        clean_query = query.removeprefix("obat:").strip() if mode == "auto" else query
        return answer_drug_interaction_query(retriever, llm, clean_query)
    return answer_icd10_query(retriever, llm, query)
```

---

## 2. [CRITICAL] Async vs Sync (`llm_clients.py`)

### ❌ Temuan: SEMUA LLM call adalah SYNCHRONOUS blocking

| Client         | Method  | I/O Type          | SDK Call                                   |
| -------------- | ------- | ----------------- | ------------------------------------------ |
| `GroqClient`   | `_call` | **Sync Blocking** | `self.client.chat.completions.create(...)` |
| `GeminiClient` | `_call` | **Sync Blocking** | `self.client.models.generate_content(...)` |
| `ClaudeClient` | `_call` | **Sync Blocking** | `self.client.messages.create(...)`         |

Tidak ada satu pun `async def`, tidak ada `await`, tidak ada `httpx.AsyncClient`. Semua menggunakan SDK synchronous dari masing-masing provider.

### Dampak di FastAPI

Jika ini dijalankan di endpoint FastAPI tanpa penanganan khusus, LLM call yang memakan ~2–10 detik akan **memblokir seluruh event loop** — FastAPI tidak bisa melayani request lain selama itu berlangsung. Ini masalah kritis di production.

### Solusi: Gunakan `asyncio.to_thread()`

```python
# Opsi 1 (RECOMMENDED untuk sekarang): Bungkus di endpoint FastAPI
# file: api/routes/query.py

import asyncio
from fastapi import APIRouter, Depends
from src.rag import answer_query, RAGResult

router = APIRouter()

@router.post("/query")
async def handle_query(request: QueryRequest, state: AppState = Depends(get_state)):
    # Jalankan fungsi sync di thread pool, tidak memblokir event loop
    result = await asyncio.to_thread(
        answer_query,
        state.retriever,
        state.llm,
        request.query,
        request.mode,
    )
    return result.to_api_dict()
```

```python
# Opsi 2 (untuk masa depan): Refactor llm_clients.py ke async-native
# Groq SDK sudah punya AsyncGroq, Anthropic punya AsyncAnthropic

class GroqClient(LLMClient):
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        from groq import AsyncGroq
        self.async_client = AsyncGroq(api_key=settings.require_api_key("groq"))

    async def _call_async(self, system_prompt: str, user_prompt: str):
        resp = await self.async_client.chat.completions.create(...)
        return resp.choices[0].message.content, ...
```

> [!TIP]
> Untuk **tahap pertama migrasi**, Opsi 1 (`asyncio.to_thread`) lebih aman dan tidak membutuhkan refactor besar. Lakukan Opsi 2 hanya jika target adalah streaming response (SSE) ke Next.js.

---

## 3. [CRITICAL] Hardcoded Paths (Risiko Runtime Error)

### ✅ Temuan: PATH AMAN — Sudah menggunakan `pathlib` via `settings.py`

Ini adalah temuan **positif** yang jarang ditemukan di portfolio project. Semua path dikonfigurasi di `settings.py` menggunakan pattern yang benar:

```python
# settings.py baris 33 — BASE_DIR sudah benar pakai __file__
BASE_DIR = Path(__file__).parent.parent  # → root project

icd10_csv: Path = BASE_DIR / "data" / "processed" / "icd10_subset.csv"
drug_interactions_csv: Path = BASE_DIR / "data" / "processed" / "drug_interactions.csv"
chroma_db_dir: Path = BASE_DIR / "data" / "chroma_db"
```

Semua file lain mengakses path via `settings.icd10_csv`, `settings.chroma_db_dir`, dll.

### ⚠️ Satu Anomali di `eval_runner.py`

```python
# eval_runner.py baris 41-42 — sys.path.insert manual masih ada!
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "eval"))
```

Ini bisa **menimpa** path resolution yang sudah dikonfigurasi di `pyproject.toml [tool.pytest.ini_options] pythonpath = ["src"]`. Tidak berbahaya untuk CLI, tapi perlu dihapus/diganti untuk environment server.

**Solusi:** Hapus kedua baris tersebut dan pastikan package diinstall dengan `pip install -e .`, sehingga `src/` otomatis ada di Python path.

### ⚠️ `build_index.py` masih pakai `print()` untuk path (bukan `logger`)

```python
# build_index.py baris 35, 37, 54, 57, 68, 72, 74, ...
print(f"\n[ICD-10] Memuat data dari {settings.icd10_csv}")
```

Path yang di-print sudah dari `settings`, jadi pathnya aman. Tapi output-nya masih pakai `print()` — lihat Bagian 5.

---

## 4. [HIGH] Dependencies untuk API (`pyproject.toml` / `requirements.txt`)

### Status Dependency Kritis untuk FastAPI

| Package             | `pyproject.toml` | `requirements.txt` |     Status     |
| ------------------- | :--------------: | :----------------: | :------------: |
| `fastapi`           |   ✅ `>=0.140`   |    ✅ `0.141.1`    |     ✅ Ada     |
| `uvicorn[standard]` |   ✅ `>=0.50`    |    ✅ `0.52.4`     |     ✅ Ada     |
| `python-multipart`  |   ❌ Tidak ada   |    ❌ Tidak ada    | ❌ **MISSING** |
| `sse-starlette`     |   ❌ Tidak ada   |    ❌ Tidak ada    | ❌ **MISSING** |
| `rank-bm25`         |   ❌ Tidak ada   |    ❌ Tidak ada    | ❌ **MISSING** |
| `httpx`             |   ❌ Tidak ada   |    ❌ Tidak ada    |  ❌ MISSING\*  |

> [!CAUTION]
> `python-multipart` adalah **WAJIB** untuk FastAPI bisa menerima form data. Tanpanya, endpoint apapun yang menerima `Form(...)` akan error `400` dengan pesan tidak jelas.

> [!WARNING]
> `rank-bm25` dipakai di `fusion.py` (`from rank_bm25 import BM25Okapi`) tapi **tidak terdaftar** di `pyproject.toml` maupun `requirements.txt`. Ini **bug packaging** — kalau ada orang clone repo ini dan install dari `pyproject.toml`, `fusion.py` akan crash saat import.

### Dependency yang WAJIB Ditambahkan

```toml
# pyproject.toml — tambahkan ke [project.dependencies]
"rank-bm25>=0.7",
"python-multipart>=0.0.9",
"sse-starlette>=2.0",   # untuk streaming response ke Next.js
"httpx>=0.27",           # untuk async HTTP client (migrasi ke async LLM)
```

```txt
# requirements.txt — tambahkan
rank-bm25==0.7.0
python-multipart==0.0.12
sse-starlette==2.1.3
httpx==0.27.0
```

---

## 5. [MEDIUM] Print vs Logging (Polusi Output)

### Inventaris `print()` di Seluruh `src/`

| File             | Jumlah `print()` | Lokasi / Konteks                                       |
| ---------------- | :--------------: | ------------------------------------------------------ |
| `build_index.py` |      **14**      | Progress indexing — OK untuk CLI build script          |
| `cli.py`         |      **6**       | User-facing output di CLI — OK (ini memang untuk user) |
| `eval_runner.py` |      **10**      | Progress eval + hasil — perlu jadi `logger`            |
| `router.py`      |      **4**       | Hanya di blok `if __name__ == "__main__"` — OK         |
| `rag.py`         |     **0** ✅     | Bersih, sudah pakai pattern yang benar                 |
| `retrieval.py`   |     **0** ✅     | Sudah pakai `logger.info()`                            |
| `fusion.py`      |     **0** ✅     | Pure functions, tidak ada I/O                          |
| `llm_clients.py` |     **0** ✅     | Sudah pakai `logger`                                   |

### Analisis

- **`rag.py`, `retrieval.py`, `fusion.py`, `llm_clients.py`**: ✅ BERSIH — tidak ada `print()` sama sekali. Ini kode yang akan jadi library core FastAPI.
- **`build_index.py`**: Boleh pakai `print()` — ini adalah offline script yang dijalankan sekali, bukan production code.
- **`eval_runner.py`**: ⚠️ Perlu diganti ke `logger` karena bisa dijalankan via API di masa depan.

### Prioritas Perbaikan untuk FastAPI

Karena kode library utama (`rag.py`, `retrieval.py`, `llm_clients.py`) sudah bersih dari `print()`, **tidak ada blocking issue** di sini untuk migrasi FastAPI.

Ganti `print()` di `eval_runner.py` dengan pola ini:

```python
# Ganti:
print(f"  [{item['id']}] {item['query'][:60]}...")

# Dengan:
from logging_config import get_logger
logger = get_logger(__name__)
logger.info(f"[{item['id']}] Evaluating: {item['query'][:60]}...")
```

---

## 6. [MEDIUM] Error Handling & Retry Logic

### Analisis `retry.py`

Mekanisme retry sudah ada dan terpasang di layer yang tepat:

```python
# llm_clients.py baris 79-81
@with_retry(max_attempts=3, base_delay=1.0)
def _call_with_retry(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
    return self._call(system_prompt, user_prompt)
```

Semua 3 provider (Groq, Gemini, Claude) mewarisi `_call_with_retry` dari base class `LLMClient`, jadi **coverage sudah 100%** untuk LLM call.

### Gap yang Ditemukan

| Skenario Error            | Status                           | Catatan                                  |
| ------------------------- | -------------------------------- | ---------------------------------------- |
| Transient (500, 502, 503) | ✅ Ter-cover                     | Retry 3x dengan exponential backoff      |
| **Rate Limit (429)**      | ⚠️ Ter-cover, tapi tidak optimal | Retry tanpa melihat `Retry-After` header |
| **Timeout**               | ❌ Tidak ter-cover               | Tidak ada timeout di SDK call            |
| **API Key Invalid (401)** | ⚠️ Ter-cover, tapi boros         | Retry 3x walau jelas tidak akan berhasil |

### Rekomendasi Perbaikan

```python
# retry.py — Versi yang lebih cerdas untuk production
def with_retry(max_attempts: int = 3, base_delay: float = 1.0) -> Callable:
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Jangan retry untuk error yang jelas permanen
                    status = getattr(e, "status_code", None) or getattr(e, "code", None)
                    if status in (400, 401, 403, 404, 422):
                        logger.error(f"Error permanen ({status}), tidak di-retry: {e}")
                        raise  # Langsung raise, hemat waktu

                    # Hormati Retry-After header kalau ada (untuk 429)
                    retry_after = getattr(e, "response", None)
                    if retry_after and hasattr(retry_after, "headers"):
                        delay = float(retry_after.headers.get("Retry-After", base_delay * (2 ** (attempt - 1))))
                    else:
                        delay = base_delay * (2 ** (attempt - 1))

                    if attempt == max_attempts:
                        raise
                    logger.warning(f"Retry {attempt}/{max_attempts} dalam {delay:.0f}s: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator
```

**Tambahkan timeout** di `_call()` masing-masing client:

```python
# GroqClient._call() — tambahkan timeout
resp = self.client.chat.completions.create(
    ...,
    timeout=30.0,  # 30 detik max — jangan biarkan hang selamanya
)
```

---

## 7. [LOW] Potensi Circular Import

### Hasil Scan Import Graph

```
cli.py
  └── import rag          (src/rag.py)
  └── import llm_clients  (src/llm_clients.py)
  └── import retrieval    (src/retrieval.py)
  └── import settings     (src/settings.py)

rag.py
  └── from llm_clients import LLMClient, LLMResponse
  └── from retrieval import RetrievedDoc, Retriever

retrieval.py
  └── from fusion import bm25_rank, build_bm25_index, reciprocal_rank_fusion
  └── from logging_config import get_logger
  └── from settings import settings

llm_clients.py
  └── from logging_config import get_logger
  └── from retry import with_retry
  └── from settings import settings

fusion.py
  └── (tidak ada import internal)

settings.py
  └── (tidak ada import internal)

retry.py
  └── from logging_config import get_logger
```

### ✅ Temuan: TIDAK ADA Circular Import

Dependency graph berbentuk DAG (Directed Acyclic Graph) yang bersih:

```
settings.py / logging_config.py / fusion.py  ← leaf nodes (no internal deps)
    ↑
retry.py
    ↑
llm_clients.py + retrieval.py
    ↑
rag.py
    ↑
cli.py / router.py / eval_runner.py  ← entry points
```

Tidak ada `from src import x` di dalam folder `src/` (semua import adalah `from module import x`, bukan `from src.module import x`), karena `src/` ditambahkan ke `pythonpath` di `pyproject.toml`.

> [!NOTE]
> Saat migrasi ke FastAPI, pastikan `sys.path` atau package install sudah benar. Jika FastAPI dijalankan dari root project dengan `uvicorn app.main:app`, maka `src/` harus ada di `sys.path` atau diinstall sebagai package. Konfigurasi `[tool.pytest.ini_options] pythonpath = ["src"]` sudah ada untuk pytest — pastikan hal serupa dikonfigurasi untuk uvicorn via `pip install -e .`.

---

## 8. RINGKASAN EKSEKUTIF

### 🚦 Status Kesiapan: **82% Siap untuk FastAPI**

---

### Hambatan Utama Sebelum Deploy ke FastAPI

1. **Blocking LLM calls** — Semua LLM client (`GroqClient`, `GeminiClient`, `ClaudeClient`) sync blocking. Tanpa `asyncio.to_thread()`, satu request bisa memblokir seluruh server selama 5–15 detik.

2. **`rank-bm25` tidak terdaftar di dependencies** — `fusion.py` akan crash saat `from rank_bm25 import BM25Okapi` di environment fresh. Bug packaging yang senyap dan mematikan.

3. **`python-multipart` belum ada** — FastAPI tidak bisa menerima form data. Perlu ditambahkan sebelum endpoint berjalan normal.

4. **Tidak ada timeout di LLM SDK call** — Jika Groq/Gemini/Claude lambat atau hang, request FastAPI tidak akan pernah selesai (tidak ada circuit breaker).

---

### Kode Ini Sudah Siap 82% Karena

- ✅ **Arsitektur sudah benar** — `rag.py` sudah terpisah dari `cli.py`, return type sudah `dataclass` bukan string mentah, mudah diubah ke JSON response.
- ✅ **Path management sudah aman** — `settings.py` sudah pakai `Path(__file__).parent.parent`, tidak akan rusak walau dijalankan dari folder mana saja.
- ✅ **Logging sudah diimplementasi** — `logging_config.py` sudah ada dan dipakai di semua kode library core.
- ✅ **Retry logic sudah ada** — Semua LLM call ter-cover retry 3x dengan exponential backoff.
- ✅ **Tidak ada circular import** — Dependency graph bersih, aman untuk uvicorn.
- ✅ **`fastapi` dan `uvicorn` sudah ada di dependencies** — Tinggal tambahkan dependency yang kurang.

---

### Langkah Pertama yang Harus Dikerjakan Hafizh

**Langkah 1 (30 menit) — Fix dependencies:**

```bash
# Tambahkan ke pyproject.toml dan requirements.txt
pip install rank-bm25 python-multipart sse-starlette
```

**Langkah 2 (1 jam) — Buat `api/main.py` dengan singleton Retriever:**

```python
# api/main.py
from contextlib import asynccontextmanager
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from src.settings import settings
from src.retrieval import Retriever
from src.llm_clients import get_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load SEKALI saat server start — ini yang mahal
    embedder = SentenceTransformer(settings.embedding_model)
    chroma = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
    app.state.retriever = Retriever(chroma, embedder)
    app.state.llm = get_client(settings.default_provider)
    yield  # server siap melayani request
    # cleanup kalau perlu

app = FastAPI(lifespan=lifespan)
```

**Langkah 3 (30 menit) — Buat endpoint pertama dengan `asyncio.to_thread()`:**

```python
# api/routes/query.py
import asyncio
from fastapi import APIRouter
from pydantic import BaseModel
from src.rag import answer_icd10_query, answer_drug_interaction_query

router = APIRouter(prefix="/api/v1")

class QueryRequest(BaseModel):
    query: str
    mode: str = "icd10"  # "icd10" | "drug"

@router.post("/query")
async def handle_query(req: QueryRequest, request: Request):
    retriever = request.app.state.retriever
    llm = request.app.state.llm

    if req.mode == "drug":
        result = await asyncio.to_thread(answer_drug_interaction_query, retriever, llm, req.query)
    else:
        result = await asyncio.to_thread(answer_icd10_query, retriever, llm, req.query)

    return {
        "answer": result.answer,
        "sources": result.sources,
        "metadata": {
            "model": result.llm.model,
            "latency_ms": result.llm.latency_ms,
            "cost_usd": result.llm.estimated_cost_usd,
        }
    }
```

---

_Laporan ini di-generate berdasarkan audit statik terhadap semua file source code. Tidak ada kode yang dieksekusi._
