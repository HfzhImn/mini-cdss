"""
retry.py
--------
TUJUAN:
Panggilan API eksternal (Groq/Gemini/Claude) bisa gagal SEMENTARA -- rate
limit sesaat, network blip, error 5xx dari server mereka. Ini beda dari
error PERMANEN (API key salah, request malformed) yang percuma diulang.

Untuk proyek 3 hari kemarin, error apapun langsung bikin script crash.
Untuk proyek yang matang, transient error harus di-retry otomatis dengan
"exponential backoff" -- jeda antar percobaan makin lama (1s, 2s, 4s, ...)
supaya tidak memperparah rate limit yang sedang terjadi.

CATATAN: decorator ini sengaja generic (retry semua Exception), bukan
cuma exception spesifik dari tiap provider. Alasannya: SDK Groq/Gemini/
Claude punya exception class masing-masing yang berbeda-beda, dan kita
tidak mau logic retry kita bergantung erat pada detail internal 3 SDK
berbeda. Trade-off yang disadari: kita jadi retry juga error yang
SEBENARNYA permanen (mis. API key salah) sampai max_attempts habis --
untuk proyek skala ini itu bisa diterima (paling boros beberapa detik),
tapi ini poin bagus untuk didiskusikan kalau ditanya interviewer:
"kalau mau lebih presisi, retry cuma untuk exception yang punya atribut
status_code di 429/500/502/503, error lain langsung raise tanpa retry."
"""

import functools
import time
from collections.abc import Callable
from typing import TypeVar

from src.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def with_retry(max_attempts: int = 3, base_delay: float = 1.0) -> Callable:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Jangan retry untuk error yang jelas permanen
                    status = getattr(e, "status_code", None) or getattr(e, "code", None)
                    if status in (400, 401, 403, 404, 422):
                        logger.error(f"Error permanen ({status}) dari {func.__qualname__}, tidak di-retry: {e}")
                        raise
                    
                    if attempt == max_attempts:
                        logger.error(f"{func.__qualname__} gagal setelah {max_attempts} percobaan: {e}")
                        raise

                    # Hormati Retry-After header kalau ada (untuk 429)
                    response_obj = getattr(e, "response", None)
                    if response_obj and hasattr(response_obj, "headers") and "Retry-After" in response_obj.headers:
                        try:
                            delay = float(response_obj.headers["Retry-After"])
                        except ValueError:
                            delay = base_delay * (2 ** (attempt - 1))
                    else:
                        delay = base_delay * (2 ** (attempt - 1))

                    logger.warning(
                        f"{func.__qualname__} gagal (percobaan {attempt}/{max_attempts}): "
                        f"{e} -- retry dalam {delay:.0f}s"
                    )
                    time.sleep(delay)
            raise RuntimeError("Unreachable")

        return wrapper

    return decorator
