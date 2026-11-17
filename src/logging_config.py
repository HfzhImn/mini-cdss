"""
logging_config.py
------------------
TUJUAN:
`print()` itu cocok untuk output yang MEMANG ditujukan ke user (mis. CLI
interaktif kita di cli.py). Tapi untuk kode LIBRARY (retrieval, indexing,
eval runner), print() punya masalah:
  - Tidak ada level (info vs warning vs error semua kelihatan sama)
  - Tidak bisa dimatikan/diarahkan tanpa ubah kode
  - Tidak ada timestamp, tidak ada nama module asal pesan
  - Susah di-parsing otomatis kalau nanti proyek ini jalan di server
    (Hari nanti: FastAPI) dan butuh agregasi log

Solusinya: logging module bawaan Python, dikonfigurasi SEKALI di sini,
dipakai oleh semua module lain lewat `get_logger(__name__)`.

Format log sengaja dibuat sederhana (bukan JSON) untuk development lokal
supaya gampang dibaca di terminal. Kalau nanti proyek ini benar-benar
di-deploy, format JSON terstruktur lebih baik untuk log aggregation
(CloudWatch, Datadog, dll) -- itu perubahan 1 baris di sini nanti,
tidak perlu ubah kode yang manggil logger di tempat lain. Ini keuntungan
utama dari sentralisasi konfigurasi logging di 1 tempat.
"""

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return  # jangan setup dua kali (mis. kalau diimpor dari beberapa entry point)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Library pihak ketiga sering sangat verbose di level INFO/DEBUG --
    # kita naikkan threshold-nya khusus supaya log kita sendiri tidak
    # tenggelam di antara ribuan baris log HTTP internal library lain.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
