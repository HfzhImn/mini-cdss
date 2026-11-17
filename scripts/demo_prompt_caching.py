"""
demo_prompt_caching.py
------------------------
TUJUAN:
Mendemonstrasikan PROMPT CACHING -- salah satu teknik cost optimization
yang eksplisit disebut JD ("Optimize prompts for cost via prompt caching,
batching, and model routing" dan "Experience with Anthropic SDK (prompt
caching, tool use)" di bonus points).

KONSEP: Kalau kamu mengirim system prompt / konteks yang SAMA berkali-kali
(contoh nyata: system prompt CDSS kita yang panjang, atau konteks hasil
retrieval yang besar), Anthropic bisa menyimpan versi "siap pakai"-nya di
server mereka. Panggilan BERIKUTNYA dengan konteks identik jadi jauh lebih
murah (cache read jauh di bawah harga input normal) dan sedikit lebih cepat
(tidak perlu re-proses ulang konteks dari nol).

Di production CDSS sungguhan, ini penting BANGET karena system prompt +
context klinis yang dikirim ke LLM untuk SETIAP query dokter itu sering
kali panjang dan repetitif antar-query dalam sesi yang sama -- caching bisa
memangkas cost drastis (ini yang JD maksud dengan "reduce cost 40%+").

⚠️ PENTING: jalankan script ini HEMAT. Trial credit Anthropic cuma ~$5.
Script ini cuma melakukan 2 API call (1x cache write, 1x cache read) --
JANGAN loop berkali-kali untuk "meyakinkan diri", 2x sudah cukup untuk
bukti konsep dan angka perbandingan yang valid.

CARA PAKAI:
    export ANTHROPIC_API_KEY=xxx   (atau taruh di .env)
    python3 src/demo_prompt_caching.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# NOTE: import di bawah baris sys.path.insert() ini SENGAJA melanggar
# aturan linter soal "import harus di paling atas file" -- dikecualikan
# secara eksplisit lewat komentar khusus di tiap baris import (BUKAN
# dibiarkan diam-diam). sys.path perlu di-set dulu sebelum Python bisa
# menemukan module di folder src/, jadi urutan ini memang wajib begini.
from llm_clients import ClaudeClient  # noqa: E402
from rag import SYSTEM_PROMPT  # noqa: E402
from settings import settings  # noqa: E402

# Simulasikan konteks retrieval yang panjang (mirip situasi nyata: system
# prompt + banyak dokumen ICD-10 hasil retrieval yang digabung jadi konteks)
LARGE_CONTEXT_SUFFIX = """

REFERENSI TAMBAHAN (contoh konteks besar yang biasanya berulang antar-query
dalam 1 sesi dokter yang sama -- di produksi nyata ini bisa berisi ratusan
baris dokumentasi klinis, guideline, atau riwayat retrieval sebelumnya):
""" + "\n".join([f"- Catatan referensi klinis nomor {i}: informasi standar yang sama dipakai berulang." for i in range(1, 80)])


def main():
    try:
        settings.require_api_key("claude")
    except ValueError as e:
        print(e)
        return

    client = ClaudeClient(model="claude-haiku-4-5")
    full_system_prompt = SYSTEM_PROMPT + LARGE_CONTEXT_SUFFIX

    print("=== Panggilan 1: CACHE WRITE (pertama kali, belum ada cache) ===")
    r1 = client.generate_with_cache(full_system_prompt, "Apa kode ICD-10 untuk kolera?")
    print(f"Latency   : {r1.latency_ms:.0f} ms")
    print(f"Input tok : {r1.input_tokens}")
    print(f"Cost      : ${r1.estimated_cost_usd:.6f}")

    print("\n=== Panggilan 2: CACHE READ (system prompt identik, harus lebih murah) ===")
    r2 = client.generate_with_cache(full_system_prompt, "Apa kode ICD-10 untuk faringitis akut?")
    print(f"Latency   : {r2.latency_ms:.0f} ms")
    print(f"Input tok : {r2.input_tokens}")
    print(f"Cost      : ${r2.estimated_cost_usd:.6f}")

    savings_pct = (1 - r2.estimated_cost_usd / r1.estimated_cost_usd) * 100 if r1.estimated_cost_usd > 0 else 0
    print(f"\n📊 Penghematan cost panggilan ke-2 vs ke-1: {savings_pct:.1f}%")
    print("(Catat angka ini untuk README/demo video -- ini bukti konkret cost optimization)")


if __name__ == "__main__":
    main()
