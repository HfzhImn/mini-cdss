"""
demo_tool_use.py
------------------
TUJUAN:
Mendemonstrasikan TOOL USE / AGENTIC BEHAVIOR -- disebut eksplisit di JD
sebagai contoh proyek ("Add tool-use / agentic behavior to clinical
assistants (lab lookups, drug DB queries)") dan bonus point ("Anthropic
SDK (prompt caching, tool use)").

BEDA dengan pipeline RAG biasa (rag.py):
  - RAG biasa: kita SELALU retrieve dulu, lalu dump semua hasilnya ke
    prompt, baru model menjawab. Model tidak punya kendali atas pencarian.
  - Tool use: model sendiri yang MEMUTUSKAN kapan perlu mencari, apa
    query pencariannya, dan bisa memanggil tool BERULANG KALI kalau
    hasil pertama belum cukup (mis. cari dulu by keyword umum, lalu
    perhalus pencarian) -- ini pola "agentic" yang lebih fleksibel.

Contoh use case nyata dari JD: "lab lookups, drug DB queries" -- yaitu
model butuh mengambil data terstruktur dari sistem lain secara mandiri
di tengah reasoning-nya, bukan cuma sekali di awal.

Di file ini kita definisikan 1 tool: `lookup_icd_code_by_keyword`, yang
mencari langsung ke dataset ICD-10 kita (bukan lewat embedding -- ini
sengaja exact/substring match untuk menunjukkan tool bisa berupa APAPUN,
tidak harus semantic search).

CARA PAKAI:
    export ANTHROPIC_API_KEY=xxx
    python3 src/demo_tool_use.py
"""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd  # noqa: E402

from settings import settings  # noqa: E402

TOOL_DEFINITION = {
    "name": "lookup_icd_code_by_keyword",
    "description": (
        "Cari kode ICD-10 berdasarkan kata kunci yang muncul di deskripsi "
        "diagnosis. Gunakan tool ini setiap kali user menanyakan kode untuk "
        "suatu penyakit/gejala. Bisa dipanggil berkali-kali dengan kata kunci "
        "berbeda kalau pencarian pertama belum menemukan hasil yang cocok."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "Kata kunci pencarian, dalam Bahasa Inggris (mis. 'cholera', 'diabetes type 2')"}
        },
        "required": ["keyword"],
    },
}


def lookup_icd_code_by_keyword(keyword: str) -> list[dict]:
    """Implementasi asli dari tool -- exact substring search ke CSV kita."""
    df = pd.read_csv(settings.icd10_csv)
    matches = df[df["description"].str.contains(keyword, case=False, na=False)]
    # `.to_dict("records")` dari pandas mengembalikan tipe `Any` menurut type
    # stub-nya (keterbatasan stub pandas, bukan indikasi bug) -- cast eksplisit
    # ke tipe yang kita janjikan di return annotation supaya konsumen fungsi
    # ini tetap dapat jaminan tipe yang jelas.
    records: list[dict] = matches.head(5)[["icd10_code", "description"]].to_dict("records")
    return records


def run_agentic_loop(client, model: str, user_query: str, max_turns: int = 4):
    """
    Loop agentic sederhana: kirim query -> kalau model minta panggil tool,
    kita jalankan tool-nya, kirim hasilnya balik -> ulangi sampai model
    kasih jawaban final (bukan tool_use lagi) atau max_turns tercapai.
    """
    # Anotasi tipe eksplisit di sini PENTING: format pesan Anthropic itu
    # heterogen -- `content` bisa berupa string biasa (giliran pertama) ATAU
    # list of content block dict (giliran-giliran berikutnya, waktu kirim
    # balik hasil tool). Kalau dibiarkan mypy infer dari literal pertama
    # (`{"role": "user", "content": user_query}`), tipe yang ke-infer cuma
    # `dict[str, str]` -- lalu meledak "incompatible type" begitu kita
    # append dict dengan content berupa list. Ini isu tipe asli yang
    # mencerminkan struktur data yang memang heterogen, bukan sekadar
    # 'suruh diam' linter dengan Any di sana-sini.
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_query}]

    for turn in range(max_turns):
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=(
                "Kamu asisten pencarian kode ICD-10. Pakai tool "
                "lookup_icd_code_by_keyword untuk mencari kode yang relevan "
                "sebelum menjawab. Jangan pernah menjawab dari ingatanmu sendiri."
            ),
            tools=[TOOL_DEFINITION],
            messages=messages,
        )

        if resp.stop_reason != "tool_use":
            # model sudah kasih jawaban final (bukan minta tool lagi)
            final_text = "".join(b.text for b in resp.content if b.type == "text")
            return final_text, turn + 1

        # model minta panggil tool -- jalankan & kirim hasilnya balik
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use" and block.name == "lookup_icd_code_by_keyword":
                keyword = block.input["keyword"]
                print(f"  [Turn {turn+1}] Model memanggil tool dengan keyword: {keyword!r}")
                result = lookup_icd_code_by_keyword(keyword)
                print(f"  [Turn {turn+1}] Tool mengembalikan {len(result)} hasil")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        messages.append({"role": "user", "content": tool_results})

    return "(max_turns tercapai tanpa jawaban final)", max_turns


def main():
    try:
        api_key = settings.require_api_key("claude")
    except ValueError as e:
        print(e)
        return

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    query = "Carikan kode ICD-10 untuk diabetes tipe 2"
    print(f"Query: {query}\n")
    answer, n_turns = run_agentic_loop(client, "claude-haiku-4-5", query)
    print(f"\nJawaban final (setelah {n_turns} turn):\n{answer}")


if __name__ == "__main__":
    main()
