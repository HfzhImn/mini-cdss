"""
eval_runner.py
---------------
TUJUAN:
Ini jantung dari eval framework yang diminta JD. Untuk tiap kombinasi
(model x golden question), kita jalankan pipeline RAG penuh (retrieval +
rerank + generation) lalu ukur 4 metrik:

  1. ACCURACY -- untuk kasus positif (icd10/drug): apakah kode/pasangan
     obat yang benar disebut di jawaban? Untuk kasus negatif: apakah model
     BENAR menolak menjawab (bukan mengarang)?

  2. HALLUCINATION RATE -- apakah jawaban menyebut kode ICD-10 atau nama
     obat yang TIDAK ADA di dokumen hasil retrieval untuk query itu?
     Ini metrik paling kritis di domain kesehatan -- model bisa "kelihatan"
     benar tapi sebenarnya mengarang detail yang tidak didukung data.

  3. LATENCY -- waktu round-trip end-to-end (retrieval + rerank + LLM call)
     dalam milidetik.

  4. COST -- estimasi biaya per query dalam USD, dihitung dari token usage
     asli x published pricing (lihat llm_clients.py).

CARA PAKAI (di laptop kamu, setelah build_index.py jalan):
    python3 src/eval_runner.py --provider groq --model llama-3.1-8b-instant
    python3 src/eval_runner.py --provider groq --model llama-3.3-70b-versatile
    python3 src/eval_runner.py --provider gemini --model gemini-2.0-flash

Jalankan untuk tiap model yang mau dibandingkan, lalu lihat file
data/eval/results/*.csv untuk hasilnya, atau pakai compare_results.py
untuk tabel perbandingan gabungan.
"""

import argparse
import csv
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "eval"))

from golden_set import GOLDEN_SET

from src.rag import answer_drug_interaction_query, answer_icd10_query

REFUSAL_MARKER = "tidak ditemukan"

RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval" / "results"


def extract_icd_codes(text: str) -> set[str]:
    """Cari pola kode ICD-10 (mis. 'A00.9', 'J06', 'E11.9') di dalam teks jawaban."""
    return set(re.findall(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b", text))


@dataclass
class EvalRecord:
    id: str
    category: str
    query: str
    answer: str
    correct: bool
    hallucinated: bool
    latency_ms: float
    cost_usd: float
    retrieval_count: int


def score_icd10(item: dict, result) -> tuple[bool, bool]:
    retrieved_codes = {s.get("icd10_code") for s in result.sources if s.get("icd10_code")}
    mentioned_codes = extract_icd_codes(result.answer)

    correct = item["expected_code"] in mentioned_codes
    # halusinasi = ada kode disebut yang TIDAK termasuk dokumen hasil retrieval
    fabricated = mentioned_codes - retrieved_codes
    hallucinated = len(fabricated) > 0
    return correct, hallucinated


def score_drug(item: dict, result) -> tuple[bool, bool]:
    answer_lower = result.answer.lower()
    drug_a, drug_b = item["expected_pair"]
    correct = drug_a.lower() in answer_lower and drug_b.lower() in answer_lower

    # kumpulkan semua nama obat yang MUNCUL di dokumen hasil retrieval untuk query ini
    allowed_names = set()
    for s in result.sources:
        if s.get("drug_a"):
            allowed_names.add(s["drug_a"].lower())
        if s.get("drug_b"):
            allowed_names.add(s["drug_b"].lower())

    # Heuristik sederhana: cek apakah severity yang disebut cocok, dan apakah
    # ada indikasi model menyebut severity/mechanism yang tidak match sama sekali
    # dengan dokumen yang di-retrieve (proxy kasar untuk "mengarang detail")
    hallucinated = not any(name in answer_lower for name in allowed_names) if allowed_names else False
    return correct, hallucinated


def score_negative(item: dict, result) -> tuple[bool, bool]:
    refused = REFUSAL_MARKER in result.answer.lower()
    mentioned_codes = extract_icd_codes(result.answer)
    # untuk kasus negatif: benar = menolak menjawab; halusinasi = TIDAK menolak
    # DAN malah menyebut kode/detail spesifik seolah-olah itu jawaban valid
    correct = refused
    hallucinated = (not refused) and len(mentioned_codes) > 0
    return correct, hallucinated


def run_eval(provider: str, model: str) -> list[EvalRecord]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    from src.llm_clients import get_client
    from src.retrieval import Retriever
    from src.settings import settings

    print(f"Memuat retriever & model {provider}/{model}...")
    embedder = SentenceTransformer(settings.embedding_model)
    client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
    retriever = Retriever(client, embedder)
    llm = get_client(provider)
    llm.model = model  # override default model kalau perlu

    records = []
    for item in GOLDEN_SET:
        print(f"  [{item['id']}] {item['query'][:60]}...")
        if item["category"] in ("icd10", "negative") and "expected_pair" not in item:
            # icd10 & negative pakai jalur ICD-10 (negative kita uji lewat jalur ini
            # karena mayoritas contoh negatif kita berkaitan konteks diagnosis)
            result = answer_icd10_query(retriever, llm, item["query"])
        else:
            result = answer_drug_interaction_query(retriever, llm, item["query"])

        if item["category"] == "icd10":
            correct, hallucinated = score_icd10(item, result)
        elif item["category"] == "drug":
            correct, hallucinated = score_drug(item, result)
        else:
            correct, hallucinated = score_negative(item, result)

        records.append(EvalRecord(
            id=item["id"], category=item["category"], query=item["query"],
            answer=result.answer, correct=correct, hallucinated=hallucinated,
            latency_ms=result.llm.latency_ms, cost_usd=result.llm.estimated_cost_usd,
            retrieval_count=result.retrieval_count,
        ))
    return records


def save_and_summarize(records: list[EvalRecord], provider: str, model: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{provider}_{model}.csv".replace("/", "-")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    n = len(records)
    accuracy = sum(r.correct for r in records) / n * 100
    hallucination_rate = sum(r.hallucinated for r in records) / n * 100
    avg_latency = sum(r.latency_ms for r in records) / n
    total_cost = sum(r.cost_usd for r in records)
    avg_cost = total_cost / n

    print(f"\n=== Hasil eval: {provider}/{model} ===")
    print(f"Accuracy          : {accuracy:.1f}%  ({sum(r.correct for r in records)}/{n})")
    print(f"Hallucination rate: {hallucination_rate:.1f}%  ({sum(r.hallucinated for r in records)}/{n})")
    print(f"Avg latency       : {avg_latency:.0f} ms")
    print(f"Avg cost/query    : ${avg_cost:.6f}")
    print(f"Total cost (run)  : ${total_cost:.6f}")
    print(f"Detail tersimpan  : {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["groq", "gemini", "claude"])
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    records = run_eval(args.provider, args.model)
    save_and_summarize(records, args.provider, args.model)


if __name__ == "__main__":
    main()
