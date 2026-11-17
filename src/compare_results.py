"""
compare_results.py
-------------------
TUJUAN:
Setelah kamu jalankan eval_runner.py untuk beberapa model (Hari 2, di
laptop kamu dengan API key asli), script ini menggabungkan semua hasil
jadi 1 tabel perbandingan -- inilah yang akan jadi bagian paling penting
di README & demo video kamu, karena ini BUKTI KONKRET kamu bisa melakukan
trade-off analysis antar model, bukan cuma "pakai LLM X karena populer".

CARA PAKAI (setelah minimal 2 file hasil ada di data/eval/results/):
    python3 src/compare_results.py
"""

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval" / "results"


def load_results():
    summaries = []
    for csv_file in sorted(RESULTS_DIR.glob("*.csv")):
        rows = list(csv.DictReader(open(csv_file)))
        if not rows:
            continue
        n = len(rows)
        accuracy = sum(r["correct"] == "True" for r in rows) / n * 100
        hallucination = sum(r["hallucinated"] == "True" for r in rows) / n * 100
        avg_latency = sum(float(r["latency_ms"]) for r in rows) / n
        avg_cost = sum(float(r["cost_usd"]) for r in rows) / n
        summaries.append({
            "name": csv_file.stem,
            "accuracy": accuracy,
            "hallucination": hallucination,
            "avg_latency_ms": avg_latency,
            "avg_cost_usd": avg_cost,
        })
    return summaries


def print_table(summaries):
    if not summaries:
        print("Belum ada hasil eval. Jalankan eval_runner.py dulu untuk minimal 1 model.")
        return

    header = f"{'Model':<30} {'Accuracy':>10} {'Halluc.':>10} {'Latency':>12} {'Cost/query':>12}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['name']:<30} {s['accuracy']:>9.1f}% {s['hallucination']:>9.1f}% "
            f"{s['avg_latency_ms']:>10.0f}ms {'$' + format(s['avg_cost_usd'], '.6f'):>12}"
        )

    print("\n--- Markdown (siap tempel ke README) ---")
    print("| Model | Accuracy | Hallucination rate | Avg latency | Avg cost/query |")
    print("|---|---|---|---|---|")
    for s in summaries:
        print(
            f"| {s['name']} | {s['accuracy']:.1f}% | {s['hallucination']:.1f}% "
            f"| {s['avg_latency_ms']:.0f} ms | ${s['avg_cost_usd']:.6f} |"
        )


if __name__ == "__main__":
    print_table(load_results())
