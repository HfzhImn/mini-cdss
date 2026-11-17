"""
golden_set.py
--------------
TUJUAN:
Ini entry point utama eval framework -- menggabungkan 2 tier:

  1. CURATED_SET (golden_set_curated.py) -- ditulis tangan, fokus ke
     kesulitan & variasi natural. Lihat docstring di file itu.

  2. Tier "generated" di bawah ini -- dibuat OTOMATIS dari isi
     icd10_subset.csv & drug_interactions.csv memakai beberapa template
     pertanyaan (ID & EN). BUKAN ditulis manual satu-satu.

KENAPA GENERATED SET DIJAMIN BENAR "BY CONSTRUCTION":
Query-nya dibuat dari deskripsi/pasangan obat yang sudah ada label
code/severity-nya di dataset -- jadi ground truth-nya otomatis konsisten,
tidak mungkin salah ketik expected_code seperti risiko kalau ditulis
manual satu-satu untuk 100+ entri.

KENAPA INI OTOMATIS IKUT UPDATE KALAU DATASET BERUBAH:
generate_breadth_examples() membaca CSV di runtime, bukan hardcode list
statis. Golden set generated ini membaca CSV WHO ICD-10 dan DDInter hasil
preprocessing setiap kali di-import, sehingga otomatis menyesuaikan tanpa
perlu ditulis ulang manual. Tier curated (golden_set_curated.py) TIDAK
otomatis ikut berubah -- itu perlu diverifikasi ulang manual jika sumber
data diganti.

RANDOM SEED TETAP (42) supaya golden set yang dihasilkan REPRODUCIBLE --
tiap kali file ini di-import, hasilnya sama persis, bukan berubah-ubah
random tiap run (yang akan bikin hasil eval antar-run tidak bisa
dibandingkan secara adil).
"""

import random
from pathlib import Path
from typing import Any

import pandas as pd
from golden_set_curated import CURATED_SET

DATA_DIR = Path(__file__).parent.parent / "processed"
SEED = 42

ICD10_QUERY_TEMPLATES = [
    "kode ICD-10 untuk {desc}",
    "what is the ICD-10 code for {desc}",
    "berapa kode diagnosis untuk {desc}?",
    "ICD-10 code for {desc}",
]

DRUG_QUERY_TEMPLATES = [
    "apakah {a} aman dikombinasi dengan {b}?",
    "interaksi antara {a} dan {b}",
    "is it safe to combine {a} with {b}?",
    "bahaya kombinasi {a} dengan {b}",
]

# Berapa contoh diambil per bab ICD-10 / per interaksi obat -- angka ini
# sengaja dibatasi (bukan "semua 596 kode") supaya eval run tidak terlalu
# lama & mahal untuk dijalankan berulang kali selama development. Sesuaikan
# ke atas kalau nanti butuh eval set lebih besar untuk laporan akhir.
ICD10_SAMPLES_PER_CHAPTER = 4
DRUG_SAMPLES = 20


def generate_icd10_breadth(seed: int = SEED) -> list[dict[str, Any]]:
    df = pd.read_csv(DATA_DIR / "icd10_subset.csv")
    rng = random.Random(seed)

    examples = []
    for _chapter, group in df.groupby("chapter_id"):
        n = min(ICD10_SAMPLES_PER_CHAPTER, len(group))
        sampled = group.sample(n=n, random_state=seed)
        for i, row in enumerate(sampled.itertuples()):
            template = ICD10_QUERY_TEMPLATES[i % len(ICD10_QUERY_TEMPLATES)]
            # Deskripsi asli sering berupa Title Case/formal -- di-lowercase
            # supaya query terasa seperti pertanyaan natural, bukan
            # copy-paste literal dari tabel data.
            query = template.format(desc=row.description.lower())
            examples.append({
                "id": f"gen_icd_{row.icd10_code.replace('.', '_')}",
                "category": "icd10",
                "query": query,
                "expected_code": row.icd10_code,
            })
    rng.shuffle(examples)
    return examples


def generate_drug_breadth(seed: int = SEED, n_samples: int = DRUG_SAMPLES) -> list[dict[str, Any]]:
    df = pd.read_csv(DATA_DIR / "drug_interactions.csv")
    n = min(n_samples, len(df))
    sampled = df.sample(n=n, random_state=seed)

    examples = []
    for i, row in enumerate(sampled.itertuples()):
        template = DRUG_QUERY_TEMPLATES[i % len(DRUG_QUERY_TEMPLATES)]
        query = template.format(a=row.drug_a, b=row.drug_b)
        examples.append({
            "id": f"gen_drug_{i:03d}",
            "category": "drug",
            "query": query,
            "expected_pair": (row.drug_a, row.drug_b),
        })
    return examples


def _build_golden_set() -> list[dict[str, Any]]:
    generated = generate_icd10_breadth() + generate_drug_breadth()
    combined = CURATED_SET + generated

    # Sanity check di waktu IMPORT (bukan cuma di test): kalau ada ID yang
    # bentrok antara tier curated & generated, gagal SEKARANG dengan pesan
    # jelas -- bukan diam-diam menimpa salah satu entri tanpa disadari.
    ids = [item["id"] for item in combined]
    duplicates = {x for x in ids if ids.count(x) > 1}
    if duplicates:
        raise ValueError(f"ID golden set bentrok antar tier: {duplicates}")

    return combined


GOLDEN_SET = _build_golden_set()


def summary():
    from collections import Counter

    counts = Counter(item["category"] for item in GOLDEN_SET)
    curated_ids = {item["id"] for item in CURATED_SET}
    print(f"Total golden set: {len(GOLDEN_SET)} entri")
    print(f"  - curated (tulis tangan): {len(curated_ids)}")
    print(f"  - generated (otomatis)  : {len(GOLDEN_SET) - len(curated_ids)}")
    print("Distribusi kategori:")
    for cat, n in counts.items():
        print(f"  - {cat}: {n}")


if __name__ == "__main__":
    summary()
