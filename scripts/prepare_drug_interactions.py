"""
prepare_drug_interactions.py
------------------------------
TUJUAN:
Membaca seluruh file CSV DDInter lokal dari `../ddinter/`. DDInter menyediakan
nama obat dan tingkat interaksi; mekanisme/deskripsi tambahan tidak tersedia
di file ini, sehingga pipeline menyimpan penjelasan singkat yang transparan
dan tidak mengarang mekanisme farmakologis.

Script ini melakukan:
  1. Validasi tidak ada baris kosong/rusak
  2. Mempertahankan ejaan/nama obat dari sumber dan membersihkan baris kosong
      untuk menjaga kecocokan dengan golden set dan metadata DDInter
  3. Bikin representasi teks gabungan per baris untuk di-embed
     (drug_a + drug_b + severity + description jadi satu "dokumen")
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "ddinter"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "drug_interactions.csv"

VALID_SEVERITY = {"major", "moderate", "minor"}


def main():
    frames = [pd.read_csv(path) for path in sorted(RAW_DIR.glob("ddinter_downloads_code_*.csv"))]
    if not frames:
        raise FileNotFoundError(f"Tidak ada CSV DDInter di {RAW_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df = df[["Drug_A", "Drug_B", "Level"]].rename(
        columns={"Drug_A": "drug_a", "Drug_B": "drug_b", "Level": "severity"}
    )
    df = df.dropna().drop_duplicates(subset=["drug_a", "drug_b"])
    df["severity"] = df["severity"].str.strip().str.lower()
    df = df[df["severity"].isin(VALID_SEVERITY)]
    df["mechanism"] = "Tidak disediakan oleh dataset DDInter yang digunakan."
    df["description"] = (
        "DDInter mencatat interaksi antara " + df["drug_a"] + " dan " + df["drug_b"]
        + " dengan tingkat keparahan " + df["severity"] + "."
    )

    # Teks gabungan yang akan di-embed -- ini yang menentukan kualitas retrieval,
    # jadi kita masukkan nama kedua obat + severity supaya query seperti
    # "interaksi warfarin dan aspirin" bisa match dengan baik secara semantik.
    df["search_text"] = (
        df["drug_a"] + " dan " + df["drug_b"]
        + " (tingkat keparahan: " + df["severity"] + "). "
        + df["description"]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Dataset final: {len(df)} interaksi obat -> disimpan ke {OUT_PATH}")
    print(f"Distribusi severity:\n{df['severity'].value_counts()}")


if __name__ == "__main__":
    main()
