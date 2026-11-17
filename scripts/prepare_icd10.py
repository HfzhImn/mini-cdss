"""
prepare_icd10.py
-----------------
TUJUAN:
Membentuk CSV RAG dari metadata ICD-10 WHO 2019 yang disimpan di folder
`../icd102019enMeta/`. File sumber memakai delimiter titik koma dan format
hierarkis; baris dengan tipe `T` adalah kode terminal yang dipakai sebagai
dokumen diagnosis. Dataset ini adalah WHO ICD-10 2019, bukan ICD-10-CM.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "icd102019enMeta" / "icd102019syst_codes.txt"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "icd10_subset.csv"


def main():
    df = pd.read_csv(RAW_PATH, sep=";", header=None, dtype=str, keep_default_na=False)
    terminal = df[df[1].eq("T")].copy()
    final = pd.DataFrame({
        # WHO memakai '*' sebagai penanda manifestasi, bukan bagian identifier.
        "icd10_code": terminal[5].str.strip().str.rstrip("*"),
        "description": terminal[8].str.strip(),
        "chapter_id": terminal[3].str.strip(),
        "group_id": terminal[4].str.strip(),
    })
    final = final[(final["icd10_code"] != "") & (final["description"] != "")]
    final = final.drop_duplicates(subset=["icd10_code"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_PATH, index=False)
    print(f"WHO ICD-10 final: {len(final)} kode -> disimpan ke {OUT_PATH}")


if __name__ == "__main__":
    main()
