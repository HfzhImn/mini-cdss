# Mini-CDSS — RAG Assistant untuk ICD-10 & Interaksi Obat

Mini-clone dari 2 modul Clinical Decision Support System: _diagnosis
verification_ (ICD-10 lookup) dan _drug interaction checking_ — RAG
pipeline multi-model dengan hybrid search, eval framework, dan grounding
untuk mencegah hallucination.

**📘 Belajar dari nol?** Buka `PANDUAN.html` di browser — itu kurikulum
bertahap (Modul 0–13) yang menjelaskan setiap konsep SEBELUM menunjukkan
kode, lengkap dengan checkpoint untuk verifikasi paham. Dokumen ini
(`README.md`) cuma referensi cepat kalau kamu sudah paham dan butuh
lihat command dengan cepat.

**⚠️ Disclaimer:** proyek ini memakai metadata **WHO ICD-10 2019** dari folder
`../icd102019enMeta` dan data **DDInter** dari folder `../ddinter`. Keduanya
dipakai untuk demo dan evaluasi teknis, bukan alat bantu keputusan klinis yang
tervalidasi. Dataset DDInter yang tersedia hanya menyediakan pasangan obat dan
level interaksi; pipeline tidak mengarang mekanisme farmakologis.

## Setup cepat

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Punya GPU NVIDIA (cek lewat dxdiag tab Display) dan mau dipakai?
pip install -e ".[dev]"
# Tidak punya GPU / mau hemat disk? Ganti baris di atas dengan 2 baris ini:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install -e ".[dev]"
# (detail trade-off-nya ada di PANDUAN.html Modul 0)

cp .env.example .env            # isi GROQ_API_KEY (gratis di console.groq.com)

python -m scripts.prepare_icd10
python -m scripts.prepare_drug_interactions
python scripts/etl_pipeline.py       # opsional: SQLite relasional untuk inspeksi

python -m src.build_index             # unduh model embedding, sekali saja
python -m src.cli
```

## Struktur proyek

```
mini-cdss/
├── PANDUAN.html           # MULAI DI SINI — kurikulum belajar bertahap
├── data/
│   ├── raw/                # seed/raw lokal bila diperlukan
│   ├── processed/          # hasil scripts/prepare_*.py
│   ├── eval/                # golden_set_curated.py + golden_set.py (generator)
│   ├── chroma_db/          # vector index (dibuat otomatis)
│   └── cdss.db             # hasil ETL relasional
├── scripts/                # persiapan data + demo cost optimization
├── src/                     # kode inti — settings, retrieval, fusion, rag, eval
├── tests/                   # pytest, jalankan: pytest -v
├── pyproject.toml          # dependencies + config ruff/mypy/pytest
└── .env.example
```

## Perintah yang sering dipakai

```bash
# Quality check (jalankan sebelum commit apapun)
ruff check src/ scripts/ tests/ data/
mypy src/ scripts/ tests/ --ignore-missing-imports
pytest -v

# Eval framework
python -m src.eval_runner --provider groq --model llama-3.1-8b-instant
python -m src.compare_results

# Demo cost optimization (Claude — hemat jatah trial $5)
python scripts/demo_prompt_caching.py
python scripts/demo_tool_use.py
```

## Status proyek

Fase 1 (fondasi software engineering) dan sebagian Fase 3 (golden set
2-tier, hybrid search BM25+RRF) sudah selesai. Detail lengkap tiap fase,
keputusan desain, dan bug nyata yang ditemukan sepanjang jalan — semua
ada di `PANDUAN.html`, bukan diduplikasi di sini supaya tidak ada 2
sumber kebenaran yang bisa saling tidak sinkron.
