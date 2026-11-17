"""
golden_set_curated.py
-----------------------
TUJUAN:
Ini TIER PERTAMA dari 2 tier golden set (lihat golden_set.py untuk tier
kedua & penggabungannya). Semua entri di sini ditulis TANGAN dan
DIVERIFIKASI MANUAL satu-per-satu terhadap data asli di icd10_subset.csv
/ drug_interactions.csv.

KENAPA DIPISAH dari tier "generated" (breadth):
  - Tier ini fokus ke KESULITAN & VARIASI -- paraphrase natural, campuran
    ID/EN, istilah medis presisi vs nama umum, kasus negatif yang butuh
    penalaran domain (bukan sekadar "kode acak yang tidak ada").
  - Tier "generated" (golden_set.py) fokus ke SKALA -- banyak variasi
    otomatis dari seluruh dataset, dijamin benar by construction, tapi
    kurang natural dibanding tulisan manusia.
  - Memisahkan keduanya membuat kualitas eval bisa dievaluasi per-tier:
    kalau model gagal di tier curated tapi lolos di tier generated, itu
    sinyal model kesulitan di kasus yang butuh nuansa, bukan sekadar
    hafalan pola sederhana.

3 KATEGORI query, sengaja dibuat berimbang:
  1. icd10       -- lookup kode diagnosis, campur ID & EN, campur tingkat
                     kesulitan (nama umum vs istilah medis presisi)
  2. drug        -- cek interaksi obat, harus sebut severity dengan benar
  3. negative    -- pertanyaan yang JAWABANNYA TIDAK ADA di dataset kita.
                     Model yang BENAR akan bilang "tidak ditemukan".
                     Model yang HALUSINASI akan mengarang jawaban.
"""

from typing import Any

CURATED_SET: list[dict[str, Any]] = [
    # ============ ICD-10 lookup (positive cases) ============
    {"id": "icd_01", "category": "icd10", "query": "kode ICD-10 untuk kolera tidak spesifik",
     "expected_code": "A00.9"},
    {"id": "icd_02", "category": "icd10", "query": "what is the ICD-10 code for common cold",
     "expected_code": "J00"},
    {"id": "icd_03", "category": "icd10", "query": "kode ICD-10 untuk diabetes tipe 2 tanpa komplikasi",
     "expected_code": "E11.9"},
    {"id": "icd_04", "category": "icd10", "query": "kode diagnosis untuk hipertensi esensial primer",
     "expected_code": "I10"},
    {"id": "icd_05", "category": "icd10", "query": "ICD-10 code for GERD without esophagitis",
     "expected_code": "K21.9"},
    {"id": "icd_06", "category": "icd10", "query": "kode ICD-10 untuk tuberkulosis paru",
     "expected_code": "A15.0"},
    {"id": "icd_07", "category": "icd10", "query": "kode untuk faringitis akut tidak spesifik",
     "expected_code": "J02.9"},
    {"id": "icd_08", "category": "icd10", "query": "kode ICD-10 untuk diabetes tipe 1 tanpa komplikasi",
     "expected_code": "E10.9"},
    {"id": "icd_09", "category": "icd10", "query": "what is the code for streptococcal pharyngitis",
     "expected_code": "J02.0"},
    {"id": "icd_10", "category": "icd10", "query": "kode ICD-10 untuk shigellosis akibat Shigella sonnei",
     "expected_code": "A03.3"},
    {"id": "icd_11", "category": "icd10", "query": "kode diagnosis untuk radang tenggorokan croup",
     "expected_code": "J05.0"},
    {"id": "icd_12", "category": "icd10", "query": "kode WHO ICD-10 untuk gastroenteritis infeksius",
     "expected_code": "A09.0"},
    {"id": "icd_13", "category": "icd10", "query": "WHO ICD-10 code for oesophagitis",
     "expected_code": "K20"},
    {"id": "icd_14", "category": "icd10", "query": "kode untuk uretritis non-spesifik",
     "expected_code": "N34.1"},
    {"id": "icd_15", "category": "icd10", "query": "kode ICD-10 demam paratifoid tipe A",
     "expected_code": "A01.1"},

    # ============ Drug interaction (positive cases) ============
    {"id": "drug_01", "category": "drug", "query": "interaksi antara naltrexone dan abacavir",
     "expected_pair": ("Naltrexone", "Abacavir"), "expected_severity": "moderate"},
    {"id": "drug_02", "category": "drug", "query": "interaksi aluminum hydroxide dengan dolutegravir",
     "expected_pair": ("Aluminum hydroxide", "Dolutegravir"), "expected_severity": "major"},
    {"id": "drug_03", "category": "drug", "query": "is it safe to combine simvastatin with clarithromycin?",
     "expected_pair": ("Simvastatin", "Clarithromycin"), "expected_severity": "major"},
    {"id": "drug_04", "category": "drug", "query": "interaksi acarbose dengan digoxin",
     "expected_pair": ("Acarbose", "Digoxin"), "expected_severity": "moderate"},
    {"id": "drug_05", "category": "drug", "query": "interaksi doxycycline dengan furosemide",
     "expected_pair": ("Doxycycline", "Furosemide"), "expected_severity": "minor"},
    {"id": "drug_06", "category": "drug", "query": "interaksi warfarin dengan ranitidine",
     "expected_pair": ("Warfarin", "Ranitidine"), "expected_severity": "moderate"},
    {"id": "drug_07", "category": "drug", "query": "interaksi digoxin dengan epinephrine",
     "expected_pair": ("Digoxin", "Epinephrine"), "expected_severity": "moderate"},
    {"id": "drug_08", "category": "drug", "query": "interaction between furosemide and alogliptin",
     "expected_pair": ("Furosemide", "Alogliptin"), "expected_severity": "moderate"},

    # ============ Negative / out-of-scope (hallucination probe) ============
    # PENTING: query di bawah ini SENGAJA menanyakan hal yang TIDAK ADA
    # di dataset kita. Jawaban benar = model bilang "tidak ditemukan".
    {"id": "neg_01", "category": "negative", "query": "berapa dosis maksimal paracetamol per hari?",
     "note": "Pertanyaan dosis, bukan lookup ICD-10 atau pencarian pasangan DDInter."},
    {"id": "neg_02", "category": "negative", "query": "what treatment should this patient receive for pneumonia?",
     "note": "Pertanyaan terapi pasien, bukan lookup kode ICD-10."},
    {"id": "neg_03", "category": "negative", "query": "apakah ibuprofen aman dikombinasi dengan paracetamol?",
     "note": "Meminta rekomendasi keselamatan klinis, bukan pasangan yang terjamin ada di corpus DDInter."},
    {"id": "neg_04", "category": "negative", "query": "berapa lama antibiotik harus diminum?",
     "note": "Pertanyaan durasi terapi, di luar scope corpus."},
    {"id": "neg_05", "category": "negative", "query": "interaksi antara amoxicillin dan ibuprofen",
     "note": "Pasangan ini harus ditolak bila tidak ada record DDInter yang di-retrieve."},
    {"id": "neg_06", "category": "negative", "query": "apa efek samping obat ini pada pasien hamil?",
     "note": "Membutuhkan data pasien dan keamanan khusus yang tidak tersedia di corpus."},
    {"id": "neg_07", "category": "negative", "query": "what is the best diagnosis for this patient?",
     "note": "Diagnosis pasien tidak dapat ditentukan dari corpus lookup."},
    {"id": "neg_08", "category": "negative", "query": "berapa dosis maksimal ibuprofen untuk anak?",
     "note": "Pertanyaan dosis pediatrik, di luar scope."},
    {"id": "neg_09", "category": "negative", "query": "apakah pasien ini perlu dirawat inap?",
     "note": "Pertanyaan disposition klinis, bukan lookup."},
    {"id": "neg_10", "category": "negative", "query": "buatkan rencana pengobatan lengkap untuk hipertensi",
     "note": "Meminta treatment plan, bukan kode diagnosis."},
    {"id": "neg_11", "category": "negative", "query": "ICD-10 code for a fictional disease called blue fever",
     "note": "Entitas fiktif tidak boleh diberi kode hasil tebakan."},
    {"id": "neg_12", "category": "negative", "query": "apakah saya boleh menghentikan obat saya hari ini?",
     "note": "Membutuhkan penilaian klinis individual."},
    {"id": "neg_13", "category": "negative", "query": "jelaskan prognosis pasien dengan penyakit kronis ini",
     "note": "Prognosis tidak tersedia di kedua dataset."},
]
