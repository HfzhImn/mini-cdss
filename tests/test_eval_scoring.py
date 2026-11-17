"""
test_eval_scoring.py
----------------------
TUJUAN:
Ini formalisasi dari 5 skenario sintetis yang kemarin saya jalankan manual
di terminal untuk validasi eval_runner.py. Bagian ini PALING PENTING untuk
diformalkan jadi test asli -- kalau logic deteksi hallucination-nya sendiri
punya bug, seluruh angka "hallucination rate" yang dilaporkan ke pembaca
README/recruiter jadi tidak bisa dipercaya. Ini uji "siapa yang mengawasi
si pengawas".

Kita pakai `SimpleNamespace` untuk mensimulasikan objek RAGResult tanpa
perlu benar-benar memanggil LLM atau ChromaDB -- test ini murni menguji
LOGIC MATEMATIKA/STRING MATCHING-nya, terpisah dari infrastruktur di sekitarnya.
Ini prinsip penting: unit test seharusnya cepat & tidak butuh API key/network.
"""

from types import SimpleNamespace

from eval_runner import extract_icd_codes, score_drug, score_icd10, score_negative


class TestExtractIcdCodes:
    def test_extracts_code_with_decimal(self):
        assert extract_icd_codes("Kodenya adalah A00.9 untuk kasus ini") == {"A00.9"}

    def test_extracts_code_without_decimal(self):
        assert extract_icd_codes("Kode J06 dipakai di sini") == {"J06"}

    def test_extracts_multiple_codes(self):
        result = extract_icd_codes("Bisa A00.9 atau J06.9 tergantung gejala")
        assert result == {"A00.9", "J06.9"}

    def test_no_false_positive_on_plain_text(self):
        # Teks tanpa pola kode ICD-10 harus menghasilkan set kosong,
        # bukan salah tangkap kata biasa sebagai kode
        assert extract_icd_codes("Tidak ditemukan informasi yang relevan") == set()

    def test_does_not_match_lowercase(self):
        # Kode ICD-10 asli selalu huruf besar -- pastikan regex tidak
        # salah tangkap huruf kecil yang kebetulan mirip pola
        assert extract_icd_codes("kode a00.9 huruf kecil") == set()


class TestScoreIcd10:
    def test_correct_and_grounded_answer(self):
        item = {"expected_code": "A00.9"}
        result = SimpleNamespace(
            answer="Kode ICD-10 untuk kolera tidak spesifik adalah A00.9.",
            sources=[{"icd10_code": "A00.9", "description": "Cholera, unspecified"}],
        )
        correct, hallucinated = score_icd10(item, result)
        assert correct is True
        assert hallucinated is False

    def test_hallucinated_extra_code_not_in_sources(self):
        item = {"expected_code": "A00.9"}
        result = SimpleNamespace(
            answer="Kodenya A00.9, atau bisa juga Z99.9 tergantung kondisi.",
            sources=[{"icd10_code": "A00.9", "description": "Cholera, unspecified"}],
        )
        correct, hallucinated = score_icd10(item, result)
        assert correct is True  # kode yang benar tetap disebut
        assert hallucinated is True  # tapi ada kode tambahan yang tidak grounded

    def test_wrong_code_entirely(self):
        item = {"expected_code": "A00.9"}
        result = SimpleNamespace(
            answer="Kodenya adalah Z99.9.",
            sources=[{"icd10_code": "A00.9", "description": "Cholera, unspecified"}],
        )
        correct, hallucinated = score_icd10(item, result)
        assert correct is False
        assert hallucinated is True  # Z99.9 tidak ada di sources


class TestScoreNegative:
    def test_model_correctly_refuses(self):
        result = SimpleNamespace(
            answer="Tidak ditemukan informasi yang relevan dalam basis data.",
            sources=[],
        )
        correct, hallucinated = score_negative({}, result)
        assert correct is True
        assert hallucinated is False

    def test_model_fabricates_instead_of_refusing(self):
        # Kasus PALING KRITIS untuk domain kesehatan: model tidak jujur
        # bilang "tidak tahu", malah mengarang jawaban yang kelihatan valid
        result = SimpleNamespace(
            answer="Kode ICD-10 untuk fraktur femur adalah S72.001A.",
            sources=[],
        )
        correct, hallucinated = score_negative({}, result)
        assert correct is False
        assert hallucinated is True

    def test_refusal_phrase_case_insensitive(self):
        result = SimpleNamespace(
            answer="TIDAK DITEMUKAN informasi yang relevan.",
            sources=[],
        )
        correct, _ = score_negative({}, result)
        assert correct is True


class TestScoreDrug:
    def test_correct_pair_mentioned(self):
        item = {"expected_pair": ("Warfarin", "Aspirin")}
        result = SimpleNamespace(
            answer="Warfarin dan Aspirin memiliki interaksi major.",
            sources=[{"drug_a": "Warfarin", "drug_b": "Aspirin", "severity": "major"}],
        )
        correct, _ = score_drug(item, result)
        assert correct is True

    def test_pair_not_mentioned_at_all(self):
        item = {"expected_pair": ("Warfarin", "Aspirin")}
        result = SimpleNamespace(
            answer="Tidak ditemukan informasi yang relevan.",
            sources=[],
        )
        correct, _ = score_drug(item, result)
        assert correct is False

    def test_case_insensitive_matching(self):
        item = {"expected_pair": ("Warfarin", "Aspirin")}
        result = SimpleNamespace(
            answer="warfarin dan aspirin berinteraksi.",
            sources=[{"drug_a": "Warfarin", "drug_b": "Aspirin", "severity": "major"}],
        )
        correct, _ = score_drug(item, result)
        assert correct is True
