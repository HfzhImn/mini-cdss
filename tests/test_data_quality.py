"""
test_data_quality.py
----------------------
TUJUAN:
Test di file lain menguji LOGIC kode. Test di sini menguji DATA-nya sendiri
-- ini kategori yang sering dilewatkan proyek RAG/ML, padahal data yang
cacat diam-diam (kolom kosong, duplikat ID, kategori typo) bisa merusak
eval framework tanpa ada satu pun exception yang muncul.

Sengaja pakai fixture yang load file asli dari data/eval/golden_set.py dan
data/processed/*.csv -- ini bukan lagi murni "unit test" (karena bergantung
pada file di disk), tapi tetap cepat & tidak butuh API key/network, jadi
masih pantas dijalankan setiap kali lewat `pytest` biasa.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "eval"))
from golden_set import GOLDEN_SET  # noqa: E402

from settings import settings

VALID_CATEGORIES = {"icd10", "drug", "negative"}


class TestGoldenSetIntegrity:
    def test_no_duplicate_ids(self):
        ids = [item["id"] for item in GOLDEN_SET]
        assert len(ids) == len(set(ids)), "Ada ID yang duplikat di golden set"

    def test_all_categories_are_valid(self):
        for item in GOLDEN_SET:
            assert item["category"] in VALID_CATEGORIES, (
                f"{item['id']} punya category tidak dikenal: {item['category']}"
            )

    def test_every_item_has_nonempty_query(self):
        for item in GOLDEN_SET:
            assert item.get("query"), f"{item['id']} punya query kosong"

    def test_icd10_items_have_expected_code(self):
        for item in GOLDEN_SET:
            if item["category"] == "icd10":
                assert "expected_code" in item, f"{item['id']} kategori icd10 tapi tidak ada expected_code"

    def test_drug_items_have_expected_pair(self):
        for item in GOLDEN_SET:
            if item["category"] == "drug":
                assert "expected_pair" in item, f"{item['id']} kategori drug tapi tidak ada expected_pair"
                assert len(item["expected_pair"]) == 2

    @pytest.mark.skipif(
        not settings.icd10_csv.exists(), reason="icd10_subset.csv belum di-generate"
    )
    def test_expected_icd10_codes_actually_exist_in_dataset(self):
        # Test PALING PENTING di file ini: memastikan ground truth di golden
        # set itu SENDIRI valid -- kalau expected_code menunjuk ke kode yang
        # bahkan tidak ada di dataset kita, eval akan selalu gagal untuk
        # entry itu apapun jawaban modelnya, dan itu bug di test kita,
        # bukan di model yang dievaluasi.
        df = pd.read_csv(settings.icd10_csv)
        known_codes = set(df["icd10_code"])
        for item in GOLDEN_SET:
            if item["category"] == "icd10":
                assert item["expected_code"] in known_codes, (
                    f"{item['id']}: expected_code {item['expected_code']!r} "
                    f"tidak ada di icd10_subset.csv -- golden set ini salah, "
                    f"bukan cuma model yang salah kalau eval gagal di sini"
                )

    @pytest.mark.skipif(
        not settings.drug_interactions_csv.exists(),
        reason="drug_interactions.csv belum di-generate",
    )
    def test_expected_drug_pairs_actually_exist_in_dataset(self):
        df = pd.read_csv(settings.drug_interactions_csv)
        known_pairs = set(zip(df["drug_a"], df["drug_b"], strict=True))
        for item in GOLDEN_SET:
            if item["category"] == "drug":
                pair = tuple(item["expected_pair"])
                assert pair in known_pairs, (
                    f"{item['id']}: pasangan obat {pair} tidak ada di "
                    f"drug_interactions.csv"
                )


@pytest.mark.skipif(not settings.icd10_csv.exists(), reason="icd10_subset.csv belum di-generate")
class TestIcd10DatasetIntegrity:
    @classmethod
    @pytest.fixture(scope="class")
    def df(cls):
        return pd.read_csv(settings.icd10_csv)

    def test_no_null_codes_or_descriptions(self, df):
        assert df["icd10_code"].notna().all()
        assert df["description"].notna().all()

    def test_hierarchy_metadata_is_present(self, df):
        assert {"chapter_id", "group_id"} <= set(df.columns)
        assert df["chapter_id"].notna().all()
        assert df["group_id"].notna().all()

    def test_no_duplicate_codes(self, df):
        assert df["icd10_code"].is_unique, "Ada kode ICD-10 yang duplikat di dataset"

    def test_all_codes_match_icd10_format(self, df):
        # Format WHO ICD-10: 1 huruf + 2 digit, opsional titik + 1-4
        # karakter alfanumerik. Penanda '*' dari file WHO sudah dibuang oleh
        # prepare_icd10.py karena bukan bagian dari identifier kode.
        pattern = r"^[A-Z]\d{2}(\.[\dX]{1,4})?$"
        invalid = df[~df["icd10_code"].str.match(pattern)]
        assert invalid.empty, f"Kode dengan format tidak valid: {invalid['icd10_code'].tolist()}"


@pytest.mark.skipif(
    not settings.drug_interactions_csv.exists(), reason="drug_interactions.csv belum di-generate"
)
class TestDrugInteractionDatasetIntegrity:
    @classmethod
    @pytest.fixture(scope="class")
    def df(cls):
        return pd.read_csv(settings.drug_interactions_csv)

    def test_no_null_fields(self, df):
        required = ["drug_a", "drug_b", "severity", "mechanism", "description"]
        for col in required:
            assert df[col].notna().all(), f"Ada nilai kosong di kolom {col}"

    def test_severity_values_are_valid(self, df):
        assert set(df["severity"].unique()) <= {"major", "moderate", "minor"}

    def test_no_self_interaction(self, df):
        # Sanity check: tidak ada baris "obat A berinteraksi dengan obat A"
        same = df[df["drug_a"].str.lower() == df["drug_b"].str.lower()]
        assert same.empty, f"Ada interaksi obat dengan dirinya sendiri: {same.to_dict('records')}"
