from dataclasses import dataclass

from src.rag import answer_icd10_query


@dataclass
class FakeResponse:
    text: str = "jawaban"
    model: str = "test-model"
    provider: str = "test"
    input_tokens: int = 1
    output_tokens: int = 1
    latency_ms: float = 1.0
    estimated_cost_usd: float = 0.0


class FakeRetriever:
    def search_icd10(self, query, chapter_id=None, group_id=None):
        return [
            type(
                "Retrieved",
                (),
                {
                    "text": "A00.0: Cholera",
                    "metadata": {
                        "icd10_code": "A00.0",
                        "description": "Cholera",
                        "chapter_id": "01",
                        "group_id": "A00",
                    },
                    "rerank_score": 1.0,
                },
            )()
        ]


class FakeLLM:
    def generate(self, system_prompt, user_prompt):
        return FakeResponse()


def test_icd10_sources_are_flat_metadata_for_api_clients():
    result = answer_icd10_query(FakeRetriever(), FakeLLM(), "kode kolera")

    source = result.to_api_dict()["sources"][0]
    assert source["icd10_code"] == "A00.0"
    assert source["chapter_id"] == "01"
    assert "metadata" not in source
    assert "text" not in source