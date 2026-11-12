from rag.config import AppConfig
from rag.eval.grounding import GroundingValidator
from rag.retrieve import SearchResult


def _source(content: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id="c1", document_id="d1", document_name="Leadership",
        source_url="https://cdac.gov.in/leadership", content=content,
        score=score, retrieval_method="bm25", rank=1,
    )


def test_abstains_without_sources():
    answer = GroundingValidator(AppConfig()).answer_with_evidence("Who is the Director?", [])
    assert answer.abstained
    assert answer.reason == "no_retrieval_hits"


def test_role_mismatch_executive_director_vs_director():
    validator = GroundingValidator(AppConfig(grounding_abstain_on_role_mismatch=True))
    answer = validator.answer_with_evidence(
        "Who is the Director?", [_source("The Executive Director leads the organization.")]
    )
    assert answer.abstained
    assert answer.reason in {"role_mismatch_related", "role_mismatch"}


def test_exact_role_match_answers():
    validator = GroundingValidator(AppConfig(grounding_min_score=0.1))
    answer = validator.answer_with_evidence(
        "Who is the Director?", [_source("The Director oversees C-DAC operations.")]
    )
    assert not answer.abstained
    assert "evidence" in answer.answer.lower()
