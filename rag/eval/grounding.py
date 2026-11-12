"""Non-agentic grounded response validation."""

from dataclasses import dataclass
import re
from typing import List, Optional

from rag.config import AppConfig
from rag.graph.resolver import EntityResolver
from rag.retrieve import SearchResult

ROLE_QUERY_PATTERN = re.compile(r"\b(who is|who was|role of|title of)\b (.+)", re.I)


@dataclass
class GroundedAnswer:
    query: str
    answer: str
    abstained: bool
    reason: Optional[str]
    sources: List[SearchResult]
    confidence: float


class GroundingValidator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.resolver = EntityResolver()

    def answer_with_evidence(self, query: str, sources: List[SearchResult]) -> GroundedAnswer:
        if not sources:
            return GroundedAnswer(
                query, "Insufficient evidence to answer.", True,
                "no_retrieval_hits", [], 0.0,
            )
        top = sources[0]
        if top.score < self.config.grounding_min_score:
            return GroundedAnswer(
                query, "Insufficient evidence to answer.", True,
                "score_below_threshold", sources, top.score,
            )
        role_match = ROLE_QUERY_PATTERN.search(query)
        requested_role = self._extract_requested_role(query)
        if role_match and requested_role and self.config.grounding_abstain_on_role_mismatch:
            candidate_roles = self._extract_roles_from_sources(sources)
            if not any(self.resolver.roles_match(requested_role, role) for role in candidate_roles):
                related = any(
                    self.resolver.roles_match(requested_role, role, exact=False)
                    for role in candidate_roles
                )
                if related:
                    return GroundedAnswer(
                        query,
                        f"Found related role evidence ({', '.join(candidate_roles[:3])}) "
                        f"but no exact match for '{requested_role}'.",
                        True, "role_mismatch_related", sources, top.score,
                    )
                return GroundedAnswer(
                    query, "Insufficient evidence for the requested role.",
                    True, "role_mismatch", sources, top.score,
                )
        snippets = [
            f"{source.content[:240].strip()} (source: {source.source_url or source.document_name})"
            for source in sources[:3]
        ]
        return GroundedAnswer(
            query, "Based on retrieved evidence: " + " | ".join(snippets),
            False, None, sources[:3], min(1.0, top.score),
        )

    @staticmethod
    def _extract_requested_role(query: str) -> Optional[str]:
        for role in ("Director", "Executive Director", "Scientist", "Engineer"):
            if role.lower() in query.lower():
                return role
        return None

    @staticmethod
    def _extract_roles_from_sources(sources: List[SearchResult]) -> List[str]:
        pattern = re.compile(r"\b(Executive Director|Director|Scientist|Engineer)\b", re.I)
        return [match.group(0) for source in sources for match in pattern.finditer(source.content)]
