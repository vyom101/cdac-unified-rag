"""Pre-generation grounding gates."""

import re
from typing import Optional

from rag.config import AppConfig
from rag.retrieve import SearchResult

ABSTAIN_ANSWER = "The information is not available in the provided context."
_ROLES = ("Executive Director", "Director", "Scientist", "Engineer")


def requested_role(query: str) -> Optional[str]:
    for role in _ROLES:
        if role.lower() in query.lower():
            return role
    return None


def grounding_failure(
    query: str, sources: list[SearchResult], config: AppConfig
) -> Optional[str]:
    if not sources:
        return "no_retrieval_hits"
    score = getattr(sources[0], "final_score", sources[0].score)
    if score < config.grounding_min_score:
        return "score_below_threshold"
    role = requested_role(query)
    if role and config.grounding_abstain_on_role_mismatch:
        found = {
            match.group(0).casefold()
            for source in sources
            for match in re.finditer(
                r"\b(Executive Director|Director|Scientist|Engineer)\b",
                source.content, re.I,
            )
        }
        if role.casefold() not in found:
            return "role_mismatch"
    return None


class GroundingValidator:
    def __init__(self, config: AppConfig):
        self.config = config

    def validate(self, query: str, sources: list[SearchResult]) -> Optional[str]:
        return grounding_failure(query, sources, self.config)
