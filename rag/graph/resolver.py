"""Entity resolution with evidence-backed merge rules."""

from typing import Dict, List

from rag.graph.models import EntityNode, RelationshipEdge


class EntityResolver:
    ALIAS_MAP = {
        "cdac": "C-DAC",
        "centre for development of advanced computing": "C-DAC",
        "c-dac": "C-DAC",
    }
    ROLE_EQUIVALENCE = {"executive director": {"director"}}

    def resolve_entities(self, entities: List[EntityNode]) -> List[EntityNode]:
        merged: dict[str, EntityNode] = {}
        for entity in entities:
            canonical_name = self.ALIAS_MAP.get(entity.normalized_name, entity.name)
            key = f"{entity.entity_type}:{canonical_name.lower()}"
            if key in merged:
                merged[key].provenance.extend(entity.provenance)
                for alias in entity.aliases:
                    if alias not in merged[key].aliases:
                        merged[key].aliases.append(alias)
            else:
                entity.name = canonical_name
                merged[key] = entity
        return list(merged.values())

    def resolve_edges(
        self, edges: List[RelationshipEdge], entity_map: Dict[str, EntityNode]
    ) -> List[RelationshipEdge]:
        seen: set[str] = set()
        resolved = []
        for edge in edges:
            if (
                edge.source_entity_id in entity_map
                and edge.target_entity_id in entity_map
                and edge.edge_id not in seen
            ):
                seen.add(edge.edge_id)
                resolved.append(edge)
        return resolved

    def roles_match(self, query_role: str, candidate_role: str, *, exact: bool = True) -> bool:
        query = query_role.strip().lower()
        candidate = candidate_role.strip().lower()
        return query == candidate or (
            not exact and candidate in self.ROLE_EQUIVALENCE.get(query, set())
        )
