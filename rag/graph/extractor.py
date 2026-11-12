"""Rule-based entity and relationship extraction from page chunks."""

from __future__ import annotations

import hashlib
import re
from typing import List, Tuple

from rag.graph.models import EntityNode, ProvenanceRef, RelationshipEdge
from rag.graph.schema import is_valid_entity_type, is_valid_relationship
from rag.models import ChunkModel

LOCATION_PATTERNS = [re.compile(r"\bPune\b", re.I), re.compile(r"\bIndia\b", re.I)]
ROLE_PATTERN = re.compile(r"\b(Director|Executive Director|Scientist|Engineer)\b", re.I)
PRODUCT_PATTERN = re.compile(r"\bPARAM\b[\w\s-]*", re.I)


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower()).replace("c-dac", "cdac")


def _entity_id(name: str, entity_type: str) -> str:
    return hashlib.sha256(f"{entity_type}:{normalize_name(name)}".encode()).hexdigest()[:16]


def _edge_id(source_id: str, relationship: str, target_id: str) -> str:
    return hashlib.sha256(f"{source_id}|{relationship}|{target_id}".encode()).hexdigest()[:16]


def extract_from_chunk(
    chunk: ChunkModel, *, crawl_timestamp: str | None = None
) -> Tuple[List[EntityNode], List[RelationshipEdge]]:
    provenance = [ProvenanceRef(
        source_url=chunk.source_url or "",
        chunk_id=chunk.chunk_id,
        section_id=chunk.metadata.get("section_id"),
        crawl_timestamp=crawl_timestamp,
    )]
    entities: dict[str, EntityNode] = {}
    edges: list[RelationshipEdge] = []

    def add_entity(name: str, entity_type: str, aliases: list[str] | None = None):
        if not is_valid_entity_type(entity_type):
            return None
        entity_id = _entity_id(name, entity_type)
        if entity_id not in entities:
            entities[entity_id] = EntityNode(
                entity_id=entity_id, name=name.strip(), normalized_name=normalize_name(name),
                entity_type=entity_type, aliases=aliases or [], provenance=list(provenance),
            )
        else:
            entities[entity_id].provenance.extend(provenance)
        return entities[entity_id]

    organization = add_entity(
        "C-DAC", "Organization",
        aliases=["CDAC", "Centre for Development of Advanced Computing"],
    )
    page = add_entity(chunk.document_name or chunk.source_url or "page", "Page")
    for pattern in LOCATION_PATTERNS:
        for match in pattern.finditer(chunk.content):
            add_entity(match.group(0), "Location")
    for match in ROLE_PATTERN.finditer(chunk.content):
        add_entity(match.group(0), "Role")
    for match in PRODUCT_PATTERN.finditer(chunk.content):
        add_entity(match.group(0).strip(), "Product")
    if page and organization:
        edges.append(RelationshipEdge(
            edge_id=_edge_id(page.entity_id, "MENTIONS", organization.entity_id),
            source_entity_id=page.entity_id,
            target_entity_id=organization.entity_id,
            relationship_type="MENTIONS",
            provenance=list(provenance),
        ))
    if organization:
        for entity in list(entities.values()):
            if entity.entity_type == "Location" and is_valid_relationship(
                "LOCATED_IN", organization.entity_type, entity.entity_type
            ):
                edges.append(RelationshipEdge(
                    edge_id=_edge_id(organization.entity_id, "LOCATED_IN", entity.entity_id),
                    source_entity_id=organization.entity_id,
                    target_entity_id=entity.entity_id,
                    relationship_type="LOCATED_IN",
                    provenance=list(provenance),
                ))
    return list(entities.values()), edges
