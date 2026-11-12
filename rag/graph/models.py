"""Knowledge graph node and edge models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProvenanceRef(BaseModel):
    source_url: str
    chunk_id: Optional[str] = None
    section_id: Optional[str] = None
    crawl_timestamp: Optional[str] = None
    extraction_method: str = "rule_based"
    confidence: float = 0.8


class EntityNode(BaseModel):
    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    aliases: List[str] = Field(default_factory=list)
    provenance: List[ProvenanceRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationshipEdge(BaseModel):
    edge_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    provenance: List[ProvenanceRef] = Field(default_factory=list)
    confidence: float = 0.8
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphSnapshot(BaseModel):
    schema_version: str = "v1"
    nodes: List[EntityNode] = Field(default_factory=list)
    edges: List[RelationshipEdge] = Field(default_factory=list)
