"""Versioned KG persistence and lightweight traversal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import networkx as nx

from rag.graph.extractor import extract_from_chunk, normalize_name
from rag.graph.models import EntityNode, KnowledgeGraphSnapshot, RelationshipEdge
from rag.graph.resolver import EntityResolver
from rag.models import ChunkModel
from rag.retrieve import SearchResult


class KnowledgeGraphStore:
    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot = KnowledgeGraphSnapshot()
        self.graph = nx.MultiDiGraph()
        self.resolver = EntityResolver()
        if self.store_path.exists():
            self.load()

    def load(self) -> None:
        self.snapshot = KnowledgeGraphSnapshot.model_validate(
            json.loads(self.store_path.read_text(encoding="utf-8"))
        )
        self._rebuild_graph()

    def save(self) -> None:
        self.store_path.write_text(self.snapshot.model_dump_json(indent=2), encoding="utf-8")

    def ingest_chunks(
        self, chunks: List[ChunkModel], *, crawl_timestamp: str | None = None
    ) -> KnowledgeGraphSnapshot:
        all_entities: list[EntityNode] = []
        all_edges: list[RelationshipEdge] = []
        for chunk in chunks:
            entities, edges = extract_from_chunk(chunk, crawl_timestamp=crawl_timestamp)
            all_entities.extend(entities)
            all_edges.extend(edges)
        merged_entities = self.resolver.resolve_entities(self.snapshot.nodes + all_entities)
        entity_map = {entity.entity_id: entity for entity in merged_entities}
        merged_edges = self.resolver.resolve_edges(self.snapshot.edges + all_edges, entity_map)
        self.snapshot = KnowledgeGraphSnapshot(nodes=merged_entities, edges=merged_edges)
        self._rebuild_graph()
        self.save()
        return self.snapshot

    def _rebuild_graph(self) -> None:
        self.graph.clear()
        for node in self.snapshot.nodes:
            self.graph.add_node(
                node.entity_id, name=node.name, entity_type=node.entity_type,
                normalized_name=node.normalized_name,
            )
        for edge in self.snapshot.edges:
            self.graph.add_edge(
                edge.source_entity_id, edge.target_entity_id, key=edge.edge_id,
                relationship_type=edge.relationship_type, confidence=edge.confidence,
            )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        normalized_query = normalize_name(query)
        tokens = set(normalized_query.split())
        scored: list[tuple[float, EntityNode]] = []
        for node in self.snapshot.nodes:
            haystack = normalize_name(
                node.name + " " + " ".join(node.aliases) + " " + node.entity_type
            )
            overlap = sum(1 for token in tokens if token in haystack)
            if overlap or normalized_query in haystack:
                scored.append((overlap / max(len(tokens), 1), node))
        scored.sort(key=lambda row: (-row[0], row[1].entity_id))
        results = []
        for rank, (score, node) in enumerate(scored[:top_k], start=1):
            provenance = node.provenance[0] if node.provenance else None
            node_provenance = [
                reference.model_dump() for reference in node.provenance
            ]
            edge_provenance = []
            for edge in self.snapshot.edges:
                if node.entity_id not in {
                    edge.source_entity_id, edge.target_entity_id
                }:
                    continue
                edge_provenance.append({
                    "edge_id": edge.edge_id,
                    "source_entity_id": edge.source_entity_id,
                    "target_entity_id": edge.target_entity_id,
                    "relationship_type": edge.relationship_type,
                    "provenance": [
                        reference.model_dump() for reference in edge.provenance
                    ],
                })
            content = f"{node.entity_type}: {node.name}"
            if node.aliases:
                content += f" (aliases: {', '.join(node.aliases)})"
            results.append(SearchResult(
                chunk_id=f"kg:{node.entity_id}",
                document_id=node.entity_id,
                document_name=node.name,
                source_url=provenance.source_url if provenance else None,
                section=node.entity_type,
                content=content,
                score=score,
                retrieval_method="knowledge_graph",
                rank=rank,
                metadata={
                    "entity_type": node.entity_type,
                    "aliases": node.aliases,
                    "node_provenance": node_provenance,
                    "edge_provenance": edge_provenance,
                },
            ))
        return results

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        return next((node for node in self.snapshot.nodes if node.entity_id == entity_id), None)
