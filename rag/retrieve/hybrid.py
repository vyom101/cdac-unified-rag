"""Candidate-pool hybrid fusion with reference-aware reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rag.config import AppConfig, load_config
from rag.extract.reference_extractor import extract_query_references, is_visual_query
from rag.retrieve import HybridSearchResult, SearchResult
from rag.retrieve.bm25 import BM25Retriever
from rag.retrieve.vector import VectorRetriever


@dataclass
class _MergedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    section: Optional[str]
    page_numbers: List[int]
    content: str
    chunk_type: str = "text"
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    figure_refs: List[str] = field(default_factory=list)
    table_refs: List[str] = field(default_factory=list)
    figure_pages: Dict[str, List[int]] = field(default_factory=dict)
    table_pages: Dict[str, List[int]] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def validate_fusion_weights(keyword_weight: float, semantic_weight: float) -> None:
    if keyword_weight < 0 or semantic_weight < 0:
        raise ValueError(
            f"Retrieval weights must be non-negative "
            f"(got KEYWORD_WEIGHT={keyword_weight}, SEMANTIC_WEIGHT={semantic_weight})."
        )
    total = keyword_weight + semantic_weight
    if abs(total - 1.0) > 1e-5:
        raise ValueError(
            f"Hybrid weights KEYWORD_WEIGHT + SEMANTIC_WEIGHT must equal 1.0 "
            f"(got KEYWORD_WEIGHT={keyword_weight} + SEMANTIC_WEIGHT={semantic_weight} = {total})."
        )


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {chunk_id: 1.0 for chunk_id in scores}
    scale = max_score - min_score
    return {chunk_id: (score - min_score) / scale for chunk_id, score in scores.items()}


def compute_weighted_min_max_scores(
    bm25_scores: Dict[str, float],
    vector_scores: Dict[str, float],
    keyword_weight: float,
    semantic_weight: float,
) -> Dict[str, Tuple[float, float, float]]:
    validate_fusion_weights(keyword_weight, semantic_weight)
    normalized_bm25 = min_max_normalize(bm25_scores)
    normalized_vector = min_max_normalize(vector_scores)
    all_chunk_ids = set(normalized_bm25) | set(normalized_vector)
    fused: Dict[str, Tuple[float, float, float]] = {}
    for chunk_id in all_chunk_ids:
        norm_kw = normalized_bm25.get(chunk_id, 0.0)
        norm_sem = normalized_vector.get(chunk_id, 0.0)
        fused[chunk_id] = (
            norm_kw,
            norm_sem,
            (keyword_weight * norm_kw) + (semantic_weight * norm_sem),
        )
    return fused


def compute_rrf_scores(
    bm25_ranked_ids: List[str],
    vector_ranked_ids: List[str],
    rrf_k: int,
) -> Dict[str, float]:
    if rrf_k <= 0:
        raise ValueError(f"RRF_K must be a positive integer (got {rrf_k}).")
    bm25_ranks = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ranked_ids, start=1)}
    vector_ranks = {chunk_id: rank for rank, chunk_id in enumerate(vector_ranked_ids, start=1)}
    fused: Dict[str, float] = {}
    for chunk_id in set(bm25_ranks) | set(vector_ranks):
        score = 0.0
        if chunk_id in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[chunk_id])
        if chunk_id in vector_ranks:
            score += 1.0 / (rrf_k + vector_ranks[chunk_id])
        fused[chunk_id] = score
    return fused


def merge_search_results(
    bm25_results: List[SearchResult],
    vector_results: List[SearchResult],
) -> Dict[str, _MergedChunk]:
    merged: Dict[str, _MergedChunk] = {}

    def _from_result(result: SearchResult) -> _MergedChunk:
        return _MergedChunk(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            document_name=result.document_name,
            section=result.section,
            page_numbers=result.page_numbers,
            content=result.content,
            chunk_type=result.chunk_type,
            figure_refs=result.figure_refs,
            table_refs=result.table_refs,
            figure_pages=result.figure_pages,
            table_pages=result.table_pages,
            extra=result.model_dump(),
        )

    for result in bm25_results:
        if result.chunk_id not in merged:
            merged[result.chunk_id] = _from_result(result)
        merged[result.chunk_id].bm25_score = result.score

    for result in vector_results:
        if result.chunk_id not in merged:
            merged[result.chunk_id] = _from_result(result)
        merged[result.chunk_id].vector_score = result.score

    return merged


def _reference_bonus(
    chunk: _MergedChunk,
    query_figure_refs: List[str],
    query_table_refs: List[str],
    reference_boost: float,
) -> bool:
    if reference_boost <= 0:
        return False
    if query_figure_refs and set(chunk.figure_refs) & set(query_figure_refs):
        return True
    if query_table_refs and set(chunk.table_refs) & set(query_table_refs):
        return True
    return False


def _is_image_boost(chunk: _MergedChunk, boost_images: bool, image_boost: float) -> bool:
    return bool(boost_images and image_boost > 0 and chunk.chunk_type == "image")


def fuse_hybrid_results(
    bm25_results: List[SearchResult],
    vector_results: List[SearchResult],
    *,
    fusion_strategy: str = "min_max",
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7,
    rrf_k: int = 60,
    top_k: int = 5,
    query_figure_refs: Optional[List[str]] = None,
    query_table_refs: Optional[List[str]] = None,
    reference_boost: float = 0.0,
    boost_images: bool = False,
    image_boost: float = 0.0,
    config: Optional[AppConfig] = None,
    query: str = "",
) -> List[HybridSearchResult]:
    if config is not None:
        fusion_strategy = config.fusion_strategy
        keyword_weight = config.keyword_weight
        semantic_weight = config.semantic_weight
        rrf_k = config.rrf_k
        if query and not query_figure_refs and not query_table_refs:
            refs = extract_query_references(query)
            query_figure_refs = refs["figure_refs"]
            query_table_refs = refs["table_refs"]
            reference_boost = config.figure_reference_boost
            boost_images = config.enable_image_query_boost and is_visual_query(query)
            image_boost = config.image_query_boost

    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer (got {top_k}).")

    strategy = fusion_strategy.lower()
    if strategy not in {"min_max", "rrf"}:
        raise ValueError(
            f"FUSION_STRATEGY must be one of {{'min_max', 'rrf'}} (got '{fusion_strategy}')."
        )
    if strategy == "min_max":
        validate_fusion_weights(keyword_weight, semantic_weight)

    query_figure_refs = query_figure_refs or []
    query_table_refs = query_table_refs or []
    merged = merge_search_results(bm25_results, vector_results)
    hybrid_results: List[HybridSearchResult] = []

    if strategy == "min_max":
        bm25_scores = {
            chunk_id: chunk.bm25_score
            for chunk_id, chunk in merged.items()
            if chunk.bm25_score is not None
        }
        vector_scores = {
            chunk_id: chunk.vector_score
            for chunk_id, chunk in merged.items()
            if chunk.vector_score is not None
        }
        fused_scores = compute_weighted_min_max_scores(
            bm25_scores, vector_scores, keyword_weight, semantic_weight
        )
        for chunk_id, chunk in merged.items():
            norm_kw, norm_sem, final_score = fused_scores[chunk_id]
            boosted = _reference_bonus(
                chunk, query_figure_refs, query_table_refs, reference_boost
            )
            if boosted:
                final_score += reference_boost
            img_boosted = _is_image_boost(chunk, boost_images, image_boost)
            if img_boosted:
                final_score += image_boost
            hybrid_results.append(
                _to_hybrid(
                    chunk,
                    final_score=final_score,
                    norm_kw=norm_kw,
                    norm_sem=norm_sem,
                    strategy="min_max",
                    boosted=boosted,
                    img_boosted=img_boosted,
                )
            )
    else:
        rrf_scores = compute_rrf_scores(
            [result.chunk_id for result in bm25_results],
            [result.chunk_id for result in vector_results],
            rrf_k,
        )
        for chunk_id, chunk in merged.items():
            final_score = rrf_scores[chunk_id]
            boosted = _reference_bonus(
                chunk, query_figure_refs, query_table_refs, reference_boost
            )
            if boosted:
                final_score += reference_boost
            img_boosted = _is_image_boost(chunk, boost_images, image_boost)
            if img_boosted:
                final_score += image_boost
            hybrid_results.append(
                _to_hybrid(
                    chunk,
                    final_score=final_score,
                    norm_kw=0.0,
                    norm_sem=0.0,
                    strategy="rrf",
                    boosted=boosted,
                    img_boosted=img_boosted,
                )
            )

    hybrid_results.sort(key=lambda item: (-item.final_score, item.chunk_id))
    return hybrid_results[:top_k]


def _to_hybrid(
    chunk: _MergedChunk,
    *,
    final_score: float,
    norm_kw: float,
    norm_sem: float,
    strategy: str,
    boosted: bool,
    img_boosted: bool,
) -> HybridSearchResult:
    data = dict(chunk.extra)
    data.update(
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "section": chunk.section,
            "page_numbers": chunk.page_numbers,
            "content": chunk.content,
            "chunk_type": chunk.chunk_type,
            "figure_refs": chunk.figure_refs,
            "table_refs": chunk.table_refs,
            "figure_pages": chunk.figure_pages,
            "table_pages": chunk.table_pages,
            "reference_boosted": boosted,
            "image_boosted": img_boosted,
            "bm25_score": chunk.bm25_score,
            "vector_score": chunk.vector_score,
            "bm25_score_normalized": norm_kw,
            "vector_score_normalized": norm_sem,
            "normalized_keyword_score": norm_kw,
            "normalized_semantic_score": norm_sem,
            "final_score": final_score,
            "score": final_score,
            "retrieval_method": "hybrid",
            "fusion_strategy": strategy,
        }
    )
    return HybridSearchResult(**data)


class HybridRetriever:
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25: Optional[BM25Retriever] = None,
        vector: Optional[VectorRetriever] = None,
        indexer=None,
        embedder=None,
        client=None,
    ):
        self.config = config or load_config()
        self.bm25_retriever = bm25_retriever or bm25 or BM25Retriever(
            self.config, indexer=indexer, client=client
        )
        self.vector_retriever = vector_retriever or vector or VectorRetriever(
            self.config, indexer=indexer, embedder=embedder, client=client
        )
        self.bm25 = self.bm25_retriever
        self.vector = self.vector_retriever

    def _validate_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query)}")
        if not query or query.strip() == "":
            raise ValueError("Query text cannot be empty or whitespace-only.")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        num_candidates: Optional[int] = None,
        keyword_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        fusion_strategy: Optional[str] = None,
        rrf_k: Optional[int] = None,
        **kwargs,
    ) -> List[HybridSearchResult]:
        self._validate_query(query)
        k = top_k if top_k is not None else self.config.top_k
        candidates = (
            num_candidates if num_candidates is not None else self.config.num_candidates
        )
        kw_weight = (
            keyword_weight if keyword_weight is not None else self.config.keyword_weight
        )
        sem_weight = (
            semantic_weight if semantic_weight is not None else self.config.semantic_weight
        )
        strategy = fusion_strategy if fusion_strategy is not None else self.config.fusion_strategy
        rrf_constant = rrf_k if rrf_k is not None else self.config.rrf_k

        if k <= 0:
            raise ValueError(f"top_k must be a positive integer (got {k}).")
        if candidates <= 0:
            raise ValueError(
                f"num_candidates must be a positive integer (got {candidates})."
            )
        if candidates < k:
            raise ValueError(
                f"Validation constraint error: num_candidates ({candidates}) "
                f"must be greater than or equal to top_k ({k})."
            )
        if rrf_constant <= 0:
            raise ValueError(f"RRF_K must be a positive integer (got {rrf_constant}).")

        strategy_normalized = strategy.lower()
        if strategy_normalized not in {"min_max", "rrf"}:
            raise ValueError(
                f"FUSION_STRATEGY must be one of {{'min_max', 'rrf'}} (got '{strategy}')."
            )
        if strategy_normalized == "min_max":
            validate_fusion_weights(kw_weight, sem_weight)

        pool = max(k, getattr(self.config, "hybrid_candidate_pool", k))
        pool_candidates = max(candidates, pool)
        extra = {key: value for key, value in kwargs.items() if value is not None}

        bm25_results = self.bm25_retriever.search(query, top_k=pool, **extra)
        vector_results = self.vector_retriever.search(
            query, top_k=pool, num_candidates=pool_candidates, **extra
        )

        query_figure_refs: List[str] = []
        query_table_refs: List[str] = []
        reference_boost = 0.0
        if getattr(self.config, "enable_reference_metadata", True):
            reference_boost = getattr(self.config, "figure_reference_boost", 0.0)
            query_refs = extract_query_references(query)
            query_figure_refs = query_refs["figure_refs"]
            query_table_refs = query_refs["table_refs"]

        boost_images = False
        image_boost = 0.0
        if getattr(self.config, "enable_image_query_boost", True):
            image_boost = getattr(self.config, "image_query_boost", 0.0)
            boost_images = image_boost > 0 and is_visual_query(query)

        return fuse_hybrid_results(
            bm25_results,
            vector_results,
            fusion_strategy=strategy_normalized,
            keyword_weight=kw_weight,
            semantic_weight=sem_weight,
            rrf_k=rrf_constant,
            top_k=k,
            query_figure_refs=query_figure_refs,
            query_table_refs=query_table_refs,
            reference_boost=reference_boost,
            boost_images=boost_images,
            image_boost=image_boost,
        )
