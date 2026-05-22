"""
Hybrid Fusion Engine  —  Reciprocal Rank Fusion (RRF) + Reranking
===================================================================
WHY THIS EXISTS:
    CF, TF-IDF, and RAG produce scores in completely different spaces:
      - CF: dot-product of latent factors  ∈ ℝ (unbounded)
      - TF-IDF: cosine similarity over sparse vectors  ∈ [0, 1]
      - RAG: cosine similarity over dense embeddings  ∈ [-1, 1]

    Direct weighted-sum fusion would require per-corpus normalisation that
    varies with query distribution.  RRF solves this by converting scores
    into rank positions (ordinals) before fusion – rank 1 always means
    "best in this engine", making the combination distribution-free.

    Optimal k = 60 is from Cormack et al. (2009).  Setting k higher
    reduces the penalty difference between rank 1 and rank 100; setting
    it lower amplifies the top-1 winner effect.

FORMULA:
    RRF_Score(m) = Σ_{R ∈ {CF, TF-IDF, RAG}}  1 / (k + rank_R(m))

    where rank_R(m) is the 1-based position of movie m in ranked list R.
    If m does not appear in R, the contribution from R is 0.

REFERENCE:
    Cormack, Clarke, Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods."  ACM SIGIR 2009.
    https://dl.acm.org/doi/10.1145/1571941.1572114
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# RRF smoothing constant (Cormack et al., 2009 optimal value)
_RRF_K = 60


@dataclass
class CandidateMovie:
    """
    Unified movie representation after merging candidates from all engines.
    Carries the RRF score plus provenance information for explainability.
    """
    movie_id: int
    title: str
    genres: str
    year: str
    description: str
    poster: str
    rrf_score: float = 0.0
    cf_rank: Optional[int] = None        # 1-based rank from CF engine, or None
    tfidf_rank: Optional[int] = None     # 1-based rank from TF-IDF engine, or None
    rag_rank: Optional[int] = None       # 1-based rank from RAG engine, or None
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.movie_id,
            "title": self.title,
            "genres": self.genres,
            "year": self.year,
            "description": self.description,
            "poster": self.poster,
            "rrf_score": round(self.rrf_score, 6),
            "cf_rank": self.cf_rank,
            "tfidf_rank": self.tfidf_rank,
            "rag_rank": self.rag_rank,
            "sources": self.sources,
        }


class HybridFusionEngine:
    """
    Combines ranked candidate lists from CF, TF-IDF, and RAG using RRF.

    Pipeline:
        1. Receive up to 3 ranked lists (each up to 100 items).
        2. Build a unified {movie_id: CandidateMovie} registry.
        3. For each list, accumulate RRF scores via 1/(k + rank).
        4. Sort merged registry by RRF score descending.
        5. Return top-K final candidates.

    Usage:
        fusion = HybridFusionEngine()
        results = fusion.fuse(
            cf_results=[CFResult(movie_id=..., score=..., rank=...), ...],
            tfidf_results=[TFIDFResult(...)],
            rag_results=[QdrantScoredPoint(...)],
            movie_meta_fn=tfidf_engine.get_by_id,
            top_k=10,
        )
    """

    def fuse(
        self,
        *,
        cf_results: List[Any] = None,
        tfidf_results: List[Any] = None,
        rag_results: List[Any] = None,
        movie_meta_fn=None,
        top_k: int = 10,
        rrf_k: int = _RRF_K,
    ) -> List[CandidateMovie]:
        """
        Execute RRF fusion across provided candidate lists.

        Args:
            cf_results      : List of CFResult objects (already ranked 1..N).
            tfidf_results   : List of TFIDFResult objects.
            rag_results     : List of Qdrant ScoredPoint objects.
            movie_meta_fn   : Callable(movie_id) → TFIDFResult | None.
                              Used to enrich CF/RAG hits with display metadata.
            top_k           : Final number of recommendations to return.
            rrf_k           : RRF smoothing constant (default 60).

        Returns:
            Top-K CandidateMovie list sorted by RRF score (highest first).
        """
        registry: Dict[int, CandidateMovie] = {}

        # ── 1. Process CF results ─────────────────────────────────────────────
        for hit in (cf_results or []):
            mid = int(hit.movie_id)
            rank = int(hit.rank)
            candidate = self._get_or_create(registry, mid, movie_meta_fn)
            candidate.rrf_score += 1.0 / (rrf_k + rank)
            candidate.cf_rank = rank
            if "CF" not in candidate.sources:
                candidate.sources.append("CF")

        # ── 2. Process TF-IDF results ─────────────────────────────────────────
        for rank_0, hit in enumerate(tfidf_results or []):
            rank = rank_0 + 1
            mid = int(hit.movie_id)
            if mid not in registry:
                registry[mid] = CandidateMovie(
                    movie_id=mid,
                    title=hit.title,
                    genres=hit.genres,
                    year=hit.year,
                    description=hit.description,
                    poster=hit.poster,
                )
            candidate = registry[mid]
            candidate.rrf_score += 1.0 / (rrf_k + rank)
            candidate.tfidf_rank = rank
            if "TF-IDF" not in candidate.sources:
                candidate.sources.append("TF-IDF")

        # ── 3. Process RAG (Qdrant) results ───────────────────────────────────
        for rank_0, point in enumerate(rag_results or []):
            rank = rank_0 + 1
            payload = point.payload or {}
            mid = int(payload.get("movie_ref", 0))
            if mid == 0:
                continue
            if mid not in registry:
                registry[mid] = CandidateMovie(
                    movie_id=mid,
                    title=str(payload.get("title", "Unknown")),
                    genres=str(payload.get("genres", "")),
                    year=str(payload.get("year", "Unknown")),
                    description=str(payload.get("description", "")),
                    poster=str(payload.get("poster_url", payload.get("poster", ""))),
                )
            candidate = registry[mid]
            candidate.rrf_score += 1.0 / (rrf_k + rank)
            candidate.rag_rank = rank
            if "RAG" not in candidate.sources:
                candidate.sources.append("RAG")

        # ── 4. Sort and return top-K ──────────────────────────────────────────
        sorted_candidates = sorted(
            registry.values(),
            key=lambda c: c.rrf_score,
            reverse=True,
        )[:top_k]

        logger.debug(
            "[RRF] Fused %d CF + %d TFIDF + %d RAG → %d merged → top %d",
            len(cf_results or []),
            len(tfidf_results or []),
            len(rag_results or []),
            len(registry),
            len(sorted_candidates),
        )
        return sorted_candidates

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_or_create(
        registry: Dict[int, CandidateMovie],
        movie_id: int,
        meta_fn,
    ) -> CandidateMovie:
        """
        Fetch or create a CandidateMovie, enriching metadata via meta_fn if needed.
        """
        if movie_id in registry:
            return registry[movie_id]

        meta = meta_fn(movie_id) if meta_fn else None
        if meta:
            candidate = CandidateMovie(
                movie_id=movie_id,
                title=meta.title,
                genres=meta.genres,
                year=meta.year,
                description=meta.description,
                poster=meta.poster,
            )
        else:
            candidate = CandidateMovie(
                movie_id=movie_id,
                title=f"Movie {movie_id}",
                genres="",
                year="Unknown",
                description="",
                poster="",
            )

        registry[movie_id] = candidate
        return candidate
