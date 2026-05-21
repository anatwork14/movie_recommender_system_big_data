"""
AI Agent Orchestration Layer
==============================
WHY THIS EXISTS:
    A plain search bar forces the user into keyword mode.  The Agent enables
    *conversational* queries like:
        "Find dark action movies about time travel, similar to Dark, that I
         haven't seen yet – by a European director."

    The Agent is responsible for:
      1. Intent classification  : Detect which engines to activate.
      2. Parallel tool dispatch : Run CF + TF-IDF + RAG concurrently.
      3. RRF fusion             : Merge ranked candidates.
      4. Contextual explanation : Produce a natural-language justification.

    We do NOT require an external LLM API key for the intent-classification
    step – a fast rule-based parser handles the common cases.  An optional
    OpenAI / Gemini key enables the richer explanation step.

TOOL CALLING DESIGN:
    Tool | Trigger condition                          | Returns
    ─────┼────────────────────────────────────────────┼──────────────────
    CF   | User ID present in session or query        | CFResult list
    TF-IDF | Entity keywords (director, actor, genre) | TFIDFResult list
    RAG  | Abstract / mood / thematic description     | ScoredPoint list

    The Agent always runs all available tools in parallel and merges via RRF,
    giving CF results higher weight when a user context is detected.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..engines.hybrid_fusion import CandidateMovie, HybridFusionEngine

logger = logging.getLogger(__name__)


async def _empty_results():
    return []


# ── Optional LLM integration (graceful fallback if no key) ────────────────────
try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Intent classifier (rule-based)
# ─────────────────────────────────────────────────────────────────────────────

_USER_ID_RE = re.compile(r'\b(?:user[_ ]?id|user)[_ ]?[:#=]?\s*(\d+)', re.I)
_SEMANTIC_TRIGGERS = re.compile(
    r'\b(mood|vibe|feel|similar to|like the movie|atmosphere|dark|theme|emotion'
    r'|heartwarming|tense|suspense|twist|philosophical|abstract|melancholy)\b',
    re.I,
)
_ENTITY_TRIGGERS = re.compile(
    r'\b(direct(?:or|ed by)|actor|cast|starring|featuring|by\s+[A-Z])\b',
    re.I,
)


def _classify_intent(query: str, user_id: Optional[int]) -> Dict[str, bool]:
    """
    Lightweight rule-based intent detection.

    Returns:
        Dict with keys 'use_cf', 'use_tfidf', 'use_rag'.
    """
    use_cf = user_id is not None
    use_tfidf = bool(_ENTITY_TRIGGERS.search(query)) or len(query.split()) <= 4
    use_rag = bool(_SEMANTIC_TRIGGERS.search(query)) or len(query.split()) > 5

    # Always ensure at least TF-IDF + RAG run
    if not use_tfidf and not use_rag:
        use_tfidf = True
        use_rag = True

    return {"use_cf": use_cf, "use_tfidf": use_tfidf, "use_rag": use_rag}


# ─────────────────────────────────────────────────────────────────────────────
# Main Agent class
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationAgent:
    """
    Central AI Agent that orchestrates the three RecSys engines.

    Args:
        cf_engine     : Initialised CollaborativeFilteringEngine instance.
        tfidf_engine  : Initialised TFIDFEngine instance.
        vector_db     : Initialised MovieVectorDB instance.
        fusion_engine : HybridFusionEngine instance (default: new instance).
        openai_key    : Optional OpenAI API key for LLM explanation step.
        llm_model     : OpenAI model to use for explanations.
    """

    def __init__(
        self,
        *,
        cf_engine,
        tfidf_engine,
        vector_db,
        fusion_engine: Optional[HybridFusionEngine] = None,
        openai_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.cf_engine = cf_engine
        self.tfidf_engine = tfidf_engine
        self.vector_db = vector_db
        self.fusion = fusion_engine or HybridFusionEngine()

        self._openai_client = None
        self._llm_model = llm_model
        if openai_key and _OPENAI_AVAILABLE:
            self._openai_client = AsyncOpenAI(api_key=openai_key)
            logger.info("[Agent] OpenAI LLM explanation enabled (model=%s).", llm_model)

    # ── Public async API ──────────────────────────────────────────────────────

    async def recommend(
        self,
        *,
        query: str,
        user_id: Optional[int] = None,
        top_k: int = 10,
        candidate_pool: int = 100,
    ) -> Dict[str, Any]:
        """
        Full agentic recommendation pipeline.

        Steps:
            1. Classify intent from query + user context.
            2. Launch CF / TF-IDF / RAG tools concurrently.
            3. Fuse candidates with RRF.
            4. Optionally generate LLM explanation.

        Args:
            query          : Natural-language query from the user.
            user_id        : Optional user ID for CF personalisation.
            top_k          : Final number of results to return.
            candidate_pool : Max candidates from each engine.

        Returns:
            {
              "recommendation_movies": [...],
              "explanation": "...",
              "intent": {...},
              "provenance": {...},
            }
        """
        intent = _classify_intent(query, user_id)
        logger.info(
            "[Agent] query=%r  user_id=%s  intent=%s",
            query, user_id, intent,
        )

        # ── Launch tools in parallel ─────────────────────────────────────────
        cf_task = asyncio.ensure_future(
            self._tool_cf(user_id, candidate_pool)
        ) if intent["use_cf"] else asyncio.ensure_future(_empty_results())

        tfidf_task = asyncio.ensure_future(
            self._tool_tfidf(query, candidate_pool)
        ) if intent["use_tfidf"] else asyncio.ensure_future(_empty_results())

        rag_task = asyncio.ensure_future(
            self._tool_rag(query, candidate_pool)
        ) if intent["use_rag"] else asyncio.ensure_future(_empty_results())

        cf_results, tfidf_results, rag_results = await asyncio.gather(
            cf_task, tfidf_task, rag_task
        )

        # ── RRF Fusion ───────────────────────────────────────────────────────
        candidates = self.fusion.fuse(
            cf_results=cf_results,
            tfidf_results=tfidf_results,
            rag_results=rag_results,
            movie_meta_fn=self.tfidf_engine.get_by_id if self.tfidf_engine.is_ready else None,
            top_k=top_k,
        )

        # ── LLM Explanation (optional) ───────────────────────────────────────
        explanation = await self._generate_explanation(query, candidates, user_id)

        return {
            "recommendation_movies": [c.to_dict() for c in candidates],
            "explanation": explanation,
            "intent": intent,
            "provenance": {
                "cf_count": len(cf_results),
                "tfidf_count": len(tfidf_results),
                "rag_count": len(rag_results),
                "merged_count": len(candidates),
            },
        }

    async def search(
        self,
        *,
        query: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Hybrid semantic + lexical search (no user context / CF).

        Returns results in the same shape as `recommend` for frontend consistency.
        """
        tfidf_results, rag_results = await asyncio.gather(
            self._tool_tfidf(query, limit),
            self._tool_rag(query, limit),
        )
        candidates = self.fusion.fuse(
            tfidf_results=tfidf_results,
            rag_results=rag_results,
            movie_meta_fn=self.tfidf_engine.get_by_id if self.tfidf_engine.is_ready else None,
            top_k=limit,
        )
        return {
            "results": [c.to_dict() for c in candidates],
            "query": query,
            "count": len(candidates),
        }

    # ── Tool implementations ──────────────────────────────────────────────────

    async def _tool_cf(self, user_id: Optional[int], limit: int):
        """Tool_CF: Personalised recommendations via dot-product on ALS latent factors."""
        if user_id is None or not self.cf_engine.is_ready:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.cf_engine.recommend, user_id, limit
        )

    async def _tool_tfidf(self, query: str, limit: int):
        """Tool_TFIDF: Lexical entity search via TF-IDF cosine similarity."""
        if not self.tfidf_engine.is_ready:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.tfidf_engine.search, query, limit
        )

    async def _tool_rag(self, query: str, limit: int):
        """Tool_RAG: Dense semantic search via HNSW ANN in Qdrant vector DB."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self.vector_db.search, query, limit
            )
        except Exception as exc:
            logger.warning("[Agent] RAG tool failed: %s", exc)
            return []

    # ── LLM Explanation ───────────────────────────────────────────────────────

    async def _generate_explanation(
        self,
        query: str,
        candidates: List[CandidateMovie],
        user_id: Optional[int],
    ) -> str:
        """
        Generate a natural-language explanation for the top recommendations.
        Falls back to a template-based explanation if no LLM is configured.
        """
        if not candidates:
            return "No matching movies found. Try broadening your query."

        titles = [c.title for c in candidates[:5]]

        if self._openai_client:
            context_block = "\n".join(
                f"- {c.title} ({c.year}): {c.description[:120]}…  [sources: {', '.join(c.sources)}]"
                for c in candidates[:5]
            )
            user_ctx = f"User ID {user_id}" if user_id else "a guest user"
            prompt = (
                f"You are an intelligent movie recommendation assistant.\n"
                f"User query: \"{query}\"\n"
                f"User context: {user_ctx}\n\n"
                f"Top recommended movies:\n{context_block}\n\n"
                f"Write a 2-3 sentence explanation of why these movies are recommended, "
                f"referencing the user's query naturally. Be enthusiastic and specific."
            )
            try:
                response = await self._openai_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("[Agent] LLM explanation failed: %s", exc)

        # Template fallback
        source_desc = []
        for c in candidates[:3]:
            if "CF" in c.sources:
                source_desc.append("your viewing history")
            if "TF-IDF" in c.sources:
                source_desc.append("keyword match")
            if "RAG" in c.sources:
                source_desc.append("semantic similarity")

        unique_sources = list(dict.fromkeys(source_desc))
        source_text = " and ".join(unique_sources[:2]) if unique_sources else "content analysis"
        return (
            f"Based on {source_text}, here are the top picks: "
            + ", ".join(f"\"{t}\"" for t in titles[:3])
            + (f" and {len(candidates) - 3} more." if len(candidates) > 3 else ".")
        )
