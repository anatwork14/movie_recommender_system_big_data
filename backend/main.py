"""
FastAPI Backend Server — Hybrid Movie Recommender System
=========================================================
Endpoints:
    GET  /api/health                — Liveness probe
    GET  /api/trending              — Top trending movies (TF-IDF catalog)
    GET  /api/movie/{movie_id}      — Single movie metadata
    GET  /api/search?q=&limit=      — Hybrid semantic+lexical search
    GET  /api/feed                  — Live Kafka interaction feed
    GET  /api/click/{movie_id}      — Track click → CF personalisation
    GET  /api/rate/{movie_id}/{r}   — Track rating  → Kafka
    GET  /api/average_rating/{id}   — Average rating from stored events
    POST /api/recommend             — Full Agentic Hybrid recommendation

Architecture:
    Lifespan startup → builds TF-IDF index + loads CF factors + pings Qdrant
    All ML work done in a ThreadPoolExecutor (CPU-bound) to avoid blocking
    the async event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from recsys.engines.tfidf_engine import TFIDFEngine
from recsys.engines.cf_engine import CollaborativeFilteringEngine
from recsys.engines.hybrid_fusion import HybridFusionEngine
from recsys.search.vector_db import MovieVectorDB
from recsys.agent.orchestrator import RecommendationAgent

# ── Kafka (optional, graceful degradation) ────────────────────────────────────
try:
    from kafka_streaming.producer import InteractionProducer
    _KAFKA_OK = True
except Exception:
    _KAFKA_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recsys.api")

# ─────────────────────────────────────────────────────────────────────────────
# Global singletons (populated during lifespan)
# ─────────────────────────────────────────────────────────────────────────────
tfidf_engine  = TFIDFEngine()
cf_engine     = CollaborativeFilteringEngine()
vector_db     = MovieVectorDB(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)
fusion_engine = HybridFusionEngine()
agent: Optional[RecommendationAgent] = None

# In-memory interaction store (replaces Redis for zero-dependency demo)
_interaction_feed: deque = deque(maxlen=200)
_rating_store: Dict[int, List[float]] = defaultdict(list)

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan – startup & shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("═" * 60)
    logger.info(" Starting Hybrid RecSys Backend")
    logger.info("═" * 60)

    # 1. Build TF-IDF index (blocking CPU task → thread)
    data_dir = ROOT / "data"
    movie_csv = data_dir / "process_movie.csv"
    loop = asyncio.get_event_loop()

    logger.info("[Startup] Building TF-IDF index from %s …", movie_csv)
    await loop.run_in_executor(None, tfidf_engine.build, str(movie_csv))

    # 2. Load CF embeddings (if ALS pickle exists)
    cf_pickle = ROOT / "models" / "als_factors.pkl"
    ratings_csv = data_dir / "process_movie_rating.csv"
    if cf_pickle.exists():
        logger.info("[Startup] Loading pre-trained ALS factors from %s …", cf_pickle)
        await loop.run_in_executor(None, cf_engine.load_from_pickle, str(cf_pickle))
    elif ratings_csv.exists():
        logger.info("[Startup] Training lightweight in-process ALS (sample 3%%) …")
        await loop.run_in_executor(
            None,
            lambda: cf_engine.build_from_ratings_csv(
                str(ratings_csv),
                rank=20,
                max_iter=5,
                reg_param=0.1,
                sample_frac=0.03,
            ),
        )
        # Persist for next boot
        import pickle, pathlib
        pathlib.Path(ROOT / "models").mkdir(exist_ok=True)
        with open(str(cf_pickle), "wb") as f:
            pickle.dump({
                "user_factors": cf_engine.user_factors,
                "item_factors": cf_engine.item_factors,
                "user_id_map":  cf_engine.user_id_map,
                "item_id_map":  cf_engine.item_id_map,
            }, f)
        logger.info("[Startup] ALS pickle saved → %s", cf_pickle)
    else:
        logger.warning("[Startup] No ratings CSV found – CF engine disabled.")

    # 3. Probe Qdrant
    try:
        colls = vector_db.client.get_collections()
        names = [c.name for c in colls.collections]
        logger.info("[Startup] Qdrant reachable.  Collections: %s", names)
        if vector_db.collection_name not in names:
            logger.warning(
                "[Startup] Qdrant collection '%s' not found. "
                "Run data_processing_pipeline/indexer.py first.",
                vector_db.collection_name,
            )
    except Exception as exc:
        logger.warning("[Startup] Qdrant unavailable (%s). RAG disabled.", exc)

    # 4. Build Agent
    openai_key = os.getenv("OPENAI_API_KEY", "")
    agent = RecommendationAgent(
        cf_engine=cf_engine,
        tfidf_engine=tfidf_engine,
        vector_db=vector_db,
        fusion_engine=fusion_engine,
        openai_key=openai_key or None,
    )
    logger.info("[Startup] RecommendationAgent ready.  LLM=%s", bool(openai_key))
    logger.info("═" * 60)

    yield  # ← server runs here

    logger.info("[Shutdown] Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hybrid Movie RecSys API",
    description=(
        "4-Layer Hybrid Recommender: CF (ALS/NCF) + Lexical TF-IDF + "
        "Semantic RAG (Qdrant) orchestrated by an AI Agent with RRF fusion."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request models
# ─────────────────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = ""
    user_id: Optional[int] = None
    top_k: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_agent():
    if agent is None:
        raise HTTPException(503, "Backend is still starting up. Try again in a moment.")


def _poster_url(path: str) -> str:
    """Resolve TMDB relative paths to full TMDB image URLs."""
    path = str(path or "")
    if not path or path == "nan":
        return ""
    if path.startswith("http"):
        return path
    if path.startswith("/"):
        return f"https://image.tmdb.org/t/p/w500{path}"
    return path


def _enrich(d: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite poster field to absolute URL for the frontend."""
    d["poster"] = _poster_url(d.get("poster", ""))
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "tfidf_ready": tfidf_engine.is_ready,
        "cf_ready": cf_engine.is_ready,
    }


@app.get("/api/trending")
async def trending(limit: int = Query(20, ge=1, le=100)):
    """Return trending movies from the TF-IDF index (first N in catalog)."""
    _require_agent()
    movies = tfidf_engine.trending(limit)
    return {
        "movies": [
            _enrich({
                "id": m.movie_id,
                "title": m.title,
                "genres": m.genres,
                "year": m.year,
                "description": m.description,
                "poster": m.poster,
            })
            for m in movies
        ]
    }


@app.get("/api/movie/{movie_id}")
async def get_movie(movie_id: int):
    """Fetch single movie metadata."""
    _require_agent()
    meta = tfidf_engine.get_by_id(movie_id)
    if not meta:
        raise HTTPException(404, f"Movie {movie_id} not found.")
    return {
        "movie": _enrich({
            "id": meta.movie_id,
            "title": meta.title,
            "genres": meta.genres,
            "year": meta.year,
            "description": meta.description,
            "poster": meta.poster,
        })
    }


@app.get("/api/search")
async def search(
    q: str = Query("", description="Search query"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Hybrid search: TF-IDF + Qdrant semantic.
    Returns list of movies sorted by RRF score.
    """
    _require_agent()
    if not q.strip():
        return {"results": [], "query": q, "count": 0}

    result = await agent.search(query=q, limit=limit)
    result["results"] = [_enrich(r) for r in result["results"]]
    return result


@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    """
    Full AI Agent agentic recommendation:
        - Detects intent (CF / TF-IDF / RAG)
        - Runs tools in parallel
        - Fuses via RRF
        - Generates natural-language explanation
    """
    _require_agent()
    result = await agent.recommend(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
    )
    result["recommendation_movies"] = [
        _enrich(r) for r in result["recommendation_movies"]
    ]
    return result


@app.get("/api/click/{movie_id}")
async def click(
    movie_id: int,
    user_id: str = Query("1337"),
):
    """
    Record a click interaction → Kafka → Realtime feed.
    Also triggers a CF-based recommendation refresh for this user.
    """
    _require_agent()
    ts = datetime.utcnow().isoformat()
    meta = tfidf_engine.get_by_id(movie_id)
    movie_title = meta.title if meta else str(movie_id)

    event = {
        "type": "Click",
        "user": user_id,
        "movie": movie_id,
        "detail": movie_title,
        "time": ts,
    }
    _interaction_feed.appendleft(event)

    # Produce to Kafka (fire-and-forget)
    if _KAFKA_OK:
        try:
            loop = asyncio.get_event_loop()
            producer = InteractionProducer(
                bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
            )
            await loop.run_in_executor(
                None, lambda: (producer.track_click(user_id, movie_id), producer.flush())
            )
        except Exception as exc:
            logger.debug("[Click] Kafka send failed (non-fatal): %s", exc)

    # Personalised recommendations for this user
    try:
        uid = int(user_id)
    except ValueError:
        uid = None

    result = await agent.recommend(
        query=movie_title,
        user_id=uid,
        top_k=20,
    )
    result["recommendation_movies"] = [
        _enrich(r) for r in result["recommendation_movies"]
    ]
    return result


@app.get("/api/rate/{movie_id}/{rating}")
async def rate(
    movie_id: int,
    rating: float,
    user_id: str = Query("1337"),
):
    """Record a rating interaction → Kafka → In-memory store."""
    _require_agent()
    if not (0.5 <= rating <= 5.0):
        raise HTTPException(400, "Rating must be between 0.5 and 5.0")

    _rating_store[movie_id].append(rating)
    ts = datetime.utcnow().isoformat()
    meta = tfidf_engine.get_by_id(movie_id)
    movie_title = meta.title if meta else str(movie_id)

    event = {
        "type": "Rating",
        "user": user_id,
        "movie": movie_id,
        "detail": f"{movie_title} — {rating}★",
        "time": ts,
    }
    _interaction_feed.appendleft(event)

    if _KAFKA_OK:
        try:
            loop = asyncio.get_event_loop()
            producer = InteractionProducer(
                bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
            )
            await loop.run_in_executor(
                None,
                lambda: (
                    producer.track_rating(user_id, movie_id, rating),
                    producer.flush(),
                ),
            )
        except Exception as exc:
            logger.debug("[Rate] Kafka send failed (non-fatal): %s", exc)

    return {"status": "ok", "movie_id": movie_id, "rating": rating, "user_id": user_id}


@app.get("/api/average_rating/{movie_id}")
async def average_rating(movie_id: int):
    """Return average rating and count for a movie from the in-memory store."""
    ratings = _rating_store.get(movie_id, [])
    if not ratings:
        return {"avg_rating": None, "rating_count": 0}
    return {
        "avg_rating": round(sum(ratings) / len(ratings), 2),
        "rating_count": len(ratings),
    }


@app.get("/api/feed")
async def feed(limit: int = Query(50, ge=1, le=200)):
    """
    Return the recent live interaction events + a personalised recommendation
    list derived from the most recent click event.
    """
    events = list(_interaction_feed)[:limit]
    last_movie_id = None
    last_user_id = None
    for ev in events:
        if ev.get("type") == "Click":
            last_movie_id = ev.get("movie")
            last_user_id = ev.get("user")
            break

    recs = []
    if last_movie_id and agent:
        try:
            meta = tfidf_engine.get_by_id(int(last_movie_id))
            q = meta.title if meta else f"movie {last_movie_id}"
            uid = int(last_user_id) if last_user_id else None
            result = await agent.recommend(query=q, user_id=uid, top_k=10)
            recs = [_enrich(r) for r in result.get("recommendation_movies", [])]
        except Exception as exc:
            logger.debug("[Feed] Rec generation failed: %s", exc)

    return {"feed": events, "recommendation_movies": recs}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        reload=bool(os.getenv("DEV", "")),
        log_level="info",
    )
