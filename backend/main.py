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
from typing import Any, Dict, List, Optional, Set, Tuple

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
from recsys.engines.hybrid_fusion import HybridFusionEngine, CandidateMovie
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
_rating_sum_store: Dict[int, float] = defaultdict(float)
_rating_count_store: Dict[int, int] = defaultdict(int)
_user_rating_store: Dict[Tuple[int, int], float] = {}
_historical_users_loaded: Set[int] = set()


def _load_rating_aggregates(csv_path: str) -> None:
    """Load original MovieLens rating aggregates so detail pages show real averages."""
    import pandas as pd

    cache_path = ROOT / "models" / "rating_aggregates.csv"
    _rating_sum_store.clear()
    _rating_count_store.clear()

    csv_mtime = Path(csv_path).stat().st_mtime
    if cache_path.exists() and cache_path.stat().st_mtime >= csv_mtime:
        logger.info("[Startup] Loading cached rating aggregates from %s …", cache_path)
        df = pd.read_csv(cache_path)
        for row in df.itertuples(index=False):
            movie_id = int(row.movieId)
            _rating_sum_store[movie_id] = float(row.rating_sum)
            _rating_count_store[movie_id] = int(row.rating_count)
        logger.info("[Startup] Cached rating aggregates loaded for %d movies.", len(_rating_count_store))
        return

    logger.info("[Startup] Building rating aggregate cache from %s …", csv_path)
    for chunk in pd.read_csv(csv_path, usecols=["movieId", "rating"], chunksize=1_000_000):
        grouped = chunk.groupby("movieId")["rating"].agg(["sum", "count"])
        for movie_id, row in grouped.iterrows():
            mid = int(movie_id)
            _rating_sum_store[mid] += float(row["sum"])
            _rating_count_store[mid] += int(row["count"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_df = pd.DataFrame({
        "movieId": list(_rating_count_store.keys()),
        "rating_sum": [_rating_sum_store[mid] for mid in _rating_count_store.keys()],
        "rating_count": [_rating_count_store[mid] for mid in _rating_count_store.keys()],
    })
    tmp_path = cache_path.with_suffix(".tmp")
    aggregate_df.to_csv(tmp_path, index=False)
    tmp_path.replace(cache_path)
    logger.info("[Startup] Rating aggregates loaded for %d movies.", len(_rating_count_store))


def _load_historical_user_ratings(user_id: int) -> None:
    """Load one user's original CSV ratings into memory, assuming ratings are grouped by userId."""
    import pandas as pd

    if user_id in _historical_users_loaded:
        return

    ratings_csv = ROOT / "data" / "process_movie_rating.csv"
    if not ratings_csv.exists():
        _historical_users_loaded.add(user_id)
        return

    for chunk in pd.read_csv(
        ratings_csv,
        usecols=["userId", "movieId", "rating"],
        chunksize=1_000_000,
    ):
        if chunk["userId"].max() < user_id:
            continue
        if chunk["userId"].min() > user_id:
            break
        matches = chunk[chunk["userId"] == user_id]
        for row in matches.itertuples(index=False):
            _user_rating_store[(int(row.userId), int(row.movieId))] = float(row.rating)

    _historical_users_loaded.add(user_id)


def _record_rating(user_id: int, movie_id: int, rating: float, previous_rating: Optional[float]) -> None:
    """Update in-memory aggregate state, treating re-rates as updates."""
    if previous_rating is None:
        _rating_sum_store[movie_id] += rating
        _rating_count_store[movie_id] += 1
    else:
        _rating_sum_store[movie_id] += rating - previous_rating
    _user_rating_store[(user_id, movie_id)] = rating

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

    # 2. Load CF embeddings. Prefer the full Spark ALS model when it exists.
    als_user_factors = ROOT / "models" / "als" / "user_factors.parquet"
    als_item_factors = ROOT / "models" / "als" / "item_factors.parquet"
    cf_pickle = ROOT / "models" / "als_factors.pkl"
    ratings_csv = data_dir / "process_movie_rating.csv"
    if als_user_factors.exists() and als_item_factors.exists():
        logger.info("[Startup] Loading full Spark ALS factors from %s …", als_user_factors.parent)
        await loop.run_in_executor(
            None,
            cf_engine.load_from_parquet,
            str(als_user_factors),
            str(als_item_factors),
        )
    elif cf_pickle.exists():
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

    # 3. Load movie-level rating averages from the original ratings data.
    if ratings_csv.exists():
        await loop.run_in_executor(None, _load_rating_aggregates, str(ratings_csv))

    # 4. Probe Qdrant
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

    # 5. Build Agent
    openai_key = os.getenv("OPENAI_API_KEY", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    agent = RecommendationAgent(
        cf_engine=cf_engine,
        tfidf_engine=tfidf_engine,
        vector_db=vector_db,
        fusion_engine=fusion_engine,
        openai_key=openai_key or None,
        openai_base_url=openai_base_url or None,
        llm_model=llm_model,
    )
    logger.info("[Startup] RecommendationAgent ready.  LLM=%s, Model=%s", bool(openai_key), llm_model)
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


class LoginRequest(BaseModel):
    username: str
    password: str


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


_DEMO_USERS: Dict[str, Dict[str, Any]] = {
    "alice": {"password": "alice123", "user_id": 1337, "name": "Alice"},
    "bob": {"password": "bob123", "user_id": 2024, "name": "Bob"},
    "charlie": {"password": "charlie123", "user_id": 7777, "name": "Charlie"},
}


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


@app.get("/api/auth/accounts")
async def auth_accounts():
    return {
        "accounts": [
            {"username": u, "name": v["name"], "user_id": v["user_id"]}
            for u, v in _DEMO_USERS.items()
        ]
    }


@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    username = req.username.strip().lower()
    row = _DEMO_USERS.get(username)
    if not row or row["password"] != req.password:
        raise HTTPException(401, "Invalid username or password")
    return {
        "ok": True,
        "user": {
            "username": username,
            "name": row["name"],
            "user_id": row["user_id"],
        },
    }


@app.post("/api/auth/logout")
async def auth_logout():
    return {"ok": True}


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
    Returns immediately — recommendation refresh runs in the background
    so the frontend never hangs waiting for the ML pipeline.
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

    # Produce to Kafka (fire-and-forget, non-blocking)
    if _KAFKA_OK:
        try:
            loop = asyncio.get_event_loop()
            producer = InteractionProducer(
                bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
            )
            await loop.run_in_executor(
                None, lambda: (producer.track_click(user_id, movie_id), producer.flush(2))
            )
        except Exception as exc:
            logger.debug("[Click] Kafka send failed (non-fatal): %s", exc)

    # Run personalised recommendation refresh in the background —
    # do NOT await it so this endpoint returns instantly.
    try:
        uid = int(user_id)
    except ValueError:
        uid = None

    async def _refresh_recs():
        try:
            result = await agent.recommend(
                query=movie_title,
                user_id=uid,
                top_k=20,
            )
            enriched = [_enrich(r) for r in result.get("recommendation_movies", [])]
            # Stash in the feed so /api/feed can serve them on the next poll
            if enriched:
                _interaction_feed.appendleft({
                    "type": "_rec_cache",
                    "user": user_id,
                    "movies": enriched,
                    "time": ts,
                })
        except Exception as exc:
            logger.debug("[Click] Background rec refresh failed: %s", exc)

    asyncio.ensure_future(_refresh_recs())

    # Return immediately with current feed recommendations
    return {"recommendation_movies": [], "event": event}


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

    try:
        uid = int(user_id)
    except ValueError:
        raise HTTPException(400, "user_id must be an integer")

    previous_rating = _user_rating_store.get((uid, movie_id))
    if previous_rating is None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_historical_user_ratings, uid)
        previous_rating = _user_rating_store.get((uid, movie_id))
    _record_rating(uid, movie_id, rating, previous_rating)

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
                    producer.flush(2),
                ),
            )
        except Exception as exc:
            logger.debug("[Rate] Kafka send failed (non-fatal): %s", exc)

    count = _rating_count_store.get(movie_id, 0)
    avg = round(_rating_sum_store[movie_id] / count, 2) if count else None
    return {
        "status": "ok",
        "movie_id": movie_id,
        "rating": rating,
        "user_id": user_id,
        "avg_rating": avg,
        "rating_count": count,
        "user_rating": rating,
    }


@app.get("/api/average_rating/{movie_id}")
async def average_rating(movie_id: int):
    """Return average rating and count for a movie from original + new ratings."""
    count = _rating_count_store.get(movie_id, 0)
    if not count:
        return {"avg_rating": None, "rating_count": 0}
    return {
        "avg_rating": round(_rating_sum_store[movie_id] / count, 2),
        "rating_count": count,
    }


@app.get("/api/user_rating/{movie_id}")
async def user_rating(movie_id: int, user_id: str = Query("1337")):
    """Return the current user's rating for this movie, if known."""
    try:
        uid = int(user_id)
    except ValueError:
        raise HTTPException(400, "user_id must be an integer")

    rating = _user_rating_store.get((uid, movie_id))
    if rating is None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_historical_user_ratings, uid)
        rating = _user_rating_store.get((uid, movie_id))
    return {"movie_id": movie_id, "user_id": uid, "user_rating": rating}


@app.get("/api/feed")
async def feed(
    limit: int = Query(50, ge=1, le=200),
    user_id: Optional[str] = Query(None),
):
    """
    Return the recent live interaction events + a personalised recommendation
    list. Recommendations come from the background cache populated by /api/click
    — this endpoint never blocks on the ML pipeline.
    """
    events = list(_interaction_feed)[:limit]

    # Serve cached recs from the most recent background refresh for this user.
    recs: List[Dict[str, Any]] = []
    for ev in events:
        if user_id is not None and str(ev.get("user")) != str(user_id):
            continue
        if ev.get("type") == "_rec_cache":
            recs = ev.get("movies") or []
            break

    # Fallback 1: generate recommendations on-demand for this user using
    # the latest clicked movie title as query context.
    if not recs and user_id is not None and agent is not None:
        latest_click_title = ""
        for ev in events:
            if str(ev.get("user")) == str(user_id) and ev.get("type") == "Click":
                latest_click_title = str(ev.get("detail") or "").strip()
                break
        query = latest_click_title or "popular movies"
        try:
            uid = int(user_id)
            result = await agent.recommend(query=query, user_id=uid, top_k=20)
            recs = [_enrich(r) for r in result.get("recommendation_movies", [])]
        except Exception as exc:
            logger.debug("[Feed] on-demand recommend failed: %s", exc)

    # Fallback 2: fast TF-IDF trending (no blocking ML call)
    if not recs and tfidf_engine.is_ready:
        try:
            trending_metas = tfidf_engine.trending(10)
            recs = [_enrich(CandidateMovie(
                movie_id=m.movie_id,
                title=m.title,
                genres=m.genres,
                year=m.year,
                description=m.description,
                poster=m.poster,
                sources=["TF-IDF"],
            )) for m in trending_metas]
        except Exception:
            pass

    # Filter internal cache events from the public feed
    public_events = [ev for ev in events if ev.get("type") != "_rec_cache"]
    return {"feed": public_events, "recommendation_movies": recs}


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
