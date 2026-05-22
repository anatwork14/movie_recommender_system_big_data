"""
TF-IDF Lexical Search Engine
==============================
WHY THIS EXISTS:
    For precise entity queries (e.g., "Christopher Nolan", "Leonardo DiCaprio"),
    semantic similarity is overkill and expensive. TF-IDF provides sub-millisecond
    exact lexical match against the combined metadata text of every movie, returning
    a sparse cosine-similarity score with 100% entity recall.

IMPLEMENTATION NOTE:
    We build a single "soup" document per movie that concatenates genres, title,
    cast, director, and description. This mirrors how practitioners in production
    RecSys pipelines handle metadata-based retrieval.

    Formula:
        TF-IDF(t, d, D) = TF(t, d) × log(|D| / (1 + |{d ∈ D : t ∈ d}|))
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Trailing article normalizer  (e.g. "Matrix, The (1999)" → "The Matrix (1999)")
# ─────────────────────────────────────────────────────────────
_TRAILING_ARTICLE_RE = re.compile(
    r'^(?P<title>.+),\s*(?P<article>The|A|An)(?P<year>\s+\(\d{4}\))?\s*$'
)


def _display_title(raw: str) -> str:
    raw = str(raw or "Unknown").strip()
    m = _TRAILING_ARTICLE_RE.match(raw)
    if not m:
        return raw
    return f"{m.group('article')} {m.group('title')}{m.group('year') or ''}"


@dataclass
class TFIDFResult:
    """A single hit returned by the TF-IDF engine."""
    movie_id: int
    title: str
    genres: str
    year: str
    description: str
    poster: str
    score: float  # Cosine similarity in TF-IDF sparse space


class TFIDFEngine:
    """
    Builds and queries a TF-IDF matrix over movie metadata.

    Attributes:
        df        : Raw DataFrame with movie metadata.
        vectorizer: Fitted TfidfVectorizer instance.
        matrix    : Sparse TF-IDF matrix, shape (n_movies, n_terms).
    """

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None  # scipy sparse matrix
        self._is_ready = False

    # ── Public API ──────────────────────────────────────────────────────────

    def build(self, csv_path: str) -> None:
        """
        Load movie CSV and fit the TF-IDF vectorizer.

        WHY THIS EXISTS:
            Fitting happens once at startup and the result is kept in RAM as a
            sparse matrix. The footprint for 26 k movies is ~ 15–25 MB – negligible
            compared to the embedding model or Qdrant index.

        Args:
            csv_path: Absolute or relative path to process_movie.csv.
        """
        t0 = time.perf_counter()
        df = pd.read_csv(csv_path)
        df["year"] = df["year"].fillna("Unknown").astype(str).str.replace(r"\.0$", "", regex=True)
        df["title_display"] = df["title"].apply(_display_title)
        df["poster"] = df.get("poster", pd.Series([""] * len(df))).fillna("")

        # Build semantic soup document per movie
        df["soup"] = (
            df["title_display"].fillna("")
            + " " + df["genres"].fillna("").str.replace("|", " ", regex=False)
            + " " + df["description"].fillna("")
        )

        self.df = df.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,   # apply log(1 + TF) instead of raw TF
        )
        self.matrix = self.vectorizer.fit_transform(self.df["soup"])
        self._is_ready = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "[TF-IDF] Built index for %d movies in %.2fs  "
            "(vocab size=%d, matrix shape=%s)",
            len(self.df),
            elapsed,
            len(self.vectorizer.vocabulary_),
            self.matrix.shape,
        )

    def search(self, query: str, limit: int = 100) -> List[TFIDFResult]:
        """
        Execute a lexical search query against the TF-IDF matrix.

        WHY THIS EXISTS:
            Exact-entity queries ("Leonardo DiCaprio", "Quentin Tarantino") where
            the token must appear verbatim in the metadata. Dense semantic search
            would dilute these with genre-adjacent noise.

        Args:
            query : Free-text search string (e.g. "Christopher Nolan sci-fi").
            limit : Maximum number of hits to return.

        Returns:
            List of TFIDFResult sorted by cosine similarity (descending).
        """
        self._require_ready()
        query_vec = self.vectorizer.transform([query])
        # linear_kernel is cosine similarity for L2-normalised TF-IDF vectors
        scores = linear_kernel(query_vec, self.matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:limit]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            row = self.df.iloc[idx]
            results.append(
                TFIDFResult(
                    movie_id=int(row["movieId"]),
                    title=str(row["title_display"]),
                    genres=str(row["genres"]),
                    year=str(row["year"]),
                    description=str(row["description"]),
                    poster=str(row["poster"]),
                    score=float(scores[idx]),
                )
            )
        return results

    def get_by_id(self, movie_id: int) -> Optional[TFIDFResult]:
        """Retrieve a single movie's metadata by its integer movie_id."""
        self._require_ready()
        rows = self.df[self.df["movieId"] == movie_id]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return TFIDFResult(
            movie_id=int(row["movieId"]),
            title=str(row["title_display"]),
            genres=str(row["genres"]),
            year=str(row["year"]),
            description=str(row["description"]),
            poster=str(row["poster"]),
            score=1.0,
        )

    def trending(self, limit: int = 20) -> List[TFIDFResult]:
        """Return the first N movies from the index (used as a trending placeholder)."""
        self._require_ready()
        subset = self.df.head(limit)
        return [
            TFIDFResult(
                movie_id=int(row["movieId"]),
                title=str(row["title_display"]),
                genres=str(row["genres"]),
                year=str(row["year"]),
                description=str(row["description"]),
                poster=str(row["poster"]),
                score=1.0,
            )
            for _, row in subset.iterrows()
        ]

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # ── Internals ────────────────────────────────────────────────────────────

    def _require_ready(self) -> None:
        if not self._is_ready:
            raise RuntimeError(
                "TFIDFEngine is not ready. Call .build(csv_path) first."
            )
