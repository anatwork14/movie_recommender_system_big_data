"""
Collaborative Filtering Engine (ALS + NCF)
============================================
WHY THIS EXISTS:
    The TF-IDF and RAG engines work on item metadata only – they cannot model the
    collective intelligence embedded in millions of user interaction events.  CF
    mines the User-Item interaction matrix for latent patterns that transcend any
    individual item's content description.

    This module provides two interchangeable backends:
      1. ALS (via PySpark) – industrial-strength distributed factorisation used in
         production at Netflix, Spotify, etc.  Train results are saved as Parquet
         and loaded here as a precomputed lookup table.
      2. NCF (PyTorch) – deep matrix factorisation that learns non-linear interaction
         functions; useful when you have GPU budget and want a learnable embedding
         tower beyond inner-product.

COLD START STRATEGY:
    If a user_id has no stored embedding (new user), fall back gracefully:
        - Return an empty list so the Agent router activates the RAG fallback.
        - Never raise – the caller must not see CF failures as fatal.

REFERENCES:
    He et al. (2017). "Neural Collaborative Filtering." WWW 2017.
    https://arxiv.org/abs/1708.05031

    Hu, Koren, Volinsky (2008). "Collaborative Filtering for Implicit Feedback."
    IEEE ICDM 2008.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CFResult:
    """A movie predicted to be of interest to a specific user."""
    movie_id: int
    score: float          # predicted rating / dot-product score
    rank: int             # 1-based position within this engine's output


# ─────────────────────────────────────────────────────────────────────────────
# Neural Collaborative Filtering  (PyTorch, standalone)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("[CF] PyTorch not found – NCF backend will be disabled.")


class _NCFModel(nn.Module if _TORCH_AVAILABLE else object):
    """
    Two-branch NCF: GMF path (element-wise product) + MLP path.
    Concatenated and projected to a single rating-logit.

    Architecture:
        GMF branch : user_emb ⊙ item_emb  →  ReLU → Linear(k, 1)
        MLP branch : concat(user_emb, item_emb) → [Linear-BN-ReLU] × 3 → Linear(h, 1)
        Output     : sigmoid(gmf_out + mlp_out)  ∈ [0, 1]
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32, mlp_layers: Tuple[int, ...] = (128, 64, 32)):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for NCF.")
        super().__init__()

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(n_items, emb_dim)

        # MLP embeddings
        self.mlp_user_emb = nn.Embedding(n_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, emb_dim)

        # MLP tower
        mlp_input_dim = emb_dim * 2
        layers: List[nn.Module] = []
        for out_dim in mlp_layers:
            layers += [
                nn.Linear(mlp_input_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            ]
            mlp_input_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Output projection
        self.output_layer = nn.Linear(mlp_layers[-1] + emb_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, user_ids: "torch.Tensor", item_ids: "torch.Tensor") -> "torch.Tensor":
        # GMF path
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i  # element-wise

        # MLP path
        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_cat = torch.cat([mlp_u, mlp_i], dim=1)
        mlp_out = self.mlp(mlp_cat)

        # Concat GMF + MLP → output
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        return self.sigmoid(self.output_layer(combined)).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# ALS-based CF Engine  (uses precomputed embedding lookup)
# ─────────────────────────────────────────────────────────────────────────────

class CollaborativeFilteringEngine:
    """
    Loads ALS user/item factor matrices produced by spark_jobs/als_training.py
    and serves top-K recommendations at sub-millisecond latency via dot-product
    on the pre-computed factor matrices.

    WHY PRECOMPUTED:
        Calling PySpark at inference time is prohibitively slow (JVM warmup + DAG
        compilation takes > 5 s per request).  Instead, the ALS Spark job saves
        the factor matrices to Parquet once per day.  We load them into numpy at
        startup and do the ranking in-process.

    COLD START:
        If user_id has no factor vector, return [] so the Agent gracefully falls
        back to RAG or TF-IDF.

    Attributes:
        user_factors  : np.ndarray shape (n_users, k)
        item_factors  : np.ndarray shape (n_items, k)
        user_id_map   : {original_user_id: row_index}
        item_id_map   : {original_movie_id: col_index}
        item_id_reverse: {col_index: original_movie_id}
    """

    def __init__(self) -> None:
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.item_id_reverse: Dict[int, int] = {}
        self._is_ready = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_from_parquet(self, user_factors_path: str, item_factors_path: str) -> None:
        """
        Load user and item factor matrices from Parquet produced by the ALS Spark job.

        Args:
            user_factors_path : Path to user_factors.parquet directory.
            item_factors_path : Path to item_factors.parquet directory.
        """
        logger.info("[CF] Loading ALS factors from Parquet …")
        user_df = pd.read_parquet(user_factors_path)
        item_df = pd.read_parquet(item_factors_path)

        # Expect columns: id, features (list of float)
        self.user_id_map = {int(uid): idx for idx, uid in enumerate(user_df["id"])}
        self.item_id_map = {int(iid): idx for idx, iid in enumerate(item_df["id"])}
        self.item_id_reverse = {v: k for k, v in self.item_id_map.items()}

        self.user_factors = np.stack(user_df["features"].tolist()).astype(np.float32)
        self.item_factors = np.stack(item_df["features"].tolist()).astype(np.float32)
        self._is_ready = True
        logger.info(
            "[CF] ALS factors loaded: %d users × %d items, rank=%d",
            len(self.user_id_map),
            len(self.item_id_map),
            self.user_factors.shape[1],
        )

    def load_from_pickle(self, path: str) -> None:
        """Alternative fast-load from a pickle produced by the lightweight trainer."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.user_factors = state["user_factors"]
        self.item_factors = state["item_factors"]
        self.user_id_map = state["user_id_map"]
        self.item_id_map = state["item_id_map"]
        self.item_id_reverse = {v: k for k, v in self.item_id_map.items()}
        self._is_ready = True
        logger.info("[CF] ALS factors loaded from pickle: %s", path)

    def build_from_ratings_csv(self, csv_path: str, rank: int = 50, max_iter: int = 15,
                               reg_param: float = 0.1, sample_frac: float = 0.05) -> None:
        """
        Lightweight in-process ALS using numpy for demo/dev purposes.
        For production, use spark_jobs/als_training.py instead.

        WHY THIS EXISTS:
            Allows the API server to boot and serve CF recommendations even when
            the full PySpark Spark job has not been run yet.  We sample the rating
            CSV to keep memory usage under 500 MB.

        Args:
            csv_path    : Path to process_movie_rating.csv
            rank        : Number of latent factors k.
            max_iter    : ALS iteration count.
            reg_param   : L2 regularisation λ.
            sample_frac : Fraction of the 20M ratings to sample (keeps RAM < 1 GB).
        """
        import time
        logger.info("[CF] Building in-process ALS from CSV (sample=%.1f%%) …", sample_frac * 100)
        t0 = time.perf_counter()

        df = pd.read_csv(csv_path, usecols=["userId", "movieId", "rating"])
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)

        # Remap IDs to contiguous indices
        user_ids = df["userId"].unique()
        item_ids = df["movieId"].unique()
        self.user_id_map = {int(uid): idx for idx, uid in enumerate(user_ids)}
        self.item_id_map = {int(iid): idx for idx, iid in enumerate(item_ids)}
        self.item_id_reverse = {v: k for k, v in self.item_id_map.items()}

        n_users = len(user_ids)
        n_items = len(item_ids)

        rng = np.random.default_rng(42)
        U = rng.standard_normal((n_users, rank)).astype(np.float32) * 0.01
        V = rng.standard_normal((n_items, rank)).astype(np.float32) * 0.01

        u_indices = df["userId"].map(self.user_id_map).values
        i_indices = df["movieId"].map(self.item_id_map).values
        ratings   = df["rating"].values.astype(np.float32)

        # Vectorised ALS: alternating ridge regression
        for iteration in range(max_iter):
            # Fix V, solve for U
            for u in range(n_users):
                mask = u_indices == u
                if not mask.any():
                    continue
                Vi = V[i_indices[mask]]   # (n_ratings_u, k)
                ri = ratings[mask]
                A = Vi.T @ Vi + reg_param * np.eye(rank)
                b = Vi.T @ ri
                U[u] = np.linalg.solve(A, b)

            # Fix U, solve for V
            for i in range(n_items):
                mask = i_indices == i
                if not mask.any():
                    continue
                Ui = U[u_indices[mask]]
                ri = ratings[mask]
                A = Ui.T @ Ui + reg_param * np.eye(rank)
                b = Ui.T @ ri
                V[i] = np.linalg.solve(A, b)

            # RMSE on training data
            preds = np.einsum('ij,ij->i', U[u_indices], V[i_indices])
            rmse = float(np.sqrt(np.mean((ratings - preds) ** 2)))
            logger.info("[CF] ALS iter %02d/%02d  RMSE=%.4f", iteration + 1, max_iter, rmse)

        self.user_factors = U
        self.item_factors = V
        self._is_ready = True
        logger.info(
            "[CF] ALS done in %.1fs  users=%d  items=%d  rank=%d",
            time.perf_counter() - t0, n_users, n_items, rank,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def recommend(self, user_id: int, limit: int = 100,
                  exclude_ids: Optional[List[int]] = None) -> List[CFResult]:
        """
        Return top-K movie recommendations for user_id via dot-product ranking.

        Args:
            user_id    : Original (non-mapped) user ID.
            limit      : Maximum candidates to return.
            exclude_ids: Movie IDs the user has already seen (filter out).

        Returns:
            Sorted list of CFResult (best first) or [] for cold-start users.
        """
        if not self._is_ready:
            return []

        u_idx = self.user_id_map.get(int(user_id))
        if u_idx is None:
            logger.debug("[CF] Cold-start: user_id=%d not in training set.", user_id)
            return []

        # Dot product: (k,) · (n_items, k).T = (n_items,)
        u_vec = self.user_factors[u_idx]
        scores = self.item_factors @ u_vec  # (n_items,)

        if exclude_ids:
            exclude_set = {self.item_id_map.get(int(m)) for m in exclude_ids if int(m) in self.item_id_map}
            exclude_set.discard(None)
            if exclude_set:
                scores[list(exclude_set)] = -np.inf

        top_indices = np.argsort(scores)[::-1][:limit]
        return [
            CFResult(
                movie_id=self.item_id_reverse[int(idx)],
                score=float(scores[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
            if scores[idx] > -np.inf
        ]

    @property
    def is_ready(self) -> bool:
        return self._is_ready
