"""
Neural Collaborative Filtering (NCF) Training Script
======================================================
WHY THIS EXISTS:
    ALS learns linear latent-factor interactions (dot-product).  For complex,
    non-linear preference patterns (e.g., "I like horror but only when combined
    with dark comedy"), NCF's MLP branch can learn those functions from data,
    outperforming ALS on sparse, long-tail distributions.

    NCF combines two paths:
      - GMF (Generalised Matrix Factorisation) — element-wise product of embeddings
      - MLP — multi-layer perceptron on concatenated embeddings
    The outputs are concatenated and projected to a rating prediction.

TRAINING SETUP:
    - Dataset  : MovieLens (ratings ≥ 4.0 treated as positive implicit feedback)
    - Loss     : Binary Cross-Entropy (implicit feedback)
    - Neg ratio: 4 negative samples per positive (random sampling)
    - Optimizer: Adam with lr=1e-3, weight_decay=1e-5
    - Metric   : HR@10 (Hit Ratio at 10), NDCG@10

USAGE:
    python spark_jobs/ncf_training.py --epochs 20 --emb-dim 32

REFERENCE:
    He et al. (2017). "Neural Collaborative Filtering." WWW 2017.
    https://arxiv.org/abs/1708.05031
"""

# ============================================================
# TRAINING SCRIPT — NCF v1
# ============================================================
# Experiment:       hybrid_recsys_ncf_v1
# Dataset:          data/process_movie_rating.csv
# Primary metric:   HR@10 on leave-one-out test protocol
# Secondary:        NDCG@10
# Seed:             42
# Hardware:         CPU (or CUDA if available)
# ============================================================

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ncf_training")

ROOT = Path(__file__).resolve().parent.parent

# ─── Reproducibility seeds ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def parse_args():
    p = argparse.ArgumentParser(description="NCF Training")
    p.add_argument("--ratings-csv", default=str(ROOT / "data" / "process_movie_rating.csv"))
    p.add_argument("--output-dir",  default=str(ROOT / "models" / "ncf"))
    p.add_argument("--sample",      type=float, default=0.05, help="Fraction of data")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch-size",  type=int,   default=2048)
    p.add_argument("--emb-dim",     type=int,   default=32)
    p.add_argument("--neg-ratio",   type=int,   default=4,   help="Neg samples per positive")
    p.add_argument("--lr",          type=float, default=1e-3)
    return p.parse_args()


def build_dataset(df: pd.DataFrame, neg_ratio: int):
    """
    Build implicit feedback dataset.

    WHY THIS EXISTS:
        ALS works on explicit ratings.  NCF is trained as a binary classification
        task (1 = interacted, 0 = not interacted), which matches real-world implicit
        signals (views, clicks) better than raw star ratings.

    Args:
        df       : DataFrame with userId, movieId, rating columns.
        neg_ratio: Number of unobserved (user, movie) pairs to sample per positive.

    Returns:
        Tuple (user_ids, item_ids, labels) as numpy arrays.
    """
    # Treat rating ≥ 3.5 as positive implicit interaction
    positives = df[df["rating"] >= 3.5][["userId", "movieId"]].values
    all_items = df["movieId"].unique()

    users, items, labels = [], [], []
    for u, i in positives:
        users.append(u)
        items.append(i)
        labels.append(1.0)
        # Negative sampling
        for _ in range(neg_ratio):
            neg = np.random.choice(all_items)
            users.append(u)
            items.append(neg)
            labels.append(0.0)

    return np.array(users), np.array(items), np.array(labels, dtype=np.float32)


def train(args):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        return

    # Set CUDA seeds
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[NCF] Device: %s", device)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("[NCF] Loading ratings from %s (sample=%.1f%%) …",
                args.ratings_csv, args.sample * 100)
    df = pd.read_csv(args.ratings_csv, usecols=["userId", "movieId", "rating"])
    if args.sample < 1.0:
        df = df.sample(frac=args.sample, random_state=SEED)

    # Remap to contiguous IDs
    user_map = {u: i for i, u in enumerate(df["userId"].unique())}
    item_map = {m: i for i, m in enumerate(df["movieId"].unique())}
    df["u"] = df["userId"].map(user_map)
    df["m"] = df["movieId"].map(item_map)
    n_users, n_items = len(user_map), len(item_map)
    logger.info("[NCF] n_users=%d  n_items=%d", n_users, n_items)

    # ── Build implicit dataset ────────────────────────────────────────────────
    logger.info("[NCF] Building implicit dataset (neg_ratio=%d) …", args.neg_ratio)
    users_arr, items_arr, labels_arr = build_dataset(
        df.rename(columns={"u": "userId", "m": "movieId"}),
        args.neg_ratio,
    )
    dataset = TensorDataset(
        torch.LongTensor(users_arr),
        torch.LongTensor(items_arr),
        torch.FloatTensor(labels_arr),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ── Build NCF model ───────────────────────────────────────────────────────
    sys_path_patch = str(ROOT / "src")
    import sys
    if sys_path_patch not in sys.path:
        sys.path.insert(0, sys_path_patch)
    from recsys.engines.cf_engine import _NCFModel

    model = _NCFModel(n_users, n_items, emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("[NCF] Starting training for %d epochs …", args.epochs)
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for u_batch, i_batch, y_batch in loader:
            u_batch = u_batch.to(device)
            i_batch = i_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(u_batch, i_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info("[NCF] Epoch %02d/%02d  loss=%.5f", epoch, args.epochs, avg_loss)

    elapsed = time.perf_counter() - t0
    logger.info("[NCF] Training done in %.1fs", elapsed)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ncf_checkpoint.pt"
    meta_path = out_dir / "ncf_meta.pkl"

    torch.save(model.state_dict(), ckpt_path)
    import pickle
    with open(meta_path, "wb") as f:
        pickle.dump({
            "user_map": user_map,
            "item_map": item_map,
            "n_users": n_users,
            "n_items": n_items,
            "emb_dim": args.emb_dim,
        }, f)

    logger.info("[NCF] Checkpoint saved → %s", ckpt_path)
    logger.info("[NCF] Metadata saved  → %s", meta_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)
