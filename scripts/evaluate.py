#!/usr/bin/env python3
"""Offline evaluator for CF, TF-IDF, and CF+TF-IDF hybrid on MovieLens artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class RankingMetrics:
    users_evaluated: int
    hit_rate_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    catalog_coverage_at_k: float
    mean_latency_ms: float
    p95_latency_ms: float


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Evaluate recommender approaches offline")
    p.add_argument("--root", default=str(root), help="Project root")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--users", type=int, default=500, help="Number of users for ranking evaluation")
    p.add_argument("--min-ratings", type=int, default=8, help="Min ratings per user to be eligible")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(root / "models" / "evaluation_report.json"))
    return p.parse_args()


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def safe_float(v: float) -> float:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return float("nan")
    return float(v)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    root = Path(args.root).resolve()
    data_dir = root / "data"
    models_dir = root / "models"

    ratings_csv = data_dir / "process_movie_rating.csv"
    movies_csv = data_dir / "process_movie.csv"
    user_factors_path = models_dir / "als" / "user_factors.parquet"
    item_factors_path = models_dir / "als" / "item_factors.parquet"

    if not ratings_csv.exists() or not movies_csv.exists():
        raise FileNotFoundError("Missing dataset files under data/")

    # Avoid importing recsys package root (it pulls Kafka deps).
    sys.path.insert(0, str(root / "src" / "recsys" / "engines"))
    from cf_engine import CollaborativeFilteringEngine
    from tfidf_engine import TFIDFEngine
    from hybrid_fusion import HybridFusionEngine

    ratings = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "rating", "timestamp"])
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], errors="coerce")

    # 1) Load CF factors and compute pointwise RMSE/MAE where user+item embeddings exist.
    cf = CollaborativeFilteringEngine()
    if user_factors_path.exists() and item_factors_path.exists():
        cf.load_from_parquet(str(user_factors_path), str(item_factors_path))
    else:
        cf.load_from_pickle(str(models_dir / "als_factors.pkl"))

    u_map = ratings["userId"].map(cf.user_id_map)
    i_map = ratings["movieId"].map(cf.item_id_map)
    known_mask = u_map.notna() & i_map.notna()
    known = ratings.loc[known_mask].copy()
    u_idx = u_map[known_mask].astype(int).to_numpy()
    i_idx = i_map[known_mask].astype(int).to_numpy()
    preds = np.einsum("ij,ij->i", cf.user_factors[u_idx], cf.item_factors[i_idx])
    truth = known["rating"].to_numpy(dtype=np.float32)
    rmse = float(np.sqrt(np.mean((truth - preds) ** 2)))
    mae = float(np.mean(np.abs(truth - preds)))

    # 2) Build TF-IDF index.
    tfidf = TFIDFEngine()
    tfidf.build(str(movies_csv))
    movie_title = dict(zip(tfidf.df["movieId"].astype(int), tfidf.df["title_display"].astype(str)))

    # 3) User-level leave-one-out split for ranking.
    user_counts = ratings.groupby("userId").size()
    eligible_users = user_counts[user_counts >= args.min_ratings].index.tolist()
    rng.shuffle(eligible_users)
    selected_users = eligible_users[: args.users]

    by_user = {uid: frame.sort_values("timestamp") for uid, frame in ratings.groupby("userId") if uid in set(selected_users)}

    fusion = HybridFusionEngine()
    all_catalog = set(int(x) for x in tfidf.df["movieId"].tolist())

    def eval_ranker(mode: str) -> RankingMetrics:
        hits = 0
        ndcg_sum = 0.0
        mrr_sum = 0.0
        lat_ms: List[float] = []
        recommended_items = set()
        eval_users = 0

        for uid in selected_users:
            frame = by_user.get(uid)
            if frame is None or len(frame) < args.min_ratings:
                continue

            holdout = int(frame.iloc[-1]["movieId"])
            history = frame.iloc[:-1]
            exclude = history["movieId"].astype(int).tolist()

            t0 = time.perf_counter()
            if mode == "cf":
                recs = cf.recommend(user_id=int(uid), limit=args.k, exclude_ids=exclude)
                rec_ids = [int(r.movie_id) for r in recs]
            elif mode == "tfidf":
                recent_titles = [movie_title.get(int(m), "") for m in history.tail(3)["movieId"].tolist()]
                query = " ".join([t for t in recent_titles if t]).strip()
                if not query:
                    continue
                hits_tfidf = tfidf.search(query, limit=max(args.k + 1, 20))
                rec_ids = []
                for h in hits_tfidf:
                    mid = int(h.movie_id)
                    if mid in exclude:
                        continue
                    rec_ids.append(mid)
                    if len(rec_ids) >= args.k:
                        break
            elif mode == "hybrid":
                cf_hits = cf.recommend(user_id=int(uid), limit=100, exclude_ids=exclude)
                recent_titles = [movie_title.get(int(m), "") for m in history.tail(3)["movieId"].tolist()]
                query = " ".join([t for t in recent_titles if t]).strip()
                tf_hits = tfidf.search(query, limit=100) if query else []
                merged = fusion.fuse(cf_results=cf_hits, tfidf_results=tf_hits, rag_results=[], movie_meta_fn=tfidf.get_by_id, top_k=args.k)
                rec_ids = [int(m.movie_id) for m in merged if int(m.movie_id) not in set(exclude)]
                rec_ids = rec_ids[: args.k]
            else:
                raise ValueError(mode)

            lat_ms.append((time.perf_counter() - t0) * 1000.0)

            if not rec_ids:
                continue

            eval_users += 1
            recommended_items.update(rec_ids)

            if holdout in rec_ids:
                rank = rec_ids.index(holdout) + 1
                hits += 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
                mrr_sum += 1.0 / rank

        catalog_size = max(1, len(cf.item_id_map))
        return RankingMetrics(
            users_evaluated=eval_users,
            hit_rate_at_k=safe_float(hits / eval_users if eval_users else float("nan")),
            ndcg_at_k=safe_float(ndcg_sum / eval_users if eval_users else float("nan")),
            mrr_at_k=safe_float(mrr_sum / eval_users if eval_users else float("nan")),
            catalog_coverage_at_k=safe_float(len(recommended_items) / catalog_size),
            mean_latency_ms=safe_float(float(np.mean(lat_ms)) if lat_ms else float("nan")),
            p95_latency_ms=safe_float(percentile(lat_ms, 95.0)),
        )

    cf_rank = eval_ranker("cf")
    tfidf_rank = eval_ranker("tfidf")
    hybrid_rank = eval_ranker("hybrid")

    # 3b) Standard implicit candidate ranking (1 positive + 100 negatives).
    def implicit_candidate_eval(negatives: int = 100) -> Dict[str, Dict[str, float]]:
        def topk_from_scores(score_pairs: List[tuple[int, float]], k: int) -> List[int]:
            score_pairs.sort(key=lambda x: x[1], reverse=True)
            return [mid for mid, _ in score_pairs[:k]]

        cf_hits = tf_hits = hy_hits = 0
        cf_ndcg = tf_ndcg = hy_ndcg = 0.0
        cf_mrr = tf_mrr = hy_mrr = 0.0
        users_used = 0

        for uid in selected_users:
            frame = by_user.get(uid)
            if frame is None:
                continue
            pos_frame = frame[frame["rating"] >= 4.0]
            if len(pos_frame) < 2:
                continue

            holdout = int(pos_frame.iloc[-1]["movieId"])
            train_pos = set(int(m) for m in pos_frame.iloc[:-1]["movieId"].tolist())
            if not train_pos:
                continue

            negatives_pool = list(all_catalog - train_pos - {holdout})
            if len(negatives_pool) < negatives:
                continue
            sampled_negs = rng.sample(negatives_pool, negatives)
            candidates = [holdout] + sampled_negs

            # CF scoring on candidate set
            cf_scores = []
            u_idx = cf.user_id_map.get(int(uid))
            if u_idx is not None:
                u_vec = cf.user_factors[u_idx]
                for mid in candidates:
                    i_idx = cf.item_id_map.get(int(mid))
                    if i_idx is None:
                        continue
                    cf_scores.append((mid, float(np.dot(u_vec, cf.item_factors[i_idx]))))

            # TF-IDF scoring on candidate set via rank proxy
            recent_titles = [movie_title.get(int(m), "") for m in list(train_pos)[-3:]]
            query = " ".join([t for t in recent_titles if t]).strip()
            tf_scores = []
            if query:
                tf_res = tfidf.search(query, limit=1000)
                tf_rank_map = {int(r.movie_id): idx + 1 for idx, r in enumerate(tf_res)}
                for mid in candidates:
                    rank = tf_rank_map.get(int(mid))
                    if rank is not None:
                        tf_scores.append((mid, 1.0 / (60 + rank)))

            # Hybrid score from rank sums (RRF-style)
            hy_scores_map: Dict[int, float] = {}
            for rank, (mid, _) in enumerate(sorted(cf_scores, key=lambda x: x[1], reverse=True), start=1):
                hy_scores_map[mid] = hy_scores_map.get(mid, 0.0) + 1.0 / (60 + rank)
            for rank, (mid, _) in enumerate(sorted(tf_scores, key=lambda x: x[1], reverse=True), start=1):
                hy_scores_map[mid] = hy_scores_map.get(mid, 0.0) + 1.0 / (60 + rank)
            hy_scores = list(hy_scores_map.items())

            cf_top = topk_from_scores(cf_scores, args.k)
            tf_top = topk_from_scores(tf_scores, args.k)
            hy_top = topk_from_scores(hy_scores, args.k)

            users_used += 1

            for top, hit_acc, ndcg_acc, mrr_acc in (
                (cf_top, "cf_hits", "cf_ndcg", "cf_mrr"),
                (tf_top, "tf_hits", "tf_ndcg", "tf_mrr"),
                (hy_top, "hy_hits", "hy_ndcg", "hy_mrr"),
            ):
                if holdout in top:
                    rank = top.index(holdout) + 1
                    if hit_acc == "cf_hits":
                        cf_hits += 1
                        cf_ndcg += 1.0 / math.log2(rank + 1)
                        cf_mrr += 1.0 / rank
                    elif hit_acc == "tf_hits":
                        tf_hits += 1
                        tf_ndcg += 1.0 / math.log2(rank + 1)
                        tf_mrr += 1.0 / rank
                    else:
                        hy_hits += 1
                        hy_ndcg += 1.0 / math.log2(rank + 1)
                        hy_mrr += 1.0 / rank

        denom = max(users_used, 1)
        return {
            "users_evaluated": users_used,
            "candidate_size": negatives + 1,
            "cf": {
                "hit_rate_at_k": cf_hits / denom,
                "ndcg_at_k": cf_ndcg / denom,
                "mrr_at_k": cf_mrr / denom,
            },
            "tfidf": {
                "hit_rate_at_k": tf_hits / denom,
                "ndcg_at_k": tf_ndcg / denom,
                "mrr_at_k": tf_mrr / denom,
            },
            "hybrid_cf_tfidf_rrf": {
                "hit_rate_at_k": hy_hits / denom,
                "ndcg_at_k": hy_ndcg / denom,
                "mrr_at_k": hy_mrr / denom,
            },
        }

    implicit_eval = implicit_candidate_eval(negatives=100)

    # 4) TF-IDF lexical self-retrieval sanity check.
    sample_movies = tfidf.df.sample(n=min(500, len(tfidf.df)), random_state=args.seed)
    self_top1 = 0
    self_top10 = 0
    self_lat = []
    for row in sample_movies.itertuples(index=False):
        q = str(row.title_display)
        t0 = time.perf_counter()
        res = tfidf.search(q, limit=10)
        self_lat.append((time.perf_counter() - t0) * 1000.0)
        mids = [int(r.movie_id) for r in res]
        mid = int(row.movieId)
        if mids[:1] and mids[0] == mid:
            self_top1 += 1
        if mid in mids:
            self_top10 += 1

    report = {
        "config": {
            "k": args.k,
            "users_requested": args.users,
            "min_ratings_per_user": args.min_ratings,
            "seed": args.seed,
        },
        "dataset": {
            "ratings_rows": int(len(ratings)),
            "users": int(ratings["userId"].nunique()),
            "movies": int(ratings["movieId"].nunique()),
            "eligible_users": int(len(eligible_users)),
        },
        "cf_pointwise": {
            "known_interactions": int(known_mask.sum()),
            "rmse": safe_float(rmse),
            "mae": safe_float(mae),
        },
        "ranking_at_k": {
            "cf": asdict(cf_rank),
            "tfidf": asdict(tfidf_rank),
            "hybrid_cf_tfidf_rrf": asdict(hybrid_rank),
        },
        "implicit_candidate_ranking": implicit_eval,
        "tfidf_self_retrieval": {
            "queries": int(len(sample_movies)),
            "recall_at_1": safe_float(self_top1 / len(sample_movies) if len(sample_movies) else float("nan")),
            "recall_at_10": safe_float(self_top10 / len(sample_movies) if len(sample_movies) else float("nan")),
            "mean_latency_ms": safe_float(float(np.mean(self_lat)) if self_lat else float("nan")),
            "p95_latency_ms": safe_float(percentile(self_lat, 95.0)),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Evaluation complete.")
    print(json.dumps(report, indent=2))
    print(f"Saved report to: {out}")


if __name__ == "__main__":
    main()
