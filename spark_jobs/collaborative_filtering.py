"""
Apache Spark ALS Collaborative Filtering Training Job
======================================================
WHY THIS EXISTS:
    The 20M-row rating CSV (MovieLens 25M) cannot fit in memory on a single
    machine for matrix factorisation at full fidelity.  PySpark distributes the
    ALS algorithm across a cluster of worker nodes, each computing a partition
    of the user/item factor updates in parallel.

    The output is a pair of Parquet files:
        models/als/user_factors.parquet   — shape (n_users, rank)
        models/als/item_factors.parquet   — shape (n_items, rank)

    These are loaded at FastAPI startup by CollaborativeFilteringEngine.load_from_parquet()
    for sub-millisecond in-process dot-product ranking.

OBJECTIVE (ALS cost function):
    min  Σ_{u,i ∈ Ω} (r_ui - u_u^T · v_i)²  +  λ(||u_u||² + ||v_i||²)
     U,V

    where Ω is the set of observed (user, movie) rating pairs.
    Alternating updates: fix V → solve for U; fix U → solve for V.

USAGE:
    # Local mode (dev)
    python spark_jobs/als_training.py --sample 0.1

    # Cluster mode (production)
    spark-submit --master spark://master:7077 spark_jobs/als_training.py

REFERENCE:
    Hu, Koren, Volinsky (2008). Collaborative Filtering for Implicit Feedback
    Datasets. IEEE ICDM 2008.
"""

# ============================================================
# TRAINING SCRIPT — ALS Collaborative Filtering v1
# ============================================================
# Experiment:       hybrid_recsys_als_v1
# Dataset:          data/process_movie_rating.csv
# Primary metric:   RMSE on held-out 20% test split
# Secondary:        Precision@10, NDCG@10
# Seed:             42 (fixed — do not change without logging)
# ============================================================

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("als_training")

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Spark ALS Training Job")
    p.add_argument("--ratings-csv", default=str(ROOT / "data" / "process_movie_rating.csv"))
    p.add_argument("--output-dir",  default=str(ROOT / "models" / "als"))
    p.add_argument("--rank",        type=int,   default=50,   help="Number of latent factors k")
    p.add_argument("--max-iter",    type=int,   default=15,   help="ALS iterations")
    p.add_argument("--reg-param",   type=float, default=0.1,  help="L2 regularisation λ")
    p.add_argument("--sample",      type=float, default=1.0,  help="Fraction of data to use (0-1]")
    p.add_argument("--test-split",  type=float, default=0.2,  help="Test set fraction")
    return p.parse_args()


def train(args):
    """
    Execute distributed ALS training on PySpark.

    Pipeline:
        1. Load ratings CSV → Spark DataFrame.
        2. Optional stratified sample for dev speed.
        3. Train/test split (seed=42 for reproducibility).
        4. Fit ALS model on train split.
        5. Evaluate RMSE on test split.
        6. Extract and save user/item factor matrices as Parquet.
        7. Log all hyperparameters and metrics.

    Args:
        args: Parsed CLI arguments.
    """
    from pyspark.sql import SparkSession
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql import functions as F

    t0 = time.perf_counter()

    spark = (
        SparkSession.builder
        .appName("HybridRecSys_ALS_v1")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("[ALS] Spark session started.")
    logger.info("[ALS] Config: rank=%d  maxIter=%d  regParam=%.3f  sample=%.2f",
                args.rank, args.max_iter, args.reg_param, args.sample)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    logger.info("[ALS] Loading ratings from %s …", args.ratings_csv)
    df = spark.read.csv(args.ratings_csv, header=True, inferSchema=True)
    df = df.select(
        F.col("userId").cast("int").alias("userId"),
        F.col("movieId").cast("int").alias("movieId"),
        F.col("rating").cast("float").alias("rating"),
    ).dropna()

    if args.sample < 1.0:
        df = df.sample(fraction=args.sample, seed=42)
        logger.info("[ALS] Sampled %.1f%% → %d rows", args.sample * 100, df.count())
    else:
        logger.info("[ALS] Full dataset: %d rows", df.count())

    # ── 2. Train / test split ──────────────────────────────────────────────────
    train_df, test_df = df.randomSplit([1 - args.test_split, args.test_split], seed=42)
    logger.info(
        "[ALS] Train=%d rows  Test=%d rows",
        train_df.count(), test_df.count(),
    )

    # ── 3. Fit ALS ─────────────────────────────────────────────────────────────
    als = ALS(
        maxIter=args.max_iter,
        regParam=args.reg_param,
        rank=args.rank,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",  # Drop NaN predictions for unknown users/items
        seed=42,
    )
    logger.info("[ALS] Fitting model …")
    model = als.fit(train_df)
    logger.info("[ALS] Model training complete.")

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction",
    )
    predictions = model.transform(test_df)
    rmse = evaluator.evaluate(predictions)
    elapsed = time.perf_counter() - t0
    logger.info("[ALS] RMSE on test set: %.4f", rmse)
    logger.info("[ALS] Total training time: %.1f seconds", elapsed)

    # ── 5. Save factor matrices ────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_factors_path = str(output_dir / "user_factors.parquet")
    item_factors_path = str(output_dir / "item_factors.parquet")

    logger.info("[ALS] Saving user factors → %s", user_factors_path)
    (
        model.userFactors
        .select(F.col("id"), F.col("features"))
        .write.mode("overwrite")
        .parquet(user_factors_path)
    )

    logger.info("[ALS] Saving item factors → %s", item_factors_path)
    (
        model.itemFactors
        .select(F.col("id"), F.col("features"))
        .write.mode("overwrite")
        .parquet(item_factors_path)
    )

    # ── 6. Save metrics log ────────────────────────────────────────────────────
    metrics_path = output_dir / "training_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"experiment: hybrid_recsys_als_v1\n")
        f.write(f"rank:       {args.rank}\n")
        f.write(f"max_iter:   {args.max_iter}\n")
        f.write(f"reg_param:  {args.reg_param}\n")
        f.write(f"sample:     {args.sample}\n")
        f.write(f"test_split: {args.test_split}\n")
        f.write(f"rmse:       {rmse:.6f}\n")
        f.write(f"elapsed_s:  {elapsed:.1f}\n")
    logger.info("[ALS] Metrics saved → %s", metrics_path)

    spark.stop()
    logger.info("[ALS] Done. User factors: %s  Item factors: %s",
                user_factors_path, item_factors_path)
    return {
        "rmse": rmse,
        "elapsed_s": elapsed,
        "user_factors": user_factors_path,
        "item_factors": item_factors_path,
    }


if __name__ == "__main__":
    args = parse_args()
    train(args)
