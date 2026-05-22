# Hybrid Movie Recommender System (Big Data)

Production-style hybrid recommender that combines:
- Collaborative Filtering (ALS on MovieLens 25M-scale ratings)
- Content retrieval (TF-IDF lexical search)
- Semantic retrieval (Qdrant vector DB)
- Agentic orchestration + Reciprocal Rank Fusion (RRF)
- Real-time interaction streaming with Kafka

The project includes a FastAPI backend, a React frontend, Spark/PyTorch training jobs, and an offline evaluation script to compare approaches consistently.

## 1. What This Repository Contains

- `backend/`: FastAPI API and orchestration logic.
- `frontend/`: React/Vite web app.
- `src/recsys/engines/`: CF, TF-IDF, and fusion engines.
- `spark_jobs/collaborative_filtering.py`: Spark ALS training (explicit feedback).
- `spark_jobs/ncf_training.py`: Experimental NCF trainer (implicit feedback).
- `data_processing_pipeline/indexer.py`: Embedding + Qdrant indexing.
- `kafka_streaming/`: Kafka producer/consumer utilities and legacy Flask demo.
- `scripts/evaluate.py`: Unified offline metrics runner (CF, TF-IDF, Hybrid).

## 2. System Architecture

1. User actions (click/rate/search) come from the frontend.
2. Backend records interactions and can push to Kafka.
3. Candidate generators run in parallel:
- CF (ALS factors)
- TF-IDF lexical retrieval
- RAG semantic retrieval from Qdrant (if available)
4. `HybridFusionEngine` merges ranked lists with RRF.
5. API returns ranked movies + explainability metadata (`sources`, per-engine ranks).

## 3. Prerequisites

Required:
- Python 3.10+
- Node.js 18+
- npm
- Docker + Docker Compose

Optional:
- OpenAI API key for richer LLM-generated explanation text in `/api/recommend`

## 4. Environment Setup

From project root:

```bash
cp .env.example .env
```

Important `.env` keys:
- `PORT=5001`
- `QDRANT_HOST=localhost`
- `QDRANT_PORT=6333`
- `KAFKA_BOOTSTRAP=localhost:9092`
- `OPENAI_API_KEY=` (optional)

## 5. Quick Start (Recommended)

### Option A: One-command orchestration script

```bash
./run.sh up
```

This starts Docker services and checks/indexes Qdrant if needed.

### Option B: Docker Compose directly

```bash
docker-compose up --build -d
```

Exposed services:
- Frontend (Docker): `http://localhost:3005`
- Backend API docs: `http://localhost:5001/docs`
- Backend health: `http://localhost:5001/api/health`
- Qdrant: `http://localhost:6333`
- Kafka broker: `localhost:9092`

## 6. Local Development Mode

Use Docker for infra (Kafka + Qdrant), run app services locally:

```bash
./run.sh local
```

Or manually:

```bash
# terminal 1
python3 -m uvicorn backend.main:app --reload --port 5001

# terminal 2
cd frontend
npm install
npm run dev -- --host
```

Local frontend URL is typically Vite default (`http://localhost:5173`).

## 7. Data and Model Preparation

### 7.1 Build Qdrant semantic index

```bash
python3 data_processing_pipeline/indexer.py
```

### 7.2 Train Collaborative Filtering (ALS)

Full training:

```bash
python3 spark_jobs/collaborative_filtering.py --sample 1.0
```

Fast validation run:

```bash
python3 spark_jobs/collaborative_filtering.py --sample 0.1
```

Outputs:
- `models/als/user_factors.parquet`
- `models/als/item_factors.parquet`
- `models/als/training_metrics.txt`

### 7.3 Train Experimental NCF (optional)

```bash
python3 spark_jobs/ncf_training.py --epochs 20 --sample 0.05
```

Outputs:
- `models/ncf/ncf_checkpoint.pt`
- `models/ncf/ncf_meta.pkl`

Note: NCF artifacts are not wired into serving by default.

## 8. How To Demonstrate The Project Correctly

Use this flow for a robust demo:

1. Open frontend and verify backend health.
2. Login with an existing account from `backend` seeded users.
3. Home page:
- Show `Top Trending` row.
- Show `Recommended for You` and switch source filters (`All`, `CF`, `TF-IDF`).
4. Search page:
- Query by movie title and mood/genre text.
- Open a result and verify detail page metadata.
5. Movie detail:
- Submit a rating.
- Confirm average rating and status update.
6. Agent panel:
- Ask natural language request (e.g. "dark sci-fi with mind-bending plot").
- Explain provenance badges (`CF`, `TF-IDF`, `RAG`) and merged behavior.
7. Kafka feed:
- Trigger clicks/ratings and show live events in sidebar.

## 9. API Surface (Main Endpoints)

- `GET /api/health`
- `GET /api/trending`
- `GET /api/movie/{movie_id}`
- `GET /api/search?q=...&limit=...`
- `GET /api/feed`
- `GET /api/click/{movie_id}`
- `GET /api/rate/{movie_id}/{rating}`
- `GET /api/average_rating/{movie_id}`
- `POST /api/recommend`

## 10. Evaluation and Metrics

This repository now includes a reproducible evaluator:

```bash
python3 scripts/evaluate.py --users 300 --k 10
```

Generated report:
- `models/evaluation_report.json`

### 10.1 Metrics included

`CF pointwise metrics`:
- RMSE
- MAE

`Ranking metrics @K` (strict leave-one-out next-item style):
- HitRate@K
- NDCG@K
- MRR@K
- Catalog coverage@K
- Mean/P95 latency (ms)

`Implicit candidate ranking` (1 positive + 100 negatives per user, standard implicit protocol):
- HitRate@K
- NDCG@K
- MRR@K

`TF-IDF lexical sanity`:
- Self-retrieval Recall@1
- Self-retrieval Recall@10
- Mean/P95 query latency (ms)

### 10.2 Latest metrics from this environment (2026-05-22)

From `models/evaluation_report.json` (`--users 300 --k 10`):

- Dataset:
  - 19,937,428 ratings
  - 138,493 users
  - 25,450 movies

- CF pointwise:
  - RMSE: `0.7761`
  - MAE: `0.6060`

- Strict ranking@10:
  - CF HitRate@10: `0.0000`
  - TF-IDF HitRate@10: `0.0133`
  - Hybrid(CF+TF-IDF) HitRate@10: `0.0033`

- Implicit candidate ranking@10 (more stable for recommender comparison):
  - CF HitRate@10: `0.5034`
  - TF-IDF HitRate@10: `0.0369`
  - Hybrid(CF+TF-IDF) HitRate@10: `0.4362`

- TF-IDF self-retrieval:
  - Recall@1: `0.772`
  - Recall@10: `0.922`
  - Mean latency: `7.29 ms`

Interpretation:
- CF is strongest on implicit candidate ranking (personal preference signal).
- TF-IDF provides broad lexical coverage and useful recall for search-like behavior.
- Hybrid is beneficial for explainability/diversification, but relative gains depend on query/user protocol and fusion weights.

### 10.3 Re-running ALS training metrics

```bash
python3 spark_jobs/collaborative_filtering.py --sample 1.0
cat models/als/training_metrics.txt
```

Current saved ALS training summary (`models/als/training_metrics.txt`):
- rank: `50`
- max_iter: `15`
- reg_param: `0.1`
- rmse: `0.805119`
- elapsed_s: `205.6`

## 11. Health Checks and Operations

Check status:

```bash
./run.sh health
```

Stop all services:

```bash
./run.sh down
```

## 12. Troubleshooting

- `vite: command not found`:
  - Run `cd frontend && npm install` first.

- `Qdrant collection movie_content not found`:
  - Run `python3 data_processing_pipeline/indexer.py`.

- No CF recommendations for a user:
  - User may be cold-start (not in ALS factors). Interact with movies, or use search/agent flow.

- Backend starts but agent explanation is basic:
  - Set `OPENAI_API_KEY` in `.env` and restart backend.

- Kafka feed empty:
  - Ensure broker is up and actions (click/rate) are being triggered from frontend.

## 13. Notes on Reproducibility

- Random seed is fixed in training/evaluation scripts where applicable.
- `scripts/evaluate.py` writes one JSON report so results can be versioned and compared over time.
- For fair model comparison, keep `--users`, `--k`, and seed constant across runs.

## 14. Roadmap

- Wire NCF model into online serving path for side-by-side online evaluation.
- Add Qdrant/RAG-specific offline metrics (semantic recall@K with labeled test set).
- Add online CTR/session metrics aggregation pipeline from Kafka events.
