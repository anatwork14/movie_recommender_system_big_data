# Big Data Recommender System 🎬

A real-time, hybrid recommendation system leveraging **Kafka** for event streaming, **Qdrant** for semantic vector search, and **PySpark** for scalable collaborative filtering.

## 🚀 Project Architecture
The system follows a modern decoupled architecture for processing and recommending content:

1.  **Frontend (React/Vite UI)**: An interactive dashboard where users can search, click, and rate movies.
2.  **Streaming Layer (Kafka)**: User interactions (clicks, ratings) are pushed to specific topics in real-time.
3.  **Real-time Logic (Content-Based)**: A background consumer retrieves the semantic vector of the interacted movie from Qdrant and finds similar content instantly.
4.  **Batch Layer (Collaborative Filtering)**: A PySpark ALS job trains user/movie latent factors from historical ratings.
5.  **Vector DB (Qdrant)**: Stores embeddings for 26,000+ movies, enabling lightning-fast similarity queries.

---

## 🛠 Tech Stack
- **Languages**: Python, HTML/JS
- **Streaming**: Apache Kafka (via Confluent-Kafka)
- **Vector Database**: Qdrant
- **ML Models**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Big Data**: Apache Spark (PySpark)
- **Web Frameworks**: FastAPI backend, React/Vite frontend, legacy Flask dashboard
- **Containerization**: Docker & Docker Compose

---

## 📊 System Components

### 1. Data Pipeline
- **`indexer.py`**: Processes the MovieLens dataset, generates semantic embeddings for movie descriptions, and upserts them to Qdrant with rich metadata.
- **`search.py`**: Provides utility functions for semantic search across the collection.

### 2. Kafka Streaming
- **`producer.py`**: A robust wrapper for sending user interaction JSON payloads to Kafka.
- **`web_dashboard.py`**: Hosts the web server and runs a background thread to consume Kafka events and trigger recommendations.

### 3. Collaborative Filtering Model
- **`spark_jobs/collaborative_filtering.py`**: Full PySpark ALS trainer. It reads `data/process_movie_rating.csv` and writes `models/als/user_factors.parquet`, `models/als/item_factors.parquet`, and `models/als/training_metrics.txt`.
- **`src/recsys/engines/cf_engine.py`**: Runtime recommendation engine. It loads ALS factors and ranks movies for a user with a dot product.
- **`spark_jobs/ncf_training.py`**: Experimental Neural Collaborative Filtering trainer. It saves a PyTorch checkpoint, but the backend does not serve this model yet.

### 4. Per-User Recommendations
The React navbar lets you switch the current demo user ID. This user ID is passed to the backend on click, rating, feed, and agent recommendation requests.

For a user that exists in `data/process_movie_rating.csv`, the ALS collaborative filtering model can return personalized candidates based on that user's historical rating pattern. For example, user `1500` and user `1337` can receive different CF recommendations because they have different learned user vectors.

For a new or unknown user ID, collaborative filtering has no stored user vector, so the backend falls back to content-based recommendations from TF-IDF and Qdrant RAG. In the UI, a brand-new user may initially show no recommended films until they click or search for movies.

When a movie is clicked, the backend recommends films using a hybrid of:
- the current user's CF taste profile,
- the clicked movie's title, genres, and description,
- semantic similarity from Qdrant,
- lexical similarity from TF-IDF.

The frontend keeps a small in-memory recommendation cache per user during the current browser session. Switching back to a previous user restores their latest recommendations, but refreshing the browser clears this cache.

---

## 📥 Input & Output
- **Input**:
    - `data/process_movie.csv`: Movie metadata (Titles, Genres, Overviews).
    - User Interactions: Real-time JSON streams (User ID, Movie ID, Timestamp, Action Type).
- **Output**:
    - **Real-time**: "Because you viewed X, you might like Y" (Semantic similarity).
    - **Search**: Top-K movies matching a natural language query.
    - **Batch CF**: Top-K recommendations based on global user/movie rating patterns.

---

## 📏 Metrics for Success
- **Relevance**: 
    - *Content-Based*: Cosine similarity score between vectors.
    - *Collaborative Filtering*: Root Mean Squared Error (RMSE) on rating predictions.
- **Performance**: 
    - *Latency*: UI update speed after a click (Target: < 500ms).
    - *Throughput*: Number of Kafka messages processed per second.
- **User Engagement**: Click-Through Rate (CTR) on suggested movies.

---

## ⚙️ Installation & Setup
1. **Infrastructure**:
   ```bash
   docker-compose up -d
   ```
2. **Environment**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Indexing**:
   ```bash
   python data_processing_pipeline/indexer.py
   ```
4. **Train the full ALS collaborative filtering model**:
   ```bash
   python spark_jobs/collaborative_filtering.py --sample 1.0
   ```
   For a faster test run:
   ```bash
   python spark_jobs/collaborative_filtering.py --sample 0.1
   ```
5. **Run FastAPI backend**:
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 5000
   ```
   On startup, the backend loads `models/als/user_factors.parquet` and `models/als/item_factors.parquet` when they exist. If they do not exist, it falls back to `models/als_factors.pkl`, or trains a small 3% demo model from `data/process_movie_rating.csv`.
6. **Run legacy Flask dashboard**:
   ```bash
   python kafka_streaming/web_dashboard.py
   ```
7. **Run frontend**:
    open a new terminal
    ```bash
   cd frontend
   npm run dev -- --host
   ```
   Some urls may not work in WSL so try opening all 3 of them

---

## 🔮 Roadmap
- [x] Kafka Producer/Consumer Integration.
- [x] Semantic Search (Natural Language).
- [x] Real-time Content-Based Feedback Loop.
- [x] Full ALS training in PySpark.
- [x] Hybridized results combining collaborative, lexical, and semantic recommendations.
- [ ] Serve the experimental NCF checkpoint in the backend.
