import pandas as pd
import re
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

TRAILING_ARTICLE_RE = re.compile(r'^(?P<title>.+),\s*(?P<article>The|A|An)(?P<year>\s+\(\d{4}\))?\s*$')


def display_title(value):
    title = str(value or 'Unknown').strip()
    match = TRAILING_ARTICLE_RE.match(title)
    if not match:
        return title
    return f"{match.group('article')} {match.group('title')}{match.group('year') or ''}"

import os

# 1. Initialize Qdrant Client (Connecting to your docker container)
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
collection_name = "movie_content"

# Modern approach: Check if exists, then create (replaces deprecated recreate_collection)
if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name)

qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 2. Load the Local Embedding Model (Free, Fast, No API keys!)
print("Loading Embedding Model (this takes a few seconds)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Load and Clean the Data
print("Loading process_movie.csv...")
df = pd.read_csv('data/process_movie.csv')

# Fix the 12 missing years by filling them with 'Unknown'
df['year'] = df['year'].fillna('Unknown')
# Convert year to string so it doesn't print as 1995.0
df['year'] = df['year'].astype(str).str.replace('.0', '', regex=False)

# 4. Generate the Vector Database Payload
print("Generating Embeddings and Upserting to Qdrant...")
points = []
batch_size = 250 # Process in chunks to keep memory usage low

for index, row in df.iterrows():
    title = display_title(row['title'])
    # THE FIX: Create a rich, semantic sentence for the AI
    # Example: "Title: Toy Story (1995). Genres: Adventure, Animation. Description: Led by Woody..."
    rich_text = f"Title: {title} ({row['year']}). Genres: {row['genres']}. Description: {row['description']}"
    
    # Generate embedding locally
    vector = model.encode(rich_text).tolist()
    
    # Define what data we want to retrieve when a user searches
    metadata_payload = {
        "movie_ref": int(row['movieId']),
        "title": title,
        "genres": str(row['genres']),
        "year": str(row['year']),
        "description": str(row['description']),
        "poster_path": str(row['poster']),
    }
    
    # Qdrant requires a UUID for each record
    point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(row['movieId'])))
    
    points.append(PointStruct(
        id=point_uuid, 
        vector=vector, 
        payload=metadata_payload
    ))
    
    # Batch Upsert to Qdrant
    if len(points) >= batch_size:
        qdrant.upsert(collection_name=collection_name, points=points)
        points = []
        print(f"Upserted up to movie index {index} / {len(df)}")

# Don't forget the last partial batch
if points:
    qdrant.upsert(collection_name=collection_name, points=points)

print("SUCCESS! All 26,619 movies are now indexed and searchable in Qdrant.")
