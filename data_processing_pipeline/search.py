from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import os

# 1. Connect to your local Qdrant
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

# 2. Load the same lightweight model to encode your search query
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Type any natural language query here!
# Let's try finding something highly specific
query_text = "A movie about dinosaurs and a theme park gone wrong"

print(f"\nSearching for: '{query_text}'\n")

# Encode the search query into a vector
query_vector = model.encode(query_text).tolist()

# 4. Search the vector database using the new Query API (v1.17+)
search_response = qdrant.query_points(
    collection_name="movie_content",
    query=query_vector,
    limit=5  # Get the top 5 closest matches
)
search_results = search_response.points

# 5. Print the results
for hit in search_results:
    payload = hit.payload
    print(f"Title: {payload['title']} ({payload['year']})")
    print(f"Genres: {payload['genres']}")
    print(f"Similarity Score: {hit.score:.4f}")
    print("-" * 40)