import re
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

TRAILING_ARTICLE_RE = re.compile(r'^(?P<title>.+),\s*(?P<article>The|A|An)(?P<year>\s+\(\d{4}\))?\s*$')


def display_title(value):
    title = str(value or 'Unknown').strip()
    match = TRAILING_ARTICLE_RE.match(title)
    if not match:
        return title
    return f"{match.group('article')} {match.group('title')}{match.group('year') or ''}"

class MovieVectorDB:
    def __init__(self, host="localhost", port=6333, collection_name="movie_content"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def init_collection(self, vector_size=384):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def generate_uuid(self, movie_id):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(int(movie_id))))

    def upsert_movies(self, df, batch_size=250):
        points = []
        for index, row in df.iterrows():
            title = display_title(row['title'])
            rich_text = f"Title: {title} ({row.get('year', 'Unknown')}). Genres: {row['genres']}. Description: {row['description']}"
            vector = self.model.encode(rich_text).tolist()
            poster = str(row.get('poster', row.get('poster_url', row.get('poster_path', ''))))
            
            payload = {
                "movie_ref": int(row['movieId']),
                "title": title,
                "genres": str(row['genres']),
                "year": str(row.get('year', 'Unknown')),
                "description": str(row['description']),
                "poster": poster,
                "poster_url": poster
            }
            
            points.append(PointStruct(
                id=self.generate_uuid(row['movieId']),
                vector=vector,
                payload=payload
            ))
            
            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_text, limit=20):
        vector = self.model.encode(query_text).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit
        )
        return results.points

    def get_movie(self, movie_id):
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self.generate_uuid(movie_id)],
            with_payload=True
        )
        return points[0] if points else None

    def get_recommendations(self, movie_id, limit=20):
        point_id = self.generate_uuid(movie_id)
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=True
        )
        if not points:
            return []
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=points[0].vector,
            limit=limit + 1
        )
        # Filter out the source movie
        return [hit for hit in search_result.points if hit.payload.get('movie_ref') != int(movie_id)][:limit]
