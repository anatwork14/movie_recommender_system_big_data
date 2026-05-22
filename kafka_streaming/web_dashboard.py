import json
import os
import re
import sys
import threading

# Add src to path for development if not installed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from confluent_kafka import Consumer
from flask import Flask, jsonify, render_template, request
from recsys import InteractionProducer, MovieVectorDB

app = Flask(__name__)

POSTER_BASE_URL = 'https://image.tmdb.org/t/p/w500'
USER_ID = 1337
TRAILING_ARTICLE_RE = re.compile(r'^(?P<title>.+),\s*(?P<article>The|A|An)(?P<year>\s+\(\d{4}\))?\s*$')

live_feed = []
recommendations = []
recommendation_movies = []

producer = InteractionProducer()
db = MovieVectorDB()

TRENDING_PLACEHOLDERS = [
    {
        "id": 1,
        "title": "Toy Story",
        "genres": "Adventure, Animation, Comedy",
        "year": "1995",
        "description": "A cowboy doll feels threatened when a new space ranger toy joins the room.",
        "poster": ""
    },
    {
        "id": 318,
        "title": "The Shawshank Redemption",
        "genres": "Crime, Drama",
        "year": "1994",
        "description": "Two imprisoned men bond over years of survival and quiet hope.",
        "poster": ""
    },
    {
        "id": 356,
        "title": "Forrest Gump",
        "genres": "Comedy, Drama, Romance",
        "year": "1994",
        "description": "A kind-hearted man crosses paths with defining moments in American history.",
        "poster": ""
    },
    {
        "id": 593,
        "title": "The Silence of the Lambs",
        "genres": "Crime, Horror, Thriller",
        "year": "1991",
        "description": "An FBI trainee seeks insight from a brilliant imprisoned killer.",
        "poster": ""
    },
    {
        "id": 2571,
        "title": "The Matrix",
        "genres": "Action, Sci-Fi, Thriller",
        "year": "1999",
        "description": "A hacker discovers the world he knows is a simulation.",
        "poster": ""
    },
    {
        "id": 260,
        "title": "Star Wars",
        "genres": "Action, Adventure, Sci-Fi",
        "year": "1977",
        "description": "A farm boy joins a rebellion against a galaxy-spanning empire.",
        "poster": ""
    },
]


def poster_url(value):
    if not value:
        return ''

    value = str(value)
    if value.lower() == 'nan':
        return ''
    if value.startswith('/'):
        return f'{POSTER_BASE_URL}{value}'
    return value


def display_title(value):
    title = str(value or 'Unknown').strip()
    match = TRAILING_ARTICLE_RE.match(title)
    if not match:
        return title
    return f"{match.group('article')} {match.group('title')}{match.group('year') or ''}"


def movie_from_payload(payload, score=None):
    movie = {
        "id": int(payload.get('movie_ref', payload.get('id', 0))),
        "title": display_title(payload.get('title', 'Unknown')),
        "genres": str(payload.get('genres', '')),
        "year": str(payload.get('year', 'Unknown')),
        "description": str(payload.get('description', '')),
        "poster": poster_url(str(payload.get('poster_path', ''))),
    }
    if score is not None:
        movie["score"] = float(score)
    return movie


def movies_from_hits(hits):
    movies = []
    for hit in hits:
        movies.append(movie_from_payload(hit.payload, getattr(hit, 'score', None)))
    return movies


def refresh_recommendations(movie_id, limit=20):
    global recommendations, recommendation_movies
    try:
        hits = db.get_recommendations(movie_id, limit=limit)
        recommendation_movies = movies_from_hits(hits)
        recommendations = [f"Library Suggests: {movie['title']}" for movie in recommendation_movies]
    except Exception as exc:
        print(f"[-] Recommendation error: {exc}")
        recommendation_movies = []
        recommendations = []


def kafka_consumer_thread():
    consumer_config = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'web-dashboard-group',
        'auto.offset.reset': 'latest'
    }
    consumer = Consumer(consumer_config)
    consumer.subscribe(['user-clicks', 'user-ratings'])

    print("[*] Library-powered Consumer thread started.")

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            continue

        try:
            data = json.loads(msg.value().decode('utf-8'))
            topic = msg.topic()
            movie_id = int(data['movie_id'])

            event_msg = {
                "type": "Click" if topic == 'user-clicks' else "Rating",
                "user": data['user_id'],
                "movie": movie_id,
                "detail": data.get('rating', 'Clicked'),
                "time": data['timestamp']
            }
            live_feed.insert(0, event_msg)
            if len(live_feed) > 20:
                live_feed.pop()

            if topic in {'user-clicks', 'user-ratings'}:
                refresh_recommendations(movie_id)

        except Exception as exc:
            print(f"[-] Consumer error: {exc}")


threading.Thread(target=kafka_consumer_thread, daemon=True).start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/click/<int:movie_id>')
def click_movie(movie_id):
    # Accept an optional user_id from the frontend (query param or header) for demo purposes
    try:
        user_id = int(request.args.get('user_id') or request.headers.get('X-User-Id') or USER_ID)
    except Exception:
        user_id = USER_ID
    producer.track_click(user_id=user_id, movie_id=movie_id)
    refresh_recommendations(movie_id)
    return jsonify({
        "status": "success",
        "recommendations": recommendations,
        "recommendation_movies": recommendation_movies
    })


@app.route('/api/rate/<int:movie_id>/<float:score>')
def rate_movie(movie_id, score):
    try:
        user_id = int(request.args.get('user_id') or request.headers.get('X-User-Id') or USER_ID)
    except Exception:
        user_id = USER_ID
    producer.track_rating(user_id=user_id, movie_id=movie_id, rating=score)
    refresh_recommendations(movie_id)
    return jsonify({
        "status": "success",
        "recommendations": recommendations,
        "recommendation_movies": recommendation_movies
    })


@app.route('/api/feed')
def get_feed():
    return jsonify({
        "feed": live_feed,
        "recommendations": recommendations,
        "recommendation_movies": recommendation_movies
    })


@app.route('/api/search')
def search():
    query = request.args.get('q', '').strip()
    try:
        limit = min(int(request.args.get('limit', 50)), 100)
    except ValueError:
        limit = 50
    if not query:
        return jsonify({"results": []})

    try:
        hits = db.search(query, limit=limit)
        return jsonify({"results": movies_from_hits(hits)})
    except Exception as exc:
        print(f"[-] Search error: {exc}")
        return jsonify({"results": [], "error": "Search unavailable"}), 503


@app.route('/api/movie/<int:movie_id>')
def get_movie(movie_id):
    try:
        point = db.get_movie(movie_id)
        if not point:
            return jsonify({"error": "Movie not found"}), 404
        return jsonify({"movie": movie_from_payload(point.payload)})
    except Exception as exc:
        print(f"[-] Movie lookup error: {exc}")
        return jsonify({"error": "Movie lookup unavailable"}), 503


# TODO: GET TOP TRENDING MOVIES
@app.route('/api/trending')
def trending():
    return jsonify({"movies": TRENDING_PLACEHOLDERS})

# TODO: GET AVERAGE RATING, RATING COUNT FOR A MOVIE
@app.route('/api/average_rating/<int:movie_id>')
def average_rating(movie_id):
    return jsonify({
        "movie_id": movie_id, "avg_rating": 4.2,
        "rating_count": 1234
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
