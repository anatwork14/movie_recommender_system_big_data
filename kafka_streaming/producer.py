import json
from datetime import datetime
from confluent_kafka import Producer

class InteractionProducer:
    """
    Handles streaming of user interactions (clicks and ratings) to Kafka topics.
    """
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'recsys-web-ui'
        }
        self.producer = Producer(self.config)
        self.topics = {
            'click': 'user-clicks',
            'rating': 'user-ratings'
        }

    def _delivery_report(self, err, msg):
        """Called once for each message produced to indicate delivery result."""
        if err is not None:
            print(f"[-] Message delivery failed: {err}")
        else:
            print(f"[+] Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

    def track_click(self, user_id, movie_id):
        """
        Produces a 'click' event to the user-clicks topic.
        """
        payload = {
            'user_id': user_id,
            'movie_id': movie_id,
            'event_type': 'click',
            'timestamp': datetime.utcnow().isoformat()
        }
        self._send_to_kafka(self.topics['click'], payload)

    def track_rating(self, user_id, movie_id, rating):
        """
        Produces a 'rating' event to the user-ratings topic.
        """
        payload = {
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': float(rating),
            'event_type': 'rating',
            'timestamp': datetime.utcnow().isoformat()
        }
        self._send_to_kafka(self.topics['rating'], payload)

    def _send_to_kafka(self, topic, payload):
        """Helper to send JSON payload to Kafka."""
        try:
            self.producer.produce(
                topic, 
                key=str(payload['user_id']), 
                value=json.dumps(payload).encode('utf-8'),
                callback=self._delivery_report
            )
            # Poll to trigger the callback
            self.producer.poll(0)
        except Exception as e:
            print(f"[-] Error producing to Kafka: {e}")

    def flush(self, timeout=None):
        """Wait for all outstanding messages to be delivered."""
        self.producer.flush(timeout)

if __name__ == "__main__":
    # Quick Test Simulation
    p = InteractionProducer()
    print("--- Simulating Interaction ---")
    p.track_click(user_id=101, movie_id=1)  # Toy Story click
    p.track_rating(user_id=101, movie_id=1, rating=5.0) # Toy Story rating
    p.flush()
    print("--- Test Complete ---")
