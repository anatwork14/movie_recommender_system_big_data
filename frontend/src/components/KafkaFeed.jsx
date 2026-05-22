function KafkaFeed({ feed = [] }) {
  return (
    <aside className="sidebar" aria-label="Live Kafka feed">
      <h2>Live Kafka Feed</h2>
      <div className="feed-container">
        {feed.length > 0 ? (
          feed.map((item, index) => (
            <div className="feed-item" key={`${item.time}-${item.movie}-${index}`}>
              <span className={`type-${item.type}`}>{item.type}</span>
              <strong>Movie {item.movie}</strong>
              <small>
                User {item.user} | {item.detail}
              </small>
              <time>{formatTime(item.time)}</time>
            </div>
          ))
        ) : (
          <p>No events yet.</p>
        )}
      </div>
    </aside>
  )
}

function formatTime(value) {
  if (!value) {
    return 'Waiting'
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }

  return date.toLocaleString([], {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export default KafkaFeed
