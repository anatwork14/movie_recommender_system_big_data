import { useState, useRef, useEffect } from 'react'

/**
 * AgentChat — Conversational AI Recommendation Panel
 *
 * Interfaces with POST /api/recommend to run the full AI Agent pipeline:
 *   1. Intent detection (CF / TF-IDF / RAG)
 *   2. Parallel tool dispatch
 *   3. RRF fusion
 *   4. Natural-language explanation
 *
 * Features:
 *   - Streaming-style token reveal animation
 *   - Shows provenance tags (CF / TF-IDF / RAG) per movie
 *   - Collapsible source badges for explainability
 */

const EXAMPLE_PROMPTS = [
  "Find me dark sci-fi movies with mind-bending plots similar to Inception",
  "Action movies directed by Christopher Nolan",
  "Emotional drama films that made everyone cry",
  "Animated movies suitable for adults, not just kids",
  "Thriller movies with unexpected endings and moral dilemmas",
]

function AgentChat({ userId, onOpenMovie }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: "Hi! I'm your AI Movie Agent 🎬 Tell me what kind of movies you're in the mood for — describe a vibe, mention an actor, or just say what you felt after your last great watch.",
      movies: [],
      provenance: null,
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleSend(text) {
    const query = (text || input).trim()
    if (!query || loading) return

    const userMsg = { role: 'user', text: query }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const resp = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          user_id: userId ? Number(userId) : null,
          top_k: 8,
        }),
      })

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()

      const assistantMsg = {
        role: 'assistant',
        text: data.explanation || 'Here are my recommendations:',
        movies: data.recommendation_movies || [],
        provenance: data.provenance || null,
        intent: data.intent || null,
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: '⚠️ Could not connect to the recommendation engine. Make sure the FastAPI backend is running.',
          movies: [],
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="agent-chat">
      <div className="chat-header">
        <span className="chat-icon">🤖</span>
        <div>
          <strong>AI Movie Agent</strong>
          <small>CF + TF-IDF + RAG · RRF Fusion</small>
        </div>
        <span className="chat-status online">LIVE</span>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <ChatMessage key={i} msg={msg} onOpenMovie={onOpenMovie} />
        ))}

        {loading && (
          <div className="chat-bubble assistant">
            <div className="typing-indicator">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Example prompts */}
      {messages.length === 1 && (
        <div className="example-prompts">
          {EXAMPLE_PROMPTS.map((p) => (
            <button
              key={p}
              type="button"
              className="prompt-chip"
              onClick={() => handleSend(p)}
            >
              {p}
            </button>
          ))}
        </div>
      )}

      <div className="chat-input-bar">
        <textarea
          className="chat-input"
          rows={2}
          placeholder="Describe the kind of movie you want to watch…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          type="button"
          className="chat-send"
          onClick={() => handleSend()}
          disabled={loading || !input.trim()}
          aria-label="Send message"
        >
          ➤
        </button>
      </div>
    </div>
  )
}

function ChatMessage({ msg, onOpenMovie }) {
  return (
    <div className={`chat-bubble ${msg.role}`}>
      {msg.role === 'assistant' && (
        <span className="bubble-avatar">
          {msg.intent ? '🎯' : '🎬'}
        </span>
      )}
      <div className="bubble-body">
        <p className="bubble-text">{msg.text}</p>

        {/* Intent badges */}
        {msg.intent && (
          <div className="intent-badges">
            {msg.intent.use_cf && <span className="badge badge-cf">CF</span>}
            {msg.intent.use_tfidf && <span className="badge badge-tfidf">TF-IDF</span>}
            {msg.intent.use_rag && <span className="badge badge-rag">RAG</span>}
            {msg.provenance && (
              <span className="badge badge-count">
                {msg.provenance.merged_count} merged → top {msg.movies.length}
              </span>
            )}
          </div>
        )}

        {/* Movie cards */}
        {msg.movies?.length > 0 && (
          <div className="agent-movie-list">
            {msg.movies.map((m) => (
              <AgentMovieCard key={m.id} movie={m} onOpen={onOpenMovie} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function AgentMovieCard({ movie, onOpen }) {
  const posterSrc = movie.poster && !movie.poster.includes('undefined')
    ? movie.poster
    : null

  return (
    <button
      type="button"
      className="agent-movie-card"
      onClick={() => onOpen(movie)}
    >
      <div className="agent-poster">
        {posterSrc ? (
          <img src={posterSrc} alt={`${movie.title} poster`} loading="lazy" />
        ) : (
          <div className="agent-poster-placeholder">🎬</div>
        )}
      </div>
      <div className="agent-movie-info">
        <strong className="agent-title">{movie.title}</strong>
        <span className="agent-meta">{[movie.year, movie.genres?.split(',')[0]?.trim()].filter(Boolean).join(' · ')}</span>
        <p className="agent-desc">{movie.description?.slice(0, 100)}…</p>

        {/* Source provenance */}
        {movie.sources?.length > 0 && (
          <div className="source-tags">
            {movie.sources.map((s) => (
              <span key={s} className={`source-tag source-${s.replace(/[-\s]/g, '').toLowerCase()}`}>
                {s}
              </span>
            ))}
          </div>
        )}
      </div>
    </button>
  )
}

export default AgentChat
