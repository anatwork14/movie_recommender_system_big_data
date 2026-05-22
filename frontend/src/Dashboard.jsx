import { useEffect, useState } from 'react'
import { fetchTrending } from './action'
import AgentChat from './components/AgentChat'
import Recommendation, { MovieRow } from './components/Recommendation'

const placeholderTrending = [
  {
    id: 1,
    title: 'Toy Story',
    genres: 'Adventure, Animation, Comedy',
    year: '1995',
    description: 'A cowboy doll feels threatened when a new space ranger toy joins the room.',
  },
  {
    id: 318,
    title: 'The Shawshank Redemption',
    genres: 'Crime, Drama',
    year: '1994',
    description: 'Two imprisoned men bond over years of survival and quiet hope.',
  },
  {
    id: 356,
    title: 'Forrest Gump',
    genres: 'Comedy, Drama, Romance',
    year: '1994',
    description: 'A kind-hearted man crosses paths with defining moments in American history.',
  },
  {
    id: 593,
    title: 'The Silence of the Lambs',
    genres: 'Crime, Horror, Thriller',
    year: '1991',
    description: 'An FBI trainee seeks insight from a brilliant imprisoned killer.',
  },
  {
    id: 2571,
    title: 'The Matrix',
    genres: 'Action, Sci-Fi, Thriller',
    year: '1999',
    description: 'A hacker discovers the world he knows is a simulation.',
  },
  {
    id: 260,
    title: 'Star Wars',
    genres: 'Action, Adventure, Sci-Fi',
    year: '1977',
    description: 'A farm boy joins a rebellion against a galaxy-spanning empire.',
  },
]

function Dashboard({ recommendations, onOpenMovie, userId }) {
  const [trending, setTrending] = useState(placeholderTrending)

  useEffect(() => {
    let isMounted = true

    fetchTrending()
      .then((data) => {
        if (isMounted && data.movies?.length > 0) {
          setTrending(data.movies)
        }
      })
      .catch(() => {
        setTrending(placeholderTrending)
      })

    return () => {
      isMounted = false
    }
  }, [])

  return (
    <div className="page-stack">
      <section className="welcome-panel">
        <p className="eyebrow">Hybrid AI · CF + TF-IDF + RAG</p>
        <h1>Discover your next favourite film</h1>
        <p className="hero-sub">
          Hybrid recommendation pipeline: we collect behavior signals (clicks, ratings,
          searches), build user-item representations, generate candidates from collaborative
          filtering and content-based TF-IDF retrieval, then fuse/rank results before showing
          top picks. Use the toggle in "Recommended for You" to inspect each source.
        </p>
      </section>

      {/* AI Agent Chat Panel */}
      <section className="agent-panel">
        <AgentChat userId={userId} onOpenMovie={onOpenMovie} />
      </section>

      <MovieRow
        title="Top Trending"
        movies={trending}
        emptyText="Trending will appear here after the backend job is connected."
        onOpen={onOpenMovie}
      />

      <Recommendation movies={recommendations} onOpen={onOpenMovie} />
    </div>
  )
}

export default Dashboard
