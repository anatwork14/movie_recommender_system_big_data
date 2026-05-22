import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchFeed, fetchJson, sendClickAction } from './action'
import Dashboard from './Dashboard'
import KafkaFeed from './components/KafkaFeed'
import Nav from './components/Nav'
import MovieDetail from './MovieDetail'
import SearchPage from './SearchPage'
import './style/dashboard.css'

function App() {
  const [route, setRoute] = useState(() => readRoute())
  const [feed, setFeed] = useState([])
  const [currentUser, setCurrentUser] = useState(() => readCurrentUser())
  const userId = String(currentUser?.user_id ?? '')
  const [recommendations, setRecommendations] = useState([])
  const [recommendationCache, setRecommendationCache] = useState({})
  const [accounts, setAccounts] = useState([])

  const refreshFeed = useCallback(async () => {
    try {
      const data = await fetchFeed()
      setFeed(data.feed ?? [])
      const movies = movieListFromResponse(data)
      setRecommendations(movies)
      setRecommendationCache((cache) => ({ ...cache, [userId]: movies }))
    } catch {
      setFeed([])
    }
  }, [userId])

  useEffect(() => {
    fetchJson('/api/auth/accounts').then((d) => setAccounts(d.accounts ?? [])).catch(() => {})
  }, [])

  useEffect(() => {
    if (!currentUser) return
    const initialRefreshId = window.setTimeout(refreshFeed, 0)
    const intervalId = window.setInterval(refreshFeed, 1000)
    return () => {
      window.clearTimeout(initialRefreshId)
      window.clearInterval(intervalId)
    }
  }, [currentUser, refreshFeed])

  const handleLogin = useCallback(async (username, password) => {
    const data = await fetchJson('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    const user = data.user
    try {
      localStorage.setItem('currentUser', JSON.stringify(user))
    } catch {}
    setCurrentUser(user)
    setRecommendations(recommendationCache[String(user.user_id)] ?? [])
  }, [recommendationCache])

  useEffect(() => {
    function handlePopState() {
      setRoute(readRoute())
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  const navigate = useCallback((path) => {
    window.history.pushState({}, '', path)
    setRoute(readRoute())
  }, [])

  const handleSearch = useCallback(
    (query) => {
      navigate(`/search?q=${encodeURIComponent(query)}`)
    },
    [navigate],
  )

  const handleLogout = useCallback(() => {
    try {
      localStorage.removeItem('currentUser')
    } catch {}
    setCurrentUser(null)
    setRecommendations([])
    navigate('/')
  }, [navigate])

  const handleOpenMovie = useCallback(
    (movie) => {
      if (!movie?.id) {
        return
      }

      sendClickAction(movie.id)
        .then((data) => {
          const movies = movieListFromResponse(data)
          setRecommendations(movies)
          setRecommendationCache((cache) => ({ ...cache, [userId]: movies }))
          refreshFeed()
        })
        .catch(() => {
          refreshFeed()
        })

      navigate(`/movie/${movie.id}`)
    },
    [navigate, refreshFeed, userId],
  )

  const page = useMemo(() => {
    if (route.name === 'search') {
      return (
        <SearchPage
          query={route.query}
          recommendations={recommendations}
          onOpenMovie={handleOpenMovie}
        />
      )
    }

    if (route.name === 'movie') {
      return (
        <MovieDetail
          movieId={route.movieId}
          recommendations={recommendations}
          onOpenMovie={handleOpenMovie}
          onRated={refreshFeed}
        />
      )
    }

    return <Dashboard recommendations={recommendations} onOpenMovie={handleOpenMovie} userId={userId} />
  }, [handleOpenMovie, recommendations, refreshFeed, route, userId])

  return (
    <div className="app-shell">
      <Nav
        key={`${route.name}-${route.query ?? ''}`}
        initialQuery={route.query ?? ''}
        user={currentUser}
        onNavigate={navigate}
        onSearch={handleSearch}
        accounts={accounts}
        onLogin={handleLogin}
        onLogout={handleLogout}
      />
      <div className="app-layout">
        <main className="main-content">{page}</main>
        <KafkaFeed feed={feed} />
      </div>
    </div>
  )
}

function readCurrentUser() {
  try {
    const raw = localStorage.getItem('currentUser')
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

function readRoute() {
  const { pathname, search } = window.location
  const params = new URLSearchParams(search)

  if (pathname === '/search') {
    return {
      name: 'search',
      query: params.get('q') ?? '',
    }
  }

  const movieMatch = pathname.match(/^\/movie\/(\d+)$/)
  if (movieMatch) {
    return {
      name: 'movie',
      movieId: Number(movieMatch[1]),
      query: '',
    }
  }

  return {
    name: 'dashboard',
    query: '',
  }
}

function movieListFromResponse(data) {
  if (Array.isArray(data?.recommendation_movies)) {
    return data.recommendation_movies
  }
  return []
}

export default App
