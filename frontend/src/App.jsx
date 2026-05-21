import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchFeed, sendClickAction } from './action'
import Dashboard from './Dashboard'
import KafkaFeed from './components/KafkaFeed'
import Nav from './components/Nav'
import MovieDetail from './MovieDetail'
import SearchPage from './SearchPage'
import './style/dashboard.css'

function App() {
  const [route, setRoute] = useState(() => readRoute())
  const [feed, setFeed] = useState([])
  const [userId, setUserId] = useState(() => readCurrentUserId())
  const [recommendations, setRecommendations] = useState([])
  const [recommendationCache, setRecommendationCache] = useState({})

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
    const initialRefreshId = window.setTimeout(refreshFeed, 0)
    const intervalId = window.setInterval(refreshFeed, 1000)
    return () => {
      window.clearTimeout(initialRefreshId)
      window.clearInterval(intervalId)
    }
  }, [refreshFeed])

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

  const handleUserChange = useCallback(
    (nextUserId) => {
      const normalized = String(nextUserId).trim()
      if (!normalized || normalized === userId) {
        return
      }

      setRecommendationCache((cache) => ({
        ...cache,
        [userId]: recommendations,
      }))

      try {
        localStorage.setItem('currentUserId', normalized)
      } catch {}

      setUserId(normalized)
      setRecommendations(recommendationCache[normalized] ?? [])
      navigate('/')
    },
    [navigate, recommendationCache, recommendations, userId],
  )

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
        userId={userId}
        onNavigate={navigate}
        onSearch={handleSearch}
        onUserChange={handleUserChange}
      />
      <div className="app-layout">
        <main className="main-content">{page}</main>
        <KafkaFeed feed={feed} />
      </div>
    </div>
  )
}

function readCurrentUserId() {
  try {
    return localStorage.getItem('currentUserId') || '1337'
  } catch {
    return '1337'
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
