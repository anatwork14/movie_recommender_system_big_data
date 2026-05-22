async function fetchJson(url, options = {}) {
  const response = await fetch(url, options)
  if (!response.ok) {
    throw new Error(`Server responded with status ${response.status}`)
  }
  return response.json()
}

function readCurrentUserId() {
  try {
    const raw = localStorage.getItem('currentUser')
    if (!raw) return '1337'
    const parsed = JSON.parse(raw)
    return String(parsed?.user_id ?? '1337')
  } catch {
    return '1337'
  }
}

async function sendClickAction(movieId) {
  const userId = readCurrentUserId()
  return fetchJson(`/api/click/${movieId}?user_id=${encodeURIComponent(userId)}`)
}

async function sendRateAction(movieId, rating) {
  const userId = readCurrentUserId()
  return fetchJson(`/api/rate/${movieId}/${Number(rating).toFixed(1)}?user_id=${encodeURIComponent(userId)}`)
}

async function fetchUserRating(movieId) {
  const userId = readCurrentUserId()
  return fetchJson(`/api/user_rating/${movieId}?user_id=${encodeURIComponent(userId)}`)
}

async function searchMovies(query, limit = 50) {
  const params = new URLSearchParams({ q: query, limit: String(limit) })
  return fetchJson(`/api/search?${params.toString()}`)
}

async function fetchMovie(movieId) {
  return fetchJson(`/api/movie/${movieId}`)
}

async function fetchAverage(movieId) {
  return fetchJson(`/api/average_rating/${movieId}`)
}

async function fetchFeed() {
  const userId = readCurrentUserId()
  const params = new URLSearchParams({ user_id: userId })
  return fetchJson(`/api/feed?${params.toString()}`)
}

async function fetchTrending() {
  return fetchJson('/api/trending')
}

function formatGenres(genres) {
  if (!genres && genres !== '') return genres
  // Ensure it's a string and normalize spacing after commas
  return String(genres).replace(/\s*,\s*/g, ', ')
}


export {
  fetchFeed,
  fetchJson,
  fetchMovie,
  fetchTrending,
  formatGenres,
  searchMovies,
  sendClickAction,
  sendRateAction,
  fetchUserRating,
  fetchAverage
}
