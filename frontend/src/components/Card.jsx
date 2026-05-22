import { formatGenres } from '../action'

const fallbackLabels = ['Stream', 'Movie', 'Cinema', 'Watch']

function Poster({ movie, compact = false }) {
  if (movie.poster) {
    return <img src={movie.poster} alt={`${movie.title} poster`} loading="lazy" />
  }

  const label = fallbackLabels[Math.abs(Number(movie.id ?? movie.movie_ref ?? 0)) % fallbackLabels.length]
  return (
    <div className={compact ? 'poster-placeholder compact' : 'poster-placeholder'}>
      <span>{label}</span>
    </div>
  )
}


function Card({ movie, onOpen }) {
  const year = movie.year && movie.year !== 'Unknown' ? movie.year : ''

  return (
    <button type="button" className="movie-card" onClick={() => onOpen(movie)}>
      <div className="poster-frame">
        <Poster movie={movie} />
      </div>
      <div className="movie-info">
        <h3>{movie.title}</h3>
        <p className="meta-line">
          {[year, formatGenres(movie.genres)].filter(Boolean).join(' • ')}
        </p>
      </div>
    </button>
  )
}

function ExtendedCard({ movie, query, onOpen }) {
  const year = movie.year && movie.year !== 'Unknown' ? movie.year : ''

  return (
    <button type="button" className="extended-card" onClick={() => onOpen(movie)}>
      <div className="extended-poster poster-frame">
        <Poster movie={movie} compact />
      </div>
      <div className="extended-copy">
        <h3>{highlightText(movie.title, query)}</h3>
        <p className="meta-line">
          {[year, formatGenres(movie.genres)].filter(Boolean).join(' • ')}
        </p>
        <p>{highlightText(movie.description || 'No description available.', query)}</p>
      </div>
    </button>
  )
}

function highlightText(value, query) {
  const text = String(value ?? '')
  const terms = String(query ?? '')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map(escapeRegExp)

  if (terms.length === 0) {
    return text
  }

  const matcher = new RegExp(`(${terms.join('|')})`, 'gi')

  return text.split(matcher).map((part, index) => {
    if (terms.some((term) => new RegExp(`^${term}$`, 'i').test(part))) {
      return <mark key={`${part}-${index}`}>{part}</mark>
    }
    return part
  })
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export { ExtendedCard }
export default Card
