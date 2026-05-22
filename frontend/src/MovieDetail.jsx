import { useEffect, useState } from 'react'
import { fetchMovie, sendRateAction, fetchAverage, fetchUserRating, formatGenres } from './action'
import Recommendation from './components/Recommendation'

function MovieDetail({ movieId, recommendations, onOpenMovie, onRated }) {
  const [movieState, setMovieState] = useState({
    movieId: null,
    movie: null,
    error: '',
  })

  const [ratingState, setRatingState] = useState({ movieId: null, value: 0 })
  const [ratingStatus, setRatingStatus] = useState({ movieId: null, text: '' })
  // const [hoverRating, setHoverRating] = useState(0)
  const [averageState, setAverageState] = useState({
    movieId: null,
    average: null,
    count: null,
  })
  const isCurrentMovie = movieState.movieId === movieId
  const movie = isCurrentMovie ? movieState.movie : null
  const rating = ratingState.movieId === movieId ? ratingState.value : 0
  const average = averageState.movieId === movieId ? averageState.average : null
  const ratingCount = averageState.movieId === movieId ? averageState.count : null
  const status = getDetailStatus(movieId, movieState, isCurrentMovie, ratingStatus)

  useEffect(() => {
    let isMounted = true

    fetchMovie(movieId)
      .then((data) => {
        if (isMounted) {
          setMovieState({
            movieId,
            movie: data.movie,
            error: '',
          })
        }
      })
      .catch(() => {
        if (isMounted) {
          setMovieState({
            movieId,
            movie: null,
            error: 'Movie details are unavailable.',
          })
        }
      })

    // fetch server-provided average rating (if available)
    fetchAverage(movieId)
      .then((data) => {
        if (isMounted) {
          setAverageState({
            movieId,
            average: data?.avg_rating == null ? null : Number(data.avg_rating),
            count: data?.rating_count == null ? null : Number(data.rating_count),
          })
        }
      })
      .catch(() => {
        // ignore - backend may not provide average/count
      })

    fetchUserRating(movieId)
      .then((data) => {
        if (isMounted) {
          setRatingState({
            movieId,
            value: data?.user_rating == null ? 0 : Number(data.user_rating),
          })
        }
      })
      .catch(() => {
        // ignore - user may not have rated this movie
      })

    return () => {
      isMounted = false
    }
  }, [movieId])

  async function handleSubmit(event) {
    event.preventDefault()
    if (!movie) {
      return
    }
    if (rating <= 0) {
      setRatingStatus({ movieId: movie.id, text: 'Choose a rating first.' })
      return
    }

    const ratingLabel = `${rating} star${rating === 1 ? '' : 's'}`
    const confirmed = window.confirm(`Send ${ratingLabel} for "${movie.title}"?`)
    if (!confirmed) {
      return
    }

    setRatingStatus({ movieId: movie.id, text: 'Sending rating...' })
    try {
      const data = await sendRateAction(movie.id, rating)
      setRatingStatus({ movieId: movie.id, text: `Rated as ${ratingLabel}.` })
      setAverageState({
        movieId: movie.id,
        average: data?.avg_rating == null ? null : Number(data.avg_rating),
        count: data?.rating_count == null ? null : Number(data.rating_count),
      })
      if (data?.user_rating != null) {
        setRatingState({ movieId: movie.id, value: Number(data.user_rating) })
      }
      onRated?.()
    } catch {
      setRatingStatus({
        movieId: movie.id,
        text: 'Could not send rating. Check the FastAPI backend.',
      })
    }
  }

  return (
    <div className="page-stack movie-detail">
      <title>Movie Detail | Let's Watch!</title>

      {movie ? (
        <article className="detail-layout">
          <div className="detail-poster">
            {movie.poster ? (
              <img src={movie.poster} alt={`${movie.title} poster`} />
            ) : (
              <div className="poster-placeholder">
                <span>Movie</span>
              </div>
            )}
          </div>

          <div className="detail-copy">
            <h1>{movie.title}</h1>
            <p className="meta-line movie-meta">
              {movie.year && <span>{movie.year}</span>}
              {movie.genres ? (
                <span className="movie-meta">
                  {formatGenres(movie.genres)
                    .split(',')
                    .map((g) => (
                      <span className="genre-tag" key={g.trim()}>{g.trim()}</span>
                    ))}
                </span>
              ) : null}
            </p>
            <p className="description">{movie.description || 'No description available.'}</p>

            <fieldset>
              <p className="info-block-title">Average Rating</p>
              {average == null ? (
                <div className="avg-count">No ratings yet</div>
              ) : (
                <div className="average-line">
                    <div className="avg-number">
                      {average.toFixed(1)}
                      <span className="secondary"> / 5</span>
                    </div>
                    <span className="avg-count">({ratingCount} users)</span>
                </div>
              )}
            </fieldset>
          </div>
        </article>
      ) : (
        <section className="empty-state">{status}</section>
      )}

      <div className="rating-section">
        <hr className="separator" />
        {/* Your Rating Form */}
        <form className="rating-form" onSubmit={handleSubmit}>
          <fieldset>
            <p className="info-block-title">Rate this movie</p>
            
            <div className="star-options">
              {[5, 4, 3, 2, 1].map((value) => (
                <label
                  key={value}
                  className={value <= rating ? 'selected' : ''}
                >
                  <input
                    type="radio"
                    name="rating"
                    value={value}
                    checked={rating === value}
                    onChange={() => setRatingState({ movieId, value })}
                  />
                  <span>★</span>
                </label>
              ))}
            </div> 

            {status && <p className="status-line">{status}</p>}
        
          </fieldset>

          <button type="submit">Submit rating</button>
        </form>
      </div>

      <Recommendation movies={recommendations} onOpen={onOpenMovie} />
    </div>
  )
}

function getDetailStatus(movieId, movieState, isCurrentMovie, ratingStatus) {
  if (ratingStatus.movieId === movieId && ratingStatus.text) {
    return ratingStatus.text
  }
  if (!isCurrentMovie) {
    return 'Loading movie...'
  }
  if (movieState.error) {
    return movieState.error
  }
  if (!movieId) {
    return 'Movie details are unavailable.'
  }
  return ''
}

export default MovieDetail
