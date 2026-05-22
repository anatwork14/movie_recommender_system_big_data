import { useMemo, useState } from 'react'
import Card from './Card'

function MovieRow({ title, movies = [], emptyText, onOpen }) {
  const visibleMovies = movies.slice(0, 20)

  return (
    <section className="movie-row" aria-label={title}>
      <div className="row-heading">
        <h2>{title}</h2>
        {visibleMovies.length > 0 && <span>{visibleMovies.length} films</span>}
      </div>

      {visibleMovies.length > 0 ? (
        <div className="scroll-row">
          {visibleMovies.map((movie) => (
            <Card key={movie.id} movie={movie} onOpen={onOpen} />
          ))}
        </div>
      ) : (
        <div className="empty-row">{emptyText}</div>
      )}
    </section>
  )
}

function Recommendation({ movies = [], onOpen }) {
  const [sourceMode, setSourceMode] = useState('all')
  const filteredMovies = useMemo(() => {
    if (sourceMode === 'all') return movies
    return movies.filter((movie) => {
      const sources = Array.isArray(movie.sources) ? movie.sources : []
      const hasCF = sources.includes('CF') || movie.cf_rank != null
      const hasTFIDF = sources.includes('TF-IDF') || movie.tfidf_rank != null
      return sourceMode === 'cf' ? hasCF : hasTFIDF
    })
  }, [movies, sourceMode])

  return (
    <section className="movie-row" aria-label="Recommended for You">
      <div className="row-heading">
        <h2>Recommended for You</h2>
        <div className="source-toggle">
          <button type="button" className={sourceMode === 'all' ? 'active' : ''} onClick={() => setSourceMode('all')}>All</button>
          <button type="button" className={sourceMode === 'cf' ? 'active' : ''} onClick={() => setSourceMode('cf')}>Collaborative Filtering</button>
          <button type="button" className={sourceMode === 'tfidf' ? 'active' : ''} onClick={() => setSourceMode('tfidf')}>Content-Based (TF-IDF)</button>
        </div>
      </div>
      {filteredMovies.length > 0 ? (
        <div className="scroll-row">
          {filteredMovies.slice(0, 20).map((movie) => (
            <Card key={movie.id} movie={movie} onOpen={onOpen} />
          ))}
        </div>
      ) : (
        <div className="empty-row">
          {sourceMode === 'all'
            ? "Let's explore some movies so we can find the best recommendations for you!"
            : 'No recommendations for this source yet. Interact with movies or use Agent chat to generate more.'}
        </div>
      )}
    </section>
  )
}

export { MovieRow }
export default Recommendation
