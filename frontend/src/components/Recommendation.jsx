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
  return (
    <MovieRow
      title="Recommended for You"
      movies={movies}
      emptyText="Let's explore some movies so we can find the best recommendations for you!"
      onOpen={onOpen}
    />
  )
}

export { MovieRow }
export default Recommendation
