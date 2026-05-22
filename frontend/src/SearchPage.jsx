import { useEffect, useState } from 'react'
import { searchMovies } from './action'
import { ExtendedCard } from './components/Card'
import Recommendation from './components/Recommendation'

const RESULTS_PER_PAGE = 5

function SearchPage({ query, recommendations, onOpenMovie }) {
  const [searchState, setSearchState] = useState({
    query: '',
    results: [],
    error: '',
  })
  const [pagination, setPagination] = useState({ query: '', page: 1 })
  const trimmedQuery = query.trim()
  const isCurrentQuery = searchState.query === trimmedQuery
  const results = trimmedQuery && isCurrentQuery ? searchState.results : []
  const totalPages = Math.max(1, Math.ceil(results.length / RESULTS_PER_PAGE))
  const resultPage = pagination.query === trimmedQuery ? pagination.page : 1
  const currentPage = Math.min(resultPage, totalPages)
  const pageStart = (currentPage - 1) * RESULTS_PER_PAGE
  const visibleResults = results.slice(pageStart, pageStart + RESULTS_PER_PAGE)
  const status = getSearchStatus(trimmedQuery, searchState, isCurrentQuery)

  function setResultPage(page) {
    setPagination({ query: trimmedQuery, page })
  }

  useEffect(() => {
    let isMounted = true

    if (!trimmedQuery) {
      return () => {
        isMounted = false
      }
    }

    searchMovies(trimmedQuery)
      .then((data) => {
        if (!isMounted) {
          return
        }
        setSearchState({
          query: trimmedQuery,
          results: data.results ?? [],
          error: '',
        })
      })
      .catch(() => {
        if (isMounted) {
          setSearchState({
            query: trimmedQuery,
            results: [],
            error: 'Search is unavailable. Start the Flask API and Qdrant, then try again.',
          })
        }
      })

    return () => {
      isMounted = false
    }
  }, [trimmedQuery])

  return (
    <div className="page-stack">
      <title>Search | Let's Watch!</title>

      <section className="search-results">
        <div className="row-heading">
          <h1>Search results</h1>
          <span>{status}</span>
        </div>

        <div className="result-list">
          {visibleResults.map((movie) => (
            <ExtendedCard key={movie.id} movie={movie} query={trimmedQuery} onOpen={onOpenMovie} />
          ))}
        </div>

        {results.length > RESULTS_PER_PAGE && (
          <nav className="pagination" aria-label="Search result pages">
            <button
              type="button"
              onClick={() => setResultPage(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
            >
              Previous
            </button>
            <div className="page-numbers">
              {Array.from({ length: totalPages }, (_, index) => {
                const page = index + 1
                return (
                  <button
                    key={page}
                    type="button"
                    className={page === currentPage ? 'active' : ''}
                    aria-current={page === currentPage ? 'page' : undefined}
                    onClick={() => setResultPage(page)}
                  >
                    {page}
                  </button>
                )
              })}
            </div>
            <button
              type="button"
              onClick={() => setResultPage(Math.min(totalPages, currentPage + 1))}
              disabled={currentPage === totalPages}
            >
              Next
            </button>
          </nav>
        )}
      </section>

      <Recommendation movies={recommendations} onOpen={onOpenMovie} />
    </div>
  )
}

function getSearchStatus(query, searchState, isCurrentQuery) {
  if (!query) {
    return 'Type a search above.'
  }
  if (!isCurrentQuery) {
    return 'Searching...'
  }
  if (searchState.error) {
    return searchState.error
  }
  if (searchState.results.length === 0) {
    return `No results for "${query}"`
  }
  return `${searchState.results.length} results for "${query}"`
}

export default SearchPage
