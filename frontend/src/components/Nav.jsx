import { useState } from 'react'

function Nav({ initialQuery = '', userId = '1337', onNavigate, onSearch, onUserChange }) {
  const [query, setQuery] = useState(initialQuery)

  function handleSubmit(event) {
    event.preventDefault()
    const trimmed = query.trim()
    if (trimmed) {
      onSearch(trimmed)
    }
  }

  return (
    <nav className="navbar">
      <button type="button" className="brand" onClick={() => onNavigate('/')}>
        <span className="brand-mark">LW</span>
        <div className="eyebrow">Let's Watch!</div>
      </button>

      <form className="search-bar" onSubmit={handleSubmit}>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by title, genre, mood, or description"
          aria-label="Search movies"
        />
        <button type="submit">Search</button>
      </form>

      <div className="user-profile" title={`Current user: ${userId}`}>
        <div className="avatar">U</div>
        <div className="user-info">
          <div className="user-label">User</div>
          <div className="user-id">{userId}</div>
        </div>
        <button
          type="button"
          className="user-edit"
          aria-label="Change user id"
          onClick={() => {
            const val = window.prompt('Set current user id (for demo):', String(userId))
            if (val !== null && val.trim() !== '') {
              onUserChange?.(val.trim())
            }
          }}
        >
          ✎
        </button>
      </div>
    </nav>
  )
}

export default Nav
