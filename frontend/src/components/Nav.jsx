import { useState } from 'react'

function Nav({ initialQuery = '', user, onNavigate, onSearch, onLogin, onLogout }) {
  const [query, setQuery] = useState(initialQuery)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

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

      {user ? (
        <div className="user-profile" title={`Current user: ${user.username}`}>
          <div className="avatar">{String(user.name || 'U').slice(0, 1).toUpperCase()}</div>
          <div className="user-info">
            <div className="user-label">{user.name}</div>
            <div className="user-id">{user.username} · #{user.user_id}</div>
          </div>
          <button type="button" className="user-edit" aria-label="Logout" onClick={onLogout}>
            Logout
          </button>
        </div>
      ) : (
        <form
          className="user-profile"
          onSubmit={(e) => {
            e.preventDefault()
            onLogin?.(username.trim(), password)
          }}
        >
          <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="username" />
          <input value={password} onChange={(e) => setPassword(e.target.value)} placeholder="password" type="password" />
          <button type="submit" className="user-edit">Login</button>
        </form>
      )}
    </nav>
  )
}

export default Nav
