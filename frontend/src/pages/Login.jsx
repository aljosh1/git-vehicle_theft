import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ErrorBanner } from '../components/ui.jsx'
import { useAuth } from '../context/AuthContext.jsx'

export default function Login() {
  const { login, isAuthenticated } = useAuth()
  const navigate = useNavigate()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (isAuthenticated) {
    navigate('/', { replace: true })
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      await login(email, password)
      navigate('/', { replace: true })
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-900 p-6">
      <div className="w-full max-w-sm">
        <div className="mb-6 text-center">
          <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
            Vehicle Theft Detection System
          </div>
          <h1 className="mt-1 text-xl font-semibold text-white">Sign in</h1>
        </div>

        <form onSubmit={handleSubmit} className="rounded-xl bg-white p-6 shadow-xl">
          <ErrorBanner message={error} onDismiss={() => setError('')} />

          <div className="mb-4">
            <label className="label" htmlFor="email">
              Email address
            </label>
            <input
              id="email"
              type="email"
              className="input"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
              autoFocus
              autoComplete="username"
            />
          </div>

          <div className="mb-5">
            <label className="label" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              className="input"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              autoComplete="current-password"
            />
          </div>

          <button type="submit" className="btn-primary w-full" disabled={busy}>
            {busy ? 'Signing in…' : 'Sign in'}
          </button>

          <p className="mt-4 text-center text-xs text-slate-500">
            Seeded administrator: <code>admin@vtds.example.com</code>
          </p>
        </form>
      </div>
    </div>
  )
}
