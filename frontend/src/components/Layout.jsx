/** Sidebar + header shell wrapping every authenticated page. */
import { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

const NAV = [
  { to: '/', label: 'Dashboard', icon: '▤', end: true },
  { to: '/live-camera', label: 'Live Camera', icon: '◉' },
  { to: '/verify', label: 'Verification', icon: '✓' },
  { to: '/vehicles', label: 'Vehicles', icon: '▭' },
  { to: '/owners', label: 'Owners', icon: '☰', adminOnly: true },
  { to: '/alerts', label: 'Alerts', icon: '⚠' },
  { to: '/training', label: 'Model training', icon: '↗', adminOnly: true },
]

export default function Layout() {
  const { user, isAdmin, logout } = useAuth()
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)

  function handleLogout() {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <div className="min-h-screen bg-slate-100 lg:flex">
      <div className="sticky top-0 z-30 flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 shadow-sm lg:hidden">
        <div>
          <div className="text-[11px] uppercase tracking-widest text-slate-500">
            Surveillance
          </div>
          <div className="text-sm font-semibold text-slate-900">Theft Detection</div>
        </div>
        <button
          type="button"
          onClick={() => setMenuOpen((open) => !open)}
          className="rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-700 transition hover:bg-slate-50"
        >
          {menuOpen ? 'Close' : 'Menu'}
        </button>
      </div>

      {menuOpen ? (
        <button
          type="button"
          aria-label="Close navigation"
          className="fixed inset-0 z-30 bg-slate-950/40 lg:hidden"
          onClick={() => setMenuOpen(false)}
        />
      ) : null}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-72 max-w-[85vw] flex-col bg-slate-900 text-slate-300 transition-transform lg:static lg:z-auto lg:w-60 lg:max-w-none lg:translate-x-0 ${
          menuOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="border-b border-slate-800 px-5 py-5">
          <div className="text-[11px] uppercase tracking-widest text-slate-500">
            Surveillance
          </div>
          <div className="mt-0.5 text-base font-semibold text-white">
            Theft Detection
          </div>
        </div>

        <nav className="flex-1 space-y-1 p-3">
          {NAV.filter((item) => !item.adminOnly || isAdmin).map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              onClick={() => setMenuOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'hover:bg-slate-800 hover:text-white'
                }`
              }
            >
              <span className="w-4 text-center">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-slate-800 p-4">
          <div className="truncate text-sm font-medium text-white">
            {user?.full_name}
          </div>
          <div className="mb-3 text-xs capitalize text-slate-500">{user?.role}</div>
          <button
            onClick={handleLogout}
            className="w-full rounded-lg bg-slate-800 px-3 py-2 text-xs
                       text-slate-300 transition hover:bg-slate-700 hover:text-white"
          >
            Sign out
          </button>
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-x-hidden p-4 sm:p-6">
        <Outlet />
      </main>
    </div>
  )
}
