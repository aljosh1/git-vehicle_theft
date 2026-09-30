/**
 * Small shared presentational components.
 *
 * Kept in one file because each is only a few lines; splitting them across six
 * modules would add imports without adding clarity.
 */

/** Threat level pill. Colours match the ThreatLevel enum in the backend. */
export function ThreatBadge({ level, score }) {
  const styles = {
    none: 'bg-slate-100 text-slate-600 ring-slate-200',
    low: 'bg-cyan-50 text-cyan-700 ring-cyan-200',
    medium: 'bg-yellow-50 text-yellow-700 ring-yellow-200',
    high: 'bg-orange-50 text-orange-700 ring-orange-200',
    critical: 'bg-red-50 text-red-700 ring-red-200',
  }
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5
                  text-xs font-semibold uppercase ring-1 ${
                    styles[level] ?? styles.none
                  }`}
    >
      {level}
      {score != null && <span className="font-normal opacity-70">{score}</span>}
    </span>
  )
}

/** Dashboard KPI tile. */
export function StatCard({ label, value, sub, accent = 'text-slate-900' }) {
  return (
    <div className="card">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className={`mt-1.5 text-3xl font-semibold tabular-nums ${accent}`}>
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

/** Inline error banner. Renders nothing when `message` is falsy. */
export function ErrorBanner({ message, onDismiss }) {
  if (!message) return null
  return (
    <div
      className="mb-4 flex items-start justify-between gap-3 rounded-lg
                 bg-red-50 px-4 py-3 text-sm text-red-800 ring-1 ring-red-200"
    >
      <span>{message}</span>
      {onDismiss && (
        <button onClick={onDismiss} className="text-red-600 hover:text-red-800">
          ✕
        </button>
      )}
    </div>
  )
}

export function Spinner({ label = 'Loading…' }) {
  return (
    <div className="flex items-center gap-3 py-10 text-sm text-slate-500">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-blue-600" />
      {label}
    </div>
  )
}

export function EmptyState({ title, hint }) {
  return (
    <div className="py-12 text-center">
      <div className="text-sm font-medium text-slate-600">{title}</div>
      {hint && <div className="mt-1 text-xs text-slate-400">{hint}</div>}
    </div>
  )
}

/** Modal dialog. Closes on backdrop click. */
export function Modal({ title, children, onClose, wide = false }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto
                 bg-slate-900/50 p-6"
      onClick={onClose}
    >
      <div
        className={`mt-8 w-full rounded-xl bg-white shadow-xl ${
          wide ? 'max-w-3xl' : 'max-w-lg'
        }`}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            ✕
          </button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  )
}

/** UTC timestamps from the API, rendered in the viewer's local timezone. */
export function formatDateTime(value) {
  if (!value) return '—'
  const iso = value.endsWith('Z') || value.includes('+') ? value : `${value}Z`
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}
