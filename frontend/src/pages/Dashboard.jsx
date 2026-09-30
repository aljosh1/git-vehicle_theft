/**
 * Dashboard: live feed preview, KPI tiles, activity chart, recent alerts.
 *
 * Stats are polled every 5 s rather than pushed over a WebSocket: the dashboard
 * only needs second-scale freshness, and polling one small JSON endpoint keeps
 * the backend free of connection state.
 */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, mjpegUrl } from '../api/client'
import { useAuth } from '../context/AuthContext.jsx'
import {
  EmptyState,
  ErrorBanner,
  formatDateTime,
  Spinner,
  StatCard,
  ThreatBadge,
} from '../components/ui.jsx'

export default function Dashboard() {
  const { isAdmin } = useAuth()
  const [stats, setStats] = useState(null)
  const [emailTest, setEmailTest] = useState({ state: 'idle', message: '' })
  const [series, setSeries] = useState([])
  const [alerts, setAlerts] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [statsData, seriesData, alertsData] = await Promise.all([
          api.stats(),
          api.timeseries(7),
          api.listAlerts('?limit=5'),
        ])
        if (cancelled) return
        setStats(statsData)
        setSeries(seriesData)
        setAlerts(alertsData)
        setError('')
      } catch (err) {
        if (!cancelled) setError(err.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    const timer = setInterval(load, 5000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  if (loading) return <Spinner label="Loading dashboard…" />

  return (
    <div>
      <header className="mb-5">
        <h1 className="text-xl font-semibold text-slate-900">Dashboard</h1>
        <p className="text-sm text-slate-500">
          Surveillance overview and recent activity
        </p>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError('')} />

      <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          label="Registered vehicles"
          value={stats?.total_vehicles ?? 0}
          sub={`${stats?.total_owners ?? 0} owners`}
        />
        <StatCard
          label="Detections today"
          value={stats?.detections_today ?? 0}
          sub={`${stats?.total_detections ?? 0} all time`}
        />
        <StatCard
          label="Open alerts"
          value={stats?.open_alerts ?? 0}
          sub={`${stats?.alerts_today ?? 0} raised today`}
          accent={stats?.open_alerts ? 'text-red-600' : 'text-slate-900'}
        />
        <StatCard
          label="Pipeline"
          value={stats?.pipeline_running ? `${stats.current_fps} FPS` : 'Stopped'}
          sub={stats?.pipeline_running ? 'Live' : 'Start it in Live monitoring'}
          accent={stats?.pipeline_running ? 'text-green-600' : 'text-slate-400'}
        />
      </div>

      {isAdmin && (
        <section className="card mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Email delivery</h2>
            <p className="text-xs text-slate-500">
              Send a real SMTP test message to the configured alert recipients.
            </p>
            {emailTest.message && (
              <p
                className={`mt-1 text-xs ${emailTest.state === 'success' ? 'text-green-600' : 'text-red-600'}`}
              >
                {emailTest.message}
              </p>
            )}
          </div>
          <button
            type="button"
            className="btn-secondary"
            disabled={emailTest.state === 'sending'}
            onClick={async () => {
              setEmailTest({ state: 'sending', message: '' })
              try {
                const result = await api.testEmail()
                setEmailTest({
                  state: 'success',
                  message: `${result.message} (${result.recipients.join(', ')})`,
                })
              } catch (err) {
                setEmailTest({ state: 'error', message: err.message })
              }
            }}
          >
            {emailTest.state === 'sending' ? 'Sending…' : 'Send test email'}
          </button>
        </section>
      )}

      <div className="mb-5 grid gap-4 lg:grid-cols-2">
        <section className="card">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-900">Live camera</h2>
            <Link to="/live-camera" className="text-xs text-blue-600 hover:underline">
              Full view →
            </Link>
          </div>
          <div className="flex aspect-video items-center justify-center overflow-hidden rounded-lg bg-slate-900">
            {stats?.pipeline_running ? (
              <img
                src={mjpegUrl('cam-0')}
                alt="Live camera feed"
                className="h-full w-full object-contain"
              />
            ) : (
              <div className="text-center text-sm text-slate-500">
                <div>No active feed</div>
                <Link to="/live-camera" className="mt-1 inline-block text-xs text-blue-400 hover:underline">
                  Start monitoring
                </Link>
              </div>
            )}
          </div>
        </section>

        <section className="card">
          <h2 className="mb-3 text-sm font-semibold text-slate-900">
            Activity — last 7 days
          </h2>
          <div className="h-[248px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={series}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} stroke="#94a3b8" />
                <YAxis tick={{ fontSize: 12 }} stroke="#94a3b8" allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="detections" fill="#2563eb" name="Detections" radius={[3, 3, 0, 0]} />
                <Bar dataKey="alerts" fill="#dc2626" name="Alerts" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section className="card">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-900">Recent alerts</h2>
          <Link to="/alerts" className="text-xs text-blue-600 hover:underline">
            All alerts →
          </Link>
        </div>

        {alerts.length === 0 ? (
          <EmptyState
            title="No alerts recorded"
            hint="Alerts appear here when the threat score crosses the configured threshold"
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-slate-200">
                <tr>
                  <th className="th">Level</th>
                  <th className="th">Incident</th>
                  <th className="th">Plate</th>
                  <th className="th">When</th>
                  <th className="th">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {alerts.map((alert) => (
                  <tr key={alert.id} className="hover:bg-slate-50">
                    <td className="td">
                      <ThreatBadge level={alert.threat_level} score={alert.threat_score} />
                    </td>
                    <td className="td max-w-md truncate">{alert.title}</td>
                    <td className="td font-mono text-xs">{alert.plate_number ?? '—'}</td>
                    <td className="td whitespace-nowrap text-xs text-slate-500">
                      {formatDateTime(alert.created_at)}
                    </td>
                    <td className="td text-xs capitalize">{alert.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
