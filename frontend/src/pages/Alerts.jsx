/**
 * Alert history and incident workflow.
 *
 * Marking an alert a false positive is the feedback signal for re-tuning the
 * scoring weights in backend/core/theft_engine.py, so it is a first-class
 * action here rather than buried in a menu.
 */
import { useCallback, useEffect, useState } from 'react'
import { api, mediaUrl } from '../api/client'
import { useAuth } from '../context/AuthContext.jsx'
import {
  EmptyState,
  ErrorBanner,
  formatDateTime,
  Modal,
  Spinner,
  ThreatBadge,
} from '../components/ui.jsx'

const STATUS_FILTERS = [
  { value: '', label: 'All' },
  { value: 'new', label: 'New' },
  { value: 'acknowledged', label: 'Acknowledged' },
  { value: 'resolved', label: 'Resolved' },
  { value: 'false_positive', label: 'False positives' },
]

export default function Alerts() {
  const { isAdmin } = useAuth()
  const [alerts, setAlerts] = useState([])
  const [statusFilter, setStatusFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null)
  const [testingEmail, setTestingEmail] = useState(false)
  const [testingSms, setTestingSms] = useState(false)
  const [channels, setChannels] = useState(null)

  const load = useCallback(async () => {
    try {
      const query = statusFilter ? `?status=${statusFilter}&limit=100` : '?limit=100'
      setAlerts(await api.listAlerts(query))
      setError('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [statusFilter])

  useEffect(() => {
    load()
    const timer = setInterval(load, 10000)
    return () => clearInterval(timer)
  }, [load])

  // Which channels are actually armed. Fetched once because it is configuration,
  // not state - and the buttons are disabled from it so an operator is never
  // told an SMS "went out" when the channel is switched off.
  useEffect(() => {
    api.alertChannels().then(setChannels).catch(() => setChannels(null))
  }, [])

  async function updateStatus(alert, status, notes) {
    try {
      await api.updateAlert(alert.id, { status, resolution_notes: notes ?? null })
      setSelected(null)
      await load()
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleTestEmail() {
    setTestingEmail(true)
    try {
      const result = await api.testEmail()
      setError(`Test email sent to: ${(result.recipients ?? []).join(', ')}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setTestingEmail(false)
    }
  }

  async function handleTestSms() {
    setTestingSms(true)
    try {
      const result = await api.testSms()
      setError(`Test SMS sent to: ${(result.recipients ?? []).join(', ')}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setTestingSms(false)
    }
  }

  if (loading) return <Spinner label="Loading alerts…" />

  return (
    <div>
      <header className="mb-5">
        <h1 className="text-xl font-semibold text-slate-900">Alert history</h1>
        <p className="text-sm text-slate-500">
          Every incident that crossed the alerting threshold
        </p>
        {isAdmin ? (
          <div className="mt-3 space-y-2">
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleTestEmail}
                className="btn-secondary"
                disabled={testingEmail}
              >
                {testingEmail ? 'Sending…' : 'Send test email'}
              </button>
              <button
                onClick={handleTestSms}
                className="btn-secondary"
                disabled={testingSms || !channels?.sms?.enabled}
                title={
                  channels?.sms?.enabled
                    ? 'Send a test SMS through Twilio'
                    : 'SMS alerts are disabled (ALERTS_SMS_ENABLED=false)'
                }
              >
                {testingSms ? 'Sending…' : 'Send test SMS'}
              </button>
            </div>
            <ChannelStatus channels={channels} />
          </div>
        ) : null}
      </header>

      <ErrorBanner message={error} onDismiss={() => setError('')} />

      <div className="mb-4 flex gap-2">
        {STATUS_FILTERS.map((filter) => (
          <button
            key={filter.value}
            onClick={() => setStatusFilter(filter.value)}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              statusFilter === filter.value
                ? 'bg-blue-600 text-white'
                : 'bg-white text-slate-600 ring-1 ring-slate-300 hover:bg-slate-50'
            }`}
          >
            {filter.label}
          </button>
        ))}
      </div>

      <div className="card overflow-x-auto p-0">
        {alerts.length === 0 ? (
          <EmptyState
            title="No alerts"
            hint="Incidents appear here when the threat score crosses the threshold"
          />
        ) : (
          <table className="w-full">
            <thead className="border-b border-slate-200 bg-slate-50">
              <tr>
                <th className="th">Level</th>
                <th className="th">Incident</th>
                <th className="th">Plate</th>
                <th className="th">Camera</th>
                <th className="th">When</th>
                <th className="th">Delivery</th>
                <th className="th">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {alerts.map((alert) => (
                <tr
                  key={alert.id}
                  onClick={() => setSelected(alert)}
                  className="cursor-pointer hover:bg-slate-50"
                >
                  <td className="td">
                    <ThreatBadge level={alert.threat_level} score={alert.threat_score} />
                  </td>
                  <td className="td max-w-xs truncate">{alert.title}</td>
                  <td className="td font-mono text-xs">{alert.plate_number ?? '—'}</td>
                  <td className="td text-xs">{alert.camera_id}</td>
                  <td className="td whitespace-nowrap text-xs text-slate-500">
                    {formatDateTime(alert.created_at)}
                  </td>
                  <td className="td text-xs">
                    <span className={alert.email_sent ? 'text-green-600' : 'text-slate-300'}>
                      ✉
                    </span>
                    <span
                      className={`ml-1.5 ${alert.sms_sent ? 'text-green-600' : 'text-slate-300'}`}
                    >
                      ✆
                    </span>
                  </td>
                  <td className="td">
                    <StatusPill status={alert.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {selected && (
        <AlertDetail
          alert={selected}
          onClose={() => setSelected(null)}
          onUpdate={updateStatus}
        />
      )}
    </div>
  )
}

function ChannelStatus({ channels }) {
  if (!channels) return null

  return (
    <div className="flex flex-wrap gap-4 text-[11px] text-slate-500">
      <ChannelPill
        name="Email"
        channel={channels.email}
        detail={channels.email?.provider}
      />
      <ChannelPill name="SMS" channel={channels.sms} detail="Twilio" />
    </div>
  )
}

/**
 * One notification channel's readiness.
 *
 * `configured` is separated from `enabled` on purpose: a channel can be switched
 * on and still be unable to send (no API key, no credentials), and collapsing
 * the two would show an operator a green light over a dead channel.
 */
function ChannelPill({ name, channel, detail }) {
  if (!channel) return null
  const ready = channel.enabled && channel.configured
  const recipients = channel.recipients ?? []

  const tone = ready
    ? 'bg-green-50 text-green-700 ring-green-200'
    : channel.enabled
      ? 'bg-amber-50 text-amber-700 ring-amber-200'
      : 'bg-slate-100 text-slate-500 ring-slate-200'

  const reason = !channel.enabled
    ? 'disabled in .env'
    : !channel.configured
      ? 'enabled but not configured'
      : recipients.length
        ? recipients.join(', ')
        : 'no recipients'

  return (
    <span className={`rounded-full px-2 py-0.5 ring-1 ${tone}`} title={reason}>
      {name} {ready ? 'ready' : 'unavailable'} · {reason}
      {detail ? ` (${detail})` : ''}
    </span>
  )
}

function StatusPill({ status }) {
  const styles = {
    new: 'bg-red-50 text-red-700 ring-red-200',
    acknowledged: 'bg-yellow-50 text-yellow-700 ring-yellow-200',
    resolved: 'bg-green-50 text-green-700 ring-green-200',
    false_positive: 'bg-slate-100 text-slate-500 ring-slate-200',
  }
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-xs font-medium capitalize ring-1 ${
        styles[status] ?? styles.new
      }`}
    >
      {status.replace('_', ' ')}
    </span>
  )
}

function AlertDetail({ alert, onClose, onUpdate }) {
  const [notes, setNotes] = useState(alert.resolution_notes ?? '')

  return (
    <Modal title={alert.title} onClose={onClose} wide>
      <div className="mb-4 flex items-center gap-3">
        <ThreatBadge level={alert.threat_level} score={alert.threat_score} />
        <StatusPill status={alert.status} />
        <span className="text-xs text-slate-500">
          {formatDateTime(alert.created_at)}
        </span>
      </div>

      <div className="mb-4 rounded-lg bg-slate-50 p-3">
        <div className="mb-1 text-xs font-semibold text-slate-500">Reason</div>
        <p className="text-sm text-slate-700">{alert.reason}</p>
        {alert.triggers && (
          <div className="mt-2 flex flex-wrap gap-1">
            {alert.triggers.split(',').map((trigger) => (
              <span
                key={trigger}
                className="rounded bg-white px-1.5 py-0.5 font-mono text-[10px] text-slate-500 ring-1 ring-slate-200"
              >
                {trigger}
              </span>
            ))}
          </div>
        )}
      </div>

      <dl className="mb-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
        <Field label="Plate" value={alert.plate_number ?? 'Not read'} mono />
        <Field label="Person" value={alert.person_status ?? '—'} />
        <Field label="Camera" value={alert.camera_id} />
        <Field label="Site" value={alert.site_name ?? '—'} />
        <Field
          label="GPS"
          value={
            alert.latitude != null ? (
              <a
                href={`https://maps.google.com/?q=${alert.latitude},${alert.longitude}`}
                target="_blank"
                rel="noreferrer"
                className="text-blue-600 hover:underline"
              >
                {alert.latitude.toFixed(4)}, {alert.longitude.toFixed(4)}
              </a>
            ) : (
              '—'
            )
          }
        />
        <Field
          label="Notified"
          value={`email ${alert.email_sent ? '✓' : '✗'} · sms ${alert.sms_sent ? '✓' : '✗'}`}
        />
      </dl>

      {alert.delivery_error && (
        <div className="mb-4 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800 ring-1 ring-amber-200">
          Delivery problem: {alert.delivery_error}
        </div>
      )}

      {(alert.face_image_path || alert.snapshot_path) && (
        <div className="mb-4">
          <div className="mb-2 text-xs font-semibold text-slate-500">Evidence</div>
          <div className="grid grid-cols-2 gap-3">
            {alert.face_image_path && (
              <figure>
                <img
                  src={mediaUrl(alert.face_image_path)}
                  alt="Captured face"
                  className="w-full rounded-lg ring-1 ring-slate-200"
                />
                <figcaption className="mt-1 text-[10px] text-slate-400">
                  Captured face
                </figcaption>
              </figure>
            )}
            {alert.snapshot_path && (
              <figure>
                <img
                  src={mediaUrl(alert.snapshot_path)}
                  alt="Scene snapshot"
                  className="w-full rounded-lg ring-1 ring-slate-200"
                />
                <figcaption className="mt-1 text-[10px] text-slate-400">
                  Scene snapshot
                </figcaption>
              </figure>
            )}
          </div>
        </div>
      )}

      <div className="mb-3">
        <label className="label">Resolution notes</label>
        <textarea
          className="input"
          rows={2}
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          placeholder="What was the outcome?"
        />
      </div>

      <div className="flex flex-wrap justify-end gap-2">
        <button
          onClick={() => onUpdate(alert, 'false_positive', notes)}
          className="btn-secondary"
        >
          False positive
        </button>
        <button
          onClick={() => onUpdate(alert, 'acknowledged', notes)}
          className="btn-secondary"
        >
          Acknowledge
        </button>
        <button onClick={() => onUpdate(alert, 'resolved', notes)} className="btn-primary">
          Resolve
        </button>
      </div>
    </Modal>
  )
}

function Field({ label, value, mono = false }) {
  return (
    <div>
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className={`capitalize text-slate-800 ${mono ? 'font-mono' : ''}`}>{value}</dd>
    </div>
  )
}
