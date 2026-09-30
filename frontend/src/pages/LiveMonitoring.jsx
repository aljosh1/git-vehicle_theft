/**
 * Live monitoring: pipeline control, MJPEG feed, real-time detection panel.
 *
 * The video arrives as MJPEG (pixels) while the side panel polls
 * /api/detections/live twice a second (structured results). Splitting them means
 * the panel stays responsive even when the video stream stalls, and the panel's
 * JSON stays small.
 */
import { useEffect, useRef, useState } from 'react'
import { api, mediaUrl, mjpegUrl } from '../api/client'
import { ErrorBanner, ThreatBadge } from '../components/ui.jsx'
import { useAuth } from '../context/AuthContext.jsx'

export default function LiveMonitoring() {
  const { isAdmin } = useAuth()
  const [cameraId, setCameraId] = useState('cam-0')
  const [source, setSource] = useState('0')
  const [liveOnlyMode, setLiveOnlyMode] = useState(false)
  const [running, setRunning] = useState(false)
  const [desiredMonitoring, setDesiredMonitoring] = useState(false)
  const [live, setLive] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [imageResult, setImageResult] = useState(null)
  const [imageDispatchAlert, setImageDispatchAlert] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  // Bumped on every start so the <img> remounts and reopens the MJPEG stream;
  // without it the browser reuses the old, now-dead connection.
  const [streamKey, setStreamKey] = useState(0)
  const videoInput = useRef(null)
  const imageInput = useRef(null)
  const streamRetryTimer = useRef(null)
  const autoStartTimer = useRef(null)

  useEffect(() => {
    return () => {
      if (streamRetryTimer.current) {
        clearTimeout(streamRetryTimer.current)
        streamRetryTimer.current = null
      }
      if (autoStartTimer.current) {
        clearTimeout(autoStartTimer.current)
        autoStartTimer.current = null
      }
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await api.liveDetections(cameraId)
        if (cancelled) return
        setLive(data)
        setRunning(Boolean(data.running))

        // Virtual cameras can drop briefly; if monitoring was requested,
        // restart the pipeline automatically instead of forcing manual restart.
        if (desiredMonitoring && !data.running && !busy && !imageResult) {
          if (!autoStartTimer.current) {
            autoStartTimer.current = setTimeout(async () => {
              autoStartTimer.current = null
              try {
                await api.startStream({
                  camera_id: cameraId,
                  source,
                  enable_tracking: true,
                  enable_alerts: !liveOnlyMode,
                  persist_data: !liveOnlyMode,
                })
                setStreamKey((key) => key + 1)
                setRunning(true)
                setError('')
              } catch {
                // Keep polling/retrying; transient camera failures are expected.
              }
            }, 1500)
          }
        }
      } catch {
        // A transient poll failure is not worth showing: the next tick retries.
      }
    }

    poll()
    const timer = setInterval(poll, 500)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [cameraId, desiredMonitoring, busy, imageResult, source, liveOnlyMode])

  useEffect(() => {
    let cancelled = false

    async function pollAlerts() {
      try {
        const data = await api.listAlerts('?limit=20')
        if (cancelled) return
        const relevant = (data ?? []).filter((alert) => alert.camera_id === cameraId)
        setAlerts(relevant)
      } catch {
        // Alerts are a bonus panel; failure should not interrupt the live feed.
      }
    }

    pollAlerts()
    const timer = setInterval(pollAlerts, 2000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [cameraId])

  async function handleStart() {
    setError('')
    setImageResult(null)
    setBusy(true)
    try {
      await api.startStream({
        camera_id: cameraId,
        source,
        enable_tracking: true,
        enable_alerts: !liveOnlyMode,
        persist_data: !liveOnlyMode,
      })
      setStreamKey((key) => key + 1)
      setRunning(true)
      setDesiredMonitoring(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleStop() {
    setError('')
    setBusy(true)
    try {
      await api.stopStream(cameraId)
      setRunning(false)
      setDesiredMonitoring(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      if (streamRetryTimer.current) {
        clearTimeout(streamRetryTimer.current)
        streamRetryTimer.current = null
      }
      if (autoStartTimer.current) {
        clearTimeout(autoStartTimer.current)
        autoStartTimer.current = null
      }
    }
  }

  function handleStreamError() {
    setError('The video stream dropped. Reconnecting...')
    if (!running || streamRetryTimer.current) return
    streamRetryTimer.current = setTimeout(() => {
      streamRetryTimer.current = null
      setStreamKey((key) => key + 1)
    }, 1200)
  }

  async function handleVideoUpload(event) {
    const file = event.target.files?.[0]
    if (!file) return
    setError('')
    setImageResult(null)
    setBusy(true)
    try {
      const result = await api.uploadVideo(file, 'upload-0')
      setCameraId(result.camera_id)
      setStreamKey((key) => key + 1)
      setRunning(true)
      setDesiredMonitoring(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      if (videoInput.current) videoInput.current.value = ''
    }
  }

  async function handleImageUpload(event) {
    const file = event.target.files?.[0]
    if (!file) return
    setError('')
    setBusy(true)
    try {
      if (running) {
        await api.stopStream(cameraId)
        setRunning(false)
      }
      setDesiredMonitoring(false)
      const result = await api.analyseImage(file, imageDispatchAlert)
      setImageResult(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      if (imageInput.current) imageInput.current.value = ''
    }
  }

  const displayed = imageResult ?? live
  const threat = displayed?.threat
  const verification = ownerVerificationDisplay(displayed?.owner_verification)
  const latestAlert = alerts[0]
  const latestAlertClipUrl = latestAlert?.video_clip_path ? mediaUrl(latestAlert.video_clip_path) : null

  return (
    <div>
      <header className="mb-5">
        <h1 className="text-xl font-semibold text-slate-900">Vehicle analysis</h1>
        <p className="text-sm text-slate-500">
          Analyse dataset images, uploaded videos, or a live camera feed
        </p>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError('')} />

      {isAdmin && (
        <div className="card mb-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-end">
            <div className="w-full sm:w-32">
              <label className="label">Camera ID</label>
              <input
                className="input"
                value={cameraId}
                onChange={(event) => setCameraId(event.target.value)}
                disabled={running}
              />
            </div>
            <div className="w-full sm:w-64 sm:flex-1">
              <label className="label">Source</label>
              <input
                className="input"
                value={source}
                onChange={(event) => setSource(event.target.value)}
                placeholder="0, or rtsp://…, or a file path"
                disabled={running}
              />
            </div>

            <label className="flex items-center gap-2 text-xs text-slate-600 sm:min-h-[42px]">
              <input
                type="checkbox"
                checked={liveOnlyMode}
                onChange={(event) => setLiveOnlyMode(event.target.checked)}
                disabled={running}
              />
              Live-only mode (disable database logging/evidence)
            </label>

            <label className="flex items-center gap-2 text-xs text-slate-600 sm:min-h-[42px]">
              <input
                type="checkbox"
                checked={imageDispatchAlert}
                onChange={(event) => setImageDispatchAlert(event.target.checked)}
                disabled={running}
              />
              Send alert for uploaded images
            </label>

            {running ? (
              <button onClick={handleStop} className="btn-danger w-full sm:w-auto" disabled={busy}>
                Stop
              </button>
            ) : (
              <button onClick={handleStart} className="btn-primary w-full sm:w-auto" disabled={busy}>
                {busy ? 'Starting…' : 'Start monitoring'}
              </button>
            )}

            <button
              onClick={() => imageInput.current?.click()}
              className="btn-secondary w-full sm:w-auto"
              disabled={busy}
            >
              {busy ? 'Analysing…' : 'Analyse image'}
            </button>
            <input
              ref={imageInput}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />

            <button
              onClick={() => videoInput.current?.click()}
              className="btn-secondary w-full sm:w-auto"
              disabled={busy}
            >
              Analyse video
            </button>
            <input
              ref={videoInput}
              type="file"
              accept="video/*"
              onChange={handleVideoUpload}
              className="hidden"
            />
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Image analysis runs every recognition stage once. Source <code>0</code> is
            the default webcam; RTSP sources and uploaded clips run continuously. In
            normal mode, live coverage verifies against registered database records.
            Enable live-only mode only when you want rolling detection without
            database persistence. Uploaded images only send email/SMS when
            <span className="font-medium text-slate-700"> Send alert for uploaded images </span>
            is checked.
          </p>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <section className="card lg:col-span-2">
          <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="text-sm font-semibold text-slate-900">
              {imageResult ? 'Annotated image' : 'Camera feed'}
            </h2>
            {imageResult ? (
              <span className="text-xs text-slate-500 sm:text-right">
                {imageResult.processing_ms} ms total · {imageResult.filename}
              </span>
            ) : running && live ? (
              <span className="text-xs text-slate-500 sm:text-right">
                {live.fps} FPS · {live.processing_ms} ms/frame · frame{' '}
                {live.frame_number}
              </span>
            ) : null}
          </div>
          <div className="flex aspect-video items-center justify-center overflow-hidden rounded-lg bg-slate-900">
            {imageResult ? (
              <img
                src={mediaUrl(imageResult.annotated_image)}
                alt="Annotated dataset analysis"
                className="h-full w-full object-contain"
              />
            ) : running ? (
              <img
                key={streamKey}
                src={mjpegUrl(cameraId)}
                alt="Live annotated feed"
                className="h-full w-full object-contain"
                onError={handleStreamError}
              />
            ) : (
              <div className="text-sm text-slate-500">
                No active analysis{isAdmin ? ' — select an image, video, or source above' : ''}
              </div>
            )}
          </div>
        </section>

        <section className="space-y-4">
          <div className="card">
            <h2 className="mb-3 text-sm font-semibold text-slate-900">
              Threat assessment
            </h2>
            {threat ? (
              <>
                {threat.is_armed || threat.is_concealed ? (
                  <ThreatBanner
                    weapons={displayed?.weapons ?? []}
                    masks={displayed?.masks ?? []}
                    occluded={displayed?.occluded_face_count ?? 0}
                    isArmed={threat.is_armed}
                  />
                ) : null}

                <div className="mb-3 flex items-center gap-3">
                  <ThreatBadge level={threat.level} />
                  <span className="text-2xl font-semibold tabular-nums">
                    {threat.score}
                    <span className="text-sm font-normal text-slate-400">/100</span>
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-200">
                  <div
                    className={`h-full transition-all ${
                      threat.score >= 60 ? 'bg-red-500' : 'bg-blue-500'
                    }`}
                    style={{ width: `${threat.score}%` }}
                  />
                </div>
                <p className="mt-3 text-xs leading-relaxed text-slate-600">
                  {threat.reason}
                </p>

                {threat.triggers ? (
                  <div className="mt-3 flex flex-wrap gap-1">
                    {threat.triggers
                      .split(',')
                      .filter(Boolean)
                      .map((trigger) => (
                        <span
                          key={trigger}
                          className={`rounded px-1.5 py-0.5 font-mono text-[10px] ring-1 ${
                            THREAT_TRIGGER_TONES[trigger] ??
                            'bg-white text-slate-500 ring-slate-200'
                          }`}
                        >
                          {trigger}
                        </span>
                      ))}
                  </div>
                ) : null}
              </>
            ) : (
              <p className="text-xs text-slate-400">No assessment yet</p>
            )}
          </div>

          <div className="card">
            <h2 className="mb-3 text-sm font-semibold text-slate-900">Latest alert clip</h2>
            {latestAlert ? (
              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <ThreatBadge level={latestAlert.threat_level} />
                  <span className="tabular-nums text-slate-500">
                    {latestAlert.threat_score}/100
                  </span>
                </div>
                <p className="leading-relaxed text-slate-600">{latestAlert.reason}</p>
                <div className="rounded-lg bg-slate-50 p-2 text-slate-600 ring-1 ring-slate-200">
                  <div className="mb-1 font-medium text-slate-700">
                    {latestAlertClipUrl ? 'Evidence clip ready' : 'Clip generating'}
                  </div>
                  <div className="break-all">
                    {latestAlertClipUrl ? (
                      <a
                        href={latestAlertClipUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="text-blue-600 hover:underline"
                      >
                        Open alert clip
                      </a>
                    ) : (
                      'Waiting for the clip to finish rendering...'
                    )}
                  </div>
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-500">
                  <span>{latestAlert.camera_id}</span>
                  <span>{latestAlert.created_at}</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-400">No alerts yet</p>
            )}
          </div>

          <div className="card">
            <h2 className="mb-3 text-sm font-semibold text-slate-900">This frame</h2>
            <dl className="space-y-2 text-xs">
              <Row label="Vehicles" value={displayed?.vehicle_count ?? 0} />
              <Row label="Persons" value={displayed?.person_count ?? 0} />
              <Row label="Animals" value={displayed?.animal_count ?? 0} />
              <Row
                label="Weapons"
                value={
                  <span className={displayed?.weapon_count ? 'font-semibold text-red-600' : ''}>
                    {displayed?.weapon_count ?? 0}
                  </span>
                }
              />
              <Row
                label="Masks / concealed"
                value={
                  <span className={displayed?.is_concealed ? 'font-semibold text-purple-600' : ''}>
                    {(displayed?.mask_count ?? 0) + (displayed?.occluded_face_count ?? 0)}
                  </span>
                }
              />
              <Row
                label="Plate"
                value={displayed?.plate?.text ?? '—'}
                mono
              />
              <Row
                label="Registered"
                value={
                  displayed?.plate?.in_database == null
                    ? '—'
                    : displayed.plate.in_database
                      ? 'Yes'
                      : 'No'
                }
                accent={
                  displayed?.plate?.in_database === false ? 'text-red-600' : undefined
                }
              />
              <Row
                label="Reported stolen"
                value={displayed?.plate?.flagged_stolen ? 'Yes' : 'No'}
                accent={displayed?.plate?.flagged_stolen ? 'text-red-600' : undefined}
              />
              <Row
                label="Person status"
                value={displayed?.person_status ?? '—'}
                accent={
                  displayed?.person_status === 'unauthorized'
                    ? 'text-red-600'
                    : displayed?.person_status === 'authorized'
                      ? 'text-green-600'
                      : undefined
                }
              />
              <Row
                label="Owner verification"
                value={verification.label}
                accent={verification.accent}
              />
              <Row
                label="Face match"
                value={
                  displayed?.face_similarity != null
                    ? `${(displayed.face_similarity * 100).toFixed(0)}%`
                    : '—'
                }
              />
            </dl>
          </div>

          {displayed?.vehicle_analyses?.length ? (
            <div className="card">
              <h2 className="mb-3 text-sm font-semibold text-slate-900">
                Vehicle analyses
              </h2>
              <ul className="max-h-64 space-y-2 overflow-y-auto text-xs">
                {displayed.vehicle_analyses.map((analysis, index) => (
                  <li
                    key={analysis.vehicle.track_id ?? index}
                    className="border-b border-slate-100 pb-2 last:border-0 last:pb-0"
                  >
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <span className="font-medium capitalize">
                        {analysis.vehicle.label}
                        {analysis.vehicle.track_id != null && ` #${analysis.vehicle.track_id}`}
                      </span>
                      <ThreatBadge level={analysis.threat.level} />
                    </div>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-slate-500">
                      <span className="font-mono text-slate-800">
                        {analysis.plate.text || 'No plate read'}
                      </span>
                      <span className="text-right tabular-nums">
                        Threat {analysis.threat.score}/100
                      </span>
                      <span>{analysis.person_status || 'No identified person'}</span>
                      <span
                        className={`text-right ${
                          ownerVerificationDisplay(analysis.owner_verification).accent
                        }`}
                      >
                        {ownerVerificationDisplay(analysis.owner_verification).label}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="card">
            <h2 className="mb-3 text-sm font-semibold text-slate-900">Detections</h2>
            {displayed?.detections?.length ? (
              <ul className="max-h-56 space-y-1.5 overflow-y-auto text-xs">
                {displayed.detections.map((det, index) => (
                  <li
                    key={index}
                    className="flex items-center justify-between rounded bg-slate-50 px-2 py-1.5"
                  >
                    <span className="font-medium capitalize">{det.label}</span>
                    <span className="tabular-nums text-slate-500">
                      {(det.confidence * 100).toFixed(0)}%
                      {det.track_id != null && ` · #${det.track_id}`}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-slate-400">Nothing detected</p>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}

function ownerVerificationDisplay(status) {
  const states = {
    verified: { label: 'Owner verified', accent: 'text-green-600' },
    mismatch: { label: 'Different registered owner', accent: 'text-red-600' },
    not_recognized: { label: 'Owner not recognized', accent: 'text-amber-600' },
    not_applicable: { label: 'Not applicable', accent: 'text-slate-500' },
  }
  return states[status] ?? { label: '—', accent: 'text-slate-500' }
}

// Trigger chips are colour-coded by how serious the rule is, so an operator
// scanning a wall of alerts sees "weapon" without reading each one.
const THREAT_TRIGGER_TONES = {
  firearm_detected: 'bg-red-100 text-red-800 ring-red-300',
  weapon_detected: 'bg-red-50 text-red-700 ring-red-200',
  threatening_tool: 'bg-orange-50 text-orange-700 ring-orange-200',
  mask_detected: 'bg-purple-50 text-purple-700 ring-purple-200',
  face_occluded: 'bg-purple-50 text-purple-600 ring-purple-200',
  vehicle_flagged_stolen: 'bg-red-100 text-red-800 ring-red-300',
  unauthorized_face: 'bg-amber-50 text-amber-700 ring-amber-200',
}

/**
 * Armed / concealed intruder banner.
 *
 * Deliberately the loudest element on the page and separate from the threat
 * score. A score of "100" and the words "ARMED" do different jobs: the score
 * ranks alerts for triage, this tells a guard what is actually in front of
 * them. It is hidden entirely when there is no threat, so it cannot cry wolf.
 *
 * An inferred occlusion is worded differently from a detected mask, because the
 * evidence is weaker and an operator needs to know which they are acting on.
 */
function ThreatBanner({ weapons, masks, occluded, isArmed }) {
  const parts = []
  if (weapons.length) parts.push(weapons.join(', '))
  if (masks.length) parts.push(masks.join(', '))
  if (occluded > 0) parts.push('face concealed (inferred)')

  const headline = isArmed
    ? 'ARMED INTRUDER DETECTED'
    : 'PERSON WITH CONCEALED FACE'

  return (
    <div
      role="alert"
      className={`mb-3 rounded-lg border-2 px-3 py-2 ${
        isArmed
          ? 'border-red-600 bg-red-600 text-white'
          : 'border-purple-500 bg-purple-50 text-purple-900'
      }`}
    >
      <div className="flex items-center gap-2 text-sm font-bold tracking-wide">
        <span aria-hidden="true">{isArmed ? '⚠' : '◐'}</span>
        {headline}
      </div>
      <div
        className={`mt-0.5 text-[11px] ${isArmed ? 'text-red-50' : 'text-purple-700'}`}
      >
        {parts.join(' · ')}
      </div>
    </div>
  )
}

function Row({ label, value, mono = false, accent }) {
  return (
    <div className="flex items-center justify-between">
      <dt className="text-slate-500">{label}</dt>
      <dd
        className={`font-medium capitalize ${mono ? 'font-mono' : ''} ${
          accent ?? 'text-slate-800'
        }`}
      >
        {value}
      </dd>
    </div>
  )
}
