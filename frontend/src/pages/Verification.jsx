import { useState } from 'react'
import { api, mediaUrl } from '../api/client'
import { ErrorBanner } from '../components/ui.jsx'

export default function Verification() {
  const [error, setError] = useState('')

  return (
    <div>
      <header className="mb-5">
        <h1 className="text-xl font-semibold text-slate-900">Verification</h1>
        <p className="text-sm text-slate-500">
          Verify a human face, a membership card, a typed plate, or a vehicle image
          instantly.
        </p>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError('')} />

      <div className="grid gap-4 lg:grid-cols-2">
        <HumanVerificationCard setPageError={setError} />
        <BarcodeVerificationCard setPageError={setError} />
        <PlateVerificationCard setPageError={setError} />
        <VehicleVerificationCard setPageError={setError} />
      </div>
    </div>
  )
}

/**
 * Face verification.
 *
 * The verdict is derived from four pieces of information the backend returns,
 * and the panel deliberately shows the similarity against the threshold rather
 * than a bare pass/fail. A guard needs to see *how close* a call was: "0.41
 * against a 0.363 cutoff" is a judgement they can make, "UNSURE" is not.
 */
function HumanVerificationCard({ setPageError }) {
  const [file, setFile] = useState(null)
  const [expectedOwnerId, setExpectedOwnerId] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  async function handleVerify(event) {
    event.preventDefault()
    if (!file) return
    setLoading(true)
    setPageError('')
    try {
      const payload = await api.verifyFaceImage(
        file,
        expectedOwnerId.trim() ? Number(expectedOwnerId) : null,
      )
      setResult(payload)
    } catch (err) {
      setPageError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const state = humanVerdict(result)

  return (
    <section className="card">
      <h2 className="mb-3 text-sm font-semibold text-slate-900">Human image</h2>
      <form className="space-y-2" onSubmit={handleVerify}>
        <div>
          <label className="label">Face image</label>
          <input
            type="file"
            accept="image/*"
            className="input block text-xs"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            required
          />
        </div>
        <div>
          <label className="label">Expected owner ID (optional)</label>
          <input
            className="input"
            placeholder="e.g. 4"
            value={expectedOwnerId}
            onChange={(event) => setExpectedOwnerId(event.target.value)}
          />
        </div>
        <button className="btn-primary" disabled={!file || loading}>
          {loading ? 'Verifying…' : 'Verify human'}
        </button>
      </form>

      <VerdictChip state={state.type} label={state.label} />

      {result ? (
        <>
          <dl className="mt-3 space-y-1 text-xs text-slate-600">
            <Line label="Status" value={PERSON_STATUS_LABELS[result.person_status] ?? result.person_status} />
            <Line
              label="Matched person"
              value={
                result.matched_user_name
                  ? `${result.matched_user_name} (#${result.matched_user_id})`
                  : '—'
              }
            />
            <Line
              label="Similarity"
              value={
                result.face_similarity != null
                  ? `${(result.face_similarity * 100).toFixed(0)}%`
                  : '—'
              }
            />
            <Line
              label="Match threshold"
              value={
                result.match_threshold != null
                  ? `${(result.match_threshold * 100).toFixed(0)}%`
                  : '—'
              }
            />
            <Line label="Owner verification" value={OWNER_VERIFICATION_LABELS[result.owner_verification] ?? result.owner_verification} />
            <Line label="Model" value={result.face_backend ?? '—'} />
            <Line label="Enrolled faces" value={result.gallery_size} />
          </dl>

          <SimilarityBar
            similarity={result.face_similarity}
            threshold={result.match_threshold}
          />

          {result.gallery_is_empty ? (
            <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800 ring-1 ring-amber-200">
              No faces are enrolled yet, so nobody can be recognised. Enrol an
              owner under Owners → Faces first.
            </p>
          ) : null}
        </>
      ) : null}
    </section>
  )
}

/**
 * Membership card verification.
 *
 * Accepts both a photo of the card and a code typed or piped in from a
 * hand-held reader, because a guard at a gate usually has a scanner rather
 * than a camera. The four outcomes are shown distinctly - a card that read
 * nothing and a card that belongs to a deleted account are very different
 * situations, and collapsing them into "invalid" would hide that.
 */
function BarcodeVerificationCard({ setPageError }) {
  const [file, setFile] = useState(null)
  const [typedCode, setTypedCode] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  async function verify(payload) {
    setLoading(true)
    setPageError('')
    try {
      setResult(await payload())
    } catch (err) {
      setPageError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const state = barcodeVerdict(result)

  return (
    <section className="card">
      <h2 className="mb-3 text-sm font-semibold text-slate-900">Membership card</h2>

      <form
        className="space-y-2"
        onSubmit={(event) => {
          event.preventDefault()
          if (file) verify(() => api.scanBarcode(file))
        }}
      >
        <div>
          <label className="label">Scan or photograph the card</label>
          <input
            type="file"
            accept="image/*"
            className="input block text-xs"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </div>
        <button className="btn-primary" disabled={!file || loading}>
          {loading ? 'Reading…' : 'Read card'}
        </button>
      </form>

      <form
        className="mt-3 space-y-2"
        onSubmit={(event) => {
          event.preventDefault()
          if (typedCode.trim()) verify(() => api.verifyCode(typedCode.trim()))
        }}
      >
        <div>
          <label className="label">Or enter the code</label>
          <input
            className="input font-mono"
            placeholder="VTD-XXXX-XXXX"
            value={typedCode}
            onChange={(event) => setTypedCode(event.target.value.toUpperCase())}
          />
        </div>
        <button
          className="btn-secondary"
          disabled={!typedCode.trim() || loading}
        >
          {loading ? 'Checking…' : 'Check code'}
        </button>
      </form>

      <VerdictChip state={state.type} label={state.label} />

      {result ? (
        <>
          {result.message ? (
            <p className="mt-3 text-xs text-slate-600">{result.message}</p>
          ) : null}
          <dl className="mt-3 space-y-1 text-xs text-slate-600">
            <Line label="Card code" value={result.normalised_code ?? result.payload ?? '—'} />
            <Line label="Owner" value={result.owner_name ?? '—'} />
            <Line label="Email" value={result.owner_email ?? '—'} />
            <Line
              label="Registered vehicles"
              value={result.vehicle_count ?? '—'}
            />
            <Line
              label="Account"
              value={
                result.is_active == null
                  ? '—'
                  : result.is_active
                    ? 'active'
                    : 'disabled'
              }
            />
          </dl>
        </>
      ) : null}
    </section>
  )
}


function PlateVerificationCard({ setPageError }) {
  const [plateInput, setPlateInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  async function handleVerify(event) {
    event.preventDefault()
    if (!plateInput.trim()) return
    setLoading(true)
    setPageError('')
    try {
      const payload = await api.verifyPlateInput(plateInput.trim())
      setResult(payload)
    } catch (err) {
      setPageError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="card">
      <h2 className="mb-3 text-sm font-semibold text-slate-900">Plate input</h2>
      <form className="space-y-2" onSubmit={handleVerify}>
        <div>
          <label className="label">License plate</label>
          <input
            className="input font-mono"
            placeholder="ABC-123XY"
            value={plateInput}
            onChange={(event) => setPlateInput(event.target.value)}
            required
          />
        </div>
        <button className="btn-primary" disabled={!plateInput.trim() || loading}>
          {loading ? 'Verifying…' : 'Verify plate'}
        </button>
      </form>

      <VerdictChip
        state={result ? (result.verified ? 'success' : 'danger') : 'idle'}
        label={result ? (result.verified ? 'VERIFIED' : 'UNVERIFIED') : 'Awaiting input'}
      />

      {result ? (
        <dl className="mt-3 space-y-1 text-xs text-slate-600">
          <Line label="Match type" value={result.match_type} />
          <Line label="Matched plate" value={result.matched_plate ?? '—'} />
          <Line label="Vehicle ID" value={result.matched_vehicle_id ?? '—'} />
          <Line label="Owner ID" value={result.owner_id ?? '—'} />
          <Line
            label="Fuzzy score"
            value={result.fuzzy_score != null ? String(result.fuzzy_score) : '—'}
          />
        </dl>
      ) : null}
    </section>
  )
}

function VehicleVerificationCard({ setPageError }) {
  const [file, setFile] = useState(null)
  const [expectedPlate, setExpectedPlate] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  async function handleVerify(event) {
    event.preventDefault()
    if (!file) return
    setLoading(true)
    setPageError('')
    try {
      const payload = await api.verifyVehicleImage(file, expectedPlate.trim() || null)
      setResult(payload)
    } catch (err) {
      setPageError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="card">
      <h2 className="mb-3 text-sm font-semibold text-slate-900">Vehicle image</h2>
      <form className="space-y-2" onSubmit={handleVerify}>
        <div>
          <label className="label">Vehicle image</label>
          <input
            type="file"
            accept="image/*"
            className="input block text-xs"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            required
          />
        </div>
        <div>
          <label className="label">Expected plate (optional)</label>
          <input
            className="input font-mono"
            placeholder="ABC-123XY"
            value={expectedPlate}
            onChange={(event) => setExpectedPlate(event.target.value)}
          />
        </div>
        <button className="btn-primary" disabled={!file || loading}>
          {loading ? 'Verifying…' : 'Verify vehicle'}
        </button>
      </form>

      <VerdictChip
        state={result ? (result.verified ? 'success' : 'danger') : 'idle'}
        label={result ? (result.verified ? 'VERIFIED' : 'UNVERIFIED') : 'Awaiting input'}
      />

      {result ? (
        <>
          <dl className="mt-3 space-y-1 text-xs text-slate-600">
            <Line label="Vehicles detected" value={result.vehicle_count} />
            <Line label="Plate detected" value={result.plate_detected ? 'Yes' : 'No'} />
            <Line
              label="Threat"
              value={`${result.threat.level} (${result.threat.score}/100)`}
            />
          </dl>

          {result.annotated_image ? (
            <img
              src={mediaUrl(result.annotated_image)}
              alt="Vehicle verification result"
              className="mt-3 w-full rounded-lg border border-slate-200"
            />
          ) : null}
        </>
      ) : null}
    </section>
  )
}

function VerdictChip({ state, label }) {
  const tone = {
    success: 'bg-green-50 text-green-700 ring-green-200',
    danger: 'bg-red-50 text-red-700 ring-red-200',
    warn: 'bg-amber-50 text-amber-700 ring-amber-200',
    idle: 'bg-slate-50 text-slate-600 ring-slate-200',
  }[state]

  return (
    <div
      className={`mt-3 inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase ring-1 ${tone}`}
    >
      {label}
    </div>
  )
}

function Line({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-medium text-slate-800">{value}</dd>
    </div>
  )
}

const PERSON_STATUS_LABELS = {
  authorized: 'Recognised owner',
  unauthorized: 'Not a registered owner',
  unknown: 'Face seen, identity uncertain',
  no_face: 'No face visible',
}

const OWNER_VERIFICATION_LABELS = {
  verified: 'Matches the expected owner',
  mismatch: 'Wrong person for that owner ID',
  not_recognized: 'No enrolled face matched',
  not_applicable: 'No expected owner given',
}

/**
 * Turn a face response into a verdict.
 *
 * Order matters. `no_face` is checked first because a photo with no face in it
 * has no similarity and no identity, and reporting it as a mismatch would be
 * actively misleading. An explicit expected-owner mismatch is checked before
 * generic recognition so that "you are not who I expected" is never softened
 * into merely "recognised as somebody".
 */
function humanVerdict(result) {
  if (!result) {
    return { type: 'idle', label: 'Awaiting input' }
  }

  if (result.person_status === 'no_face') {
    return { type: 'warn', label: 'NO FACE DETECTED' }
  }

  if (result.owner_verification === 'mismatch') {
    return { type: 'danger', label: 'MISMATCH' }
  }

  if (result.owner_verification === 'verified' || result.recognized) {
    return { type: 'success', label: 'VERIFIED' }
  }

  if (result.owner_verification === 'not_recognized') {
    return { type: 'danger', label: 'UNVERIFIED' }
  }

  if (result.person_status === 'unauthorized') {
    return { type: 'danger', label: 'UNVERIFIED' }
  }

  return { type: 'warn', label: 'UNSURE' }
}

/**
 * Turn a card-scan response into a verdict.
 *
 * The success state is gated on the account being *active* as well as the code
 * matching. A disabled owner's card is a card that must not open a gate, and
 * conflating the two would turn an administrative action into a security
 * bypass.
 */
function barcodeVerdict(result) {
  if (!result) {
    return { type: 'idle', label: 'Awaiting card' }
  }
  if (result.status === 'decoded' && result.is_active) {
    return { type: 'success', label: 'CARD VALID' }
  }
  if (result.status === 'decoded') {
    return { type: 'warn', label: 'ACCOUNT DISABLED' }
  }
  if (result.status === 'no_barcode') {
    return { type: 'warn', label: 'NO CARD READ' }
  }
  if (result.status === 'unknown_code') {
    return { type: 'danger', label: 'NOT REGISTERED' }
  }
  return { type: 'danger', label: 'NOT A MEMBER CARD' }
}

/**
 * Similarity against the decision threshold.
 *
 * A raw percentage hides the only thing that matters - which side of the
 * cutoff the score fell on and by how much - so the bar is scaled to the
 * threshold and the marker is drawn at it. Values below 0 are floored at 0 and
 * above 1 at 1 because a cosine similarity in a usable range never leaves that
 * band, and a stray negative would otherwise render a negative-width bar.
 */
function SimilarityBar({ similarity, threshold }) {
  if (similarity == null || threshold == null) return null

  const floor = Math.min(similarity, threshold) * 0.5
  const ceiling = Math.max(similarity, threshold) * 1.15
  const span = Math.max(ceiling - floor, 0.01)
  const pct = (value) => Math.min(100, Math.max(0, ((value - floor) / span) * 100))
  const passed = similarity >= threshold

  return (
    <div className="mt-3">
      <div className="relative h-2 w-full rounded-full bg-slate-100">
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${
            passed ? 'bg-green-500' : 'bg-amber-500'
          }`}
          style={{ width: `${pct(similarity)}%` }}
        />
        <div
          className="absolute inset-y-[-3px] w-0.5 bg-slate-700"
          style={{ left: `${pct(threshold)}%` }}
          title={`threshold ${(threshold * 100).toFixed(0)}%`}
        />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-slate-400">
        <span>score {(similarity * 100).toFixed(0)}%</span>
        <span>cutoff {(threshold * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}
