/** Owner management + face enrolment (administrators only). */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api, authUrl, mediaUrl } from '../api/client'
import {
  EmptyState,
  ErrorBanner,
  formatDateTime,
  Modal,
  Spinner,
} from '../components/ui.jsx'

const BLANK = {
  full_name: '',
  email: '',
  phone_number: '',
  address: '',
  password: '',
  role: 'owner',
}

export default function Owners() {
  const [owners, setOwners] = useState([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState(BLANK)
  const [saving, setSaving] = useState(false)
  const [faceOwner, setFaceOwner] = useState(null)
  const [cardOwner, setCardOwner] = useState(null)

  const load = useCallback(async () => {
    try {
      setOwners(await api.listOwners(search))
      setError('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [search])

  useEffect(() => {
    load()
  }, [load])

  async function handleCreate(event) {
    event.preventDefault()
    setSaving(true)
    setError('')
    try {
      await api.createOwner(form)
      setCreating(false)
      setForm(BLANK)
      await load()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete(owner) {
    if (
      !window.confirm(
        `Delete ${owner.full_name}? Their vehicles and face data are removed too.`,
      )
    )
      return
    try {
      await api.deleteOwner(owner.id)
      await load()
    } catch (err) {
      setError(err.message)
    }
  }

  if (loading) return <Spinner label="Loading owners…" />

  return (
    <div>
      <header className="mb-5 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Owners</h1>
          <p className="text-sm text-slate-500">
            Accounts, vehicles and enrolled faces
          </p>
        </div>
        <button onClick={() => setCreating(true)} className="btn-primary">
          Add owner
        </button>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError('')} />

      <div className="mb-4">
        <input
          className="input max-w-sm"
          placeholder="Search name or email…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>

      <div className="card overflow-x-auto p-0">
        {owners.length === 0 ? (
          <EmptyState title="No accounts found" />
        ) : (
          <table className="w-full">
            <thead className="border-b border-slate-200 bg-slate-50">
              <tr>
                <th className="th">Name</th>
                <th className="th">Contact</th>
                <th className="th">Role</th>
                <th className="th">Vehicles</th>
                <th className="th">Faces</th>
                <th className="th">Joined</th>
                <th className="th">Card code</th>
                <th className="th text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {owners.map((owner) => (
                <tr key={owner.id} className="hover:bg-slate-50">
                  <td className="td font-medium">{owner.full_name}</td>
                  <td className="td">
                    <div className="text-xs">{owner.email}</div>
                    <div className="text-xs text-slate-400">{owner.phone_number}</div>
                  </td>
                  <td className="td">
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-medium capitalize ${
                        owner.role === 'admin'
                          ? 'bg-purple-50 text-purple-700 ring-1 ring-purple-200'
                          : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      {owner.role}
                    </span>
                  </td>
                  <td className="td tabular-nums">{owner.vehicle_count}</td>
                  <td className="td">
                    <span
                      className={
                        owner.face_count === 0 ? 'text-amber-600' : 'tabular-nums'
                      }
                    >
                      {owner.face_count === 0 ? 'none' : owner.face_count}
                    </span>
                  </td>
                  <td className="td whitespace-nowrap text-xs text-slate-500">
                    {formatDateTime(owner.created_at)}
                  </td>
                  <td className="td font-mono text-[11px] text-slate-500">
                    {owner.owner_code}
                  </td>
                  <td className="td text-right">
                    <button
                      onClick={() => setFaceOwner(owner)}
                      className="text-xs text-blue-600 hover:underline"
                    >
                      Faces
                    </button>
                    <button
                      onClick={() => setCardOwner(owner)}
                      className="ml-3 text-xs text-blue-600 hover:underline"
                    >
                      Card
                    </button>
                    <button
                      onClick={() => handleDelete(owner)}
                      className="ml-3 text-xs text-red-600 hover:underline"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {creating && (
        <Modal title="Add an account" onClose={() => setCreating(false)}>
          <form onSubmit={handleCreate} className="space-y-3">
            <div>
              <label className="label">Full name *</label>
              <input
                className="input"
                value={form.full_name}
                onChange={(event) =>
                  setForm({ ...form, full_name: event.target.value })
                }
                required
                minLength={2}
              />
            </div>
            <div>
              <label className="label">Email *</label>
              <input
                className="input"
                type="email"
                value={form.email}
                onChange={(event) => setForm({ ...form, email: event.target.value })}
                required
              />
            </div>
            <div>
              <label className="label">Phone</label>
              <input
                className="input"
                value={form.phone_number}
                onChange={(event) =>
                  setForm({ ...form, phone_number: event.target.value })
                }
                placeholder="+234…"
              />
            </div>
            <div>
              <label className="label">Password *</label>
              <input
                className="input"
                type="password"
                value={form.password}
                onChange={(event) =>
                  setForm({ ...form, password: event.target.value })
                }
                required
                minLength={8}
              />
              <p className="mt-1 text-xs text-slate-400">
                At least 8 characters, with a letter and a digit
              </p>
            </div>
            <div>
              <label className="label">Role</label>
              <select
                className="input"
                value={form.role}
                onChange={(event) => setForm({ ...form, role: event.target.value })}
              >
                <option value="owner">Vehicle owner</option>
                <option value="admin">Administrator</option>
              </select>
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setCreating(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button type="submit" className="btn-primary" disabled={saving}>
                {saving ? 'Creating…' : 'Create'}
              </button>
            </div>
          </form>
        </Modal>
      )}

      {faceOwner && (
        <FaceEnrollment owner={faceOwner} onClose={() => setFaceOwner(null)} onChange={load} />
      )}

      {cardOwner && (
        <MembershipCard owner={cardOwner} onClose={() => setCardOwner(null)} />
      )}
    </div>
  )
}

/**
 * Membership card panel.
 *
 * Shows the owner's unique code alongside the rendered card, its linear barcode
 * and its QR. The code is displayed in selectable monospace because a card that
 * cannot be reprinted is a card that cannot be reissued, and an administrator
 * may legitimately need the text to hand-type it at a gate where the scanner is
 * out of service.
 */
function MembershipCard({ owner, onClose }) {
  const [card, setCard] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    let cancelled = false
    api
      .getBarcode(owner.id)
      .then((data) => {
        if (!cancelled) setCard(data)
      })
      .catch((err) => {
        if (!cancelled) setError(err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [owner.id])

  async function handleRegenerate() {
    setBusy(true)
    setError('')
    try {
      // The cache-buster is deliberate: the image URL is keyed on the owner
      // code, which does not change on a re-render, so the browser would
      // otherwise serve the stale card it already has.
      setCard({ ...(await api.regenerateBarcode(owner.id)), _v: Date.now() })
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(card.owner_code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setError('Could not copy to the clipboard - select the code and copy manually')
    }
  }

  const bust = card?._v ? `?v=${card._v}` : ''

  return (
    <Modal title={`Membership card — ${owner.full_name}`} onClose={onClose} wide>
      <ErrorBanner message={error} onDismiss={() => setError('')} />

      {loading ? (
        <p className="text-sm text-slate-500">Rendering card…</p>
      ) : card ? (
        <>
          <div className="mb-4 flex items-center justify-between gap-3 rounded-lg bg-slate-50 p-3">
            <div>
              <div className="text-xs text-slate-500">Owner code</div>
              <div className="select-all font-mono text-lg font-semibold tracking-wider text-slate-900">
                {card.owner_code}
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={handleCopy} className="btn-secondary">
                {copied ? 'Copied ✓' : 'Copy'}
              </button>
              <button
                onClick={handleRegenerate}
                className="btn-secondary"
                disabled={busy}
              >
                {busy ? 'Rebuilding…' : 'Rebuild'}
              </button>
              <a
                href={authUrl(api.cardDownloadUrl(owner.id))}
                className="btn-primary"
              >
                Download
              </a>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <figure>
              <img
                src={mediaUrl(card.card_path) + bust}
                alt={`Membership card for ${card.full_name}`}
                className="w-full rounded-lg ring-1 ring-slate-200"
              />
              <figcaption className="mt-1 text-[10px] text-slate-400">
                Printable card — QR and Code 128
              </figcaption>
            </figure>
            <div className="space-y-3">
              <figure>
                <img
                  src={mediaUrl(card.barcode_path) + bust}
                  alt="Code 128 barcode"
                  className="w-full rounded-lg bg-white ring-1 ring-slate-200"
                />
                <figcaption className="mt-1 text-[10px] text-slate-400">
                  Code 128 — scannable by any handheld reader
                </figcaption>
              </figure>
              <figure>
                <img
                  src={mediaUrl(card.qr_path) + bust}
                  alt="QR code"
                  className="mx-auto w-40 rounded-lg bg-white ring-1 ring-slate-200"
                />
                <figcaption className="mt-1 text-center text-[10px] text-slate-400">
                  QR — for phone cameras
                </figcaption>
              </figure>
            </div>
          </div>

          <p className="mt-4 text-xs text-slate-500">
            This code is unique to {card.full_name} and is what a guard scans to
            confirm ownership. It is separate from face recognition: the card is
            a deterministic check, the face is a probabilistic one. If the owner
            is deleted, the code is released and any card still in circulation
            stops verifying.
          </p>
        </>
      ) : null}
    </Modal>
  )
}

/**
 * Face enrolment panel.
 *
 * Recognition quality is bounded by enrolment quality, so this reports what
 * happened to each uploaded file rather than silently accepting photos the
 * detector could not use.
 */
function FaceEnrollment({ owner, onClose, onChange }) {
  const [faces, setFaces] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [messages, setMessages] = useState([])
  const fileInput = useRef(null)

  const load = useCallback(async () => {
    try {
      setFaces(await api.listFaces(owner.id))
    } catch (err) {
      setError(err.message)
    }
  }, [owner.id])

  useEffect(() => {
    load()
  }, [load])

  async function handleUpload(event) {
    const files = Array.from(event.target.files ?? [])
    if (files.length === 0) return
    setBusy(true)
    setError('')
    setMessages([])
    try {
      const result = await api.enrollFaces(owner.id, files)
      setMessages(result.messages)
      await load()
      await onChange()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function handleDelete(id) {
    try {
      await api.deleteFace(id)
      await load()
      await onChange()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <Modal title={`Face enrolment — ${owner.full_name}`} onClose={onClose} wide>
      <ErrorBanner message={error} onDismiss={() => setError('')} />

      <p className="mb-3 text-xs text-slate-500">
        Upload 3–5 clear photos with varied angle and lighting. Each must contain
        exactly one face. These embeddings are what let the system tell this owner
        apart from an intruder.
      </p>

      <button
        onClick={() => fileInput.current?.click()}
        className="btn-primary mb-4"
        disabled={busy}
      >
        {busy ? 'Enrolling…' : 'Upload photos'}
      </button>
      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={handleUpload}
      />

      {messages.length > 0 && (
        <ul className="mb-4 space-y-1 rounded-lg bg-slate-50 p-3 text-xs">
          {messages.map((message, index) => (
            <li
              key={index}
              className={
                message.includes('enrolled') ? 'text-green-700' : 'text-amber-700'
              }
            >
              {message}
            </li>
          ))}
        </ul>
      )}

      {faces.length === 0 ? (
        <EmptyState
          title="No faces enrolled"
          hint="Without an enrolled face this owner is treated as an unknown person"
        />
      ) : (
        <div className="grid grid-cols-4 gap-3">
          {faces.map((face) => (
            <div key={face.id} className="group relative">
              {face.source_image ? (
                <img
                  src={mediaUrl(face.source_image)}
                  alt="Enrolled face"
                  className="aspect-square w-full rounded-lg object-cover ring-1 ring-slate-200"
                />
              ) : (
                <div className="flex aspect-square w-full items-center justify-center rounded-lg bg-slate-100 text-xs text-slate-400">
                  no image
                </div>
              )}
              <div className="mt-1 text-[10px] text-slate-400">
                {face.model_name} · {face.dim}d
              </div>
              <button
                onClick={() => handleDelete(face.id)}
                className="absolute right-1 top-1 hidden rounded bg-red-600 px-1.5
                           py-0.5 text-[10px] text-white group-hover:block"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </Modal>
  )
}
