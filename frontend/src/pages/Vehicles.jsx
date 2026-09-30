/** Vehicle management: register, edit, remove, upload a photo. */
import { useCallback, useEffect, useState } from 'react'
import { api, mediaUrl } from '../api/client'
import {
  EmptyState,
  ErrorBanner,
  formatDateTime,
  Modal,
  Spinner,
} from '../components/ui.jsx'
import { useAuth } from '../context/AuthContext.jsx'

const BLANK = {
  plate_display: '',
  make: '',
  model: '',
  year: '',
  color: '',
  vehicle_type: 'car',
  vin: '',
  notes: '',
}

export default function Vehicles() {
  const { isAdmin, user } = useAuth()
  const [vehicles, setVehicles] = useState([])
  const [owners, setOwners] = useState([])
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [editing, setEditing] = useState(null)
  const [form, setForm] = useState(BLANK)
  const [vehicleImage, setVehicleImage] = useState(null)
  const [ownerFaces, setOwnerFaces] = useState([])
  const [saving, setSaving] = useState(false)

  const load = useCallback(async () => {
    try {
      const data = await api.listVehicles(search)
      setVehicles(data)
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

  useEffect(() => {
    if (!isAdmin) return
    api.listOwners().then(setOwners).catch(() => {})
  }, [isAdmin])

  function openCreate() {
    setForm({ ...BLANK, owner_id: isAdmin ? (owners[0]?.id ?? user.id) : user.id })
    setVehicleImage(null)
    setOwnerFaces([])
    setEditing('new')
  }

  function openEdit(vehicle) {
    setForm({
      plate_display: vehicle.plate_display,
      make: vehicle.make ?? '',
      model: vehicle.model ?? '',
      year: vehicle.year ?? '',
      color: vehicle.color ?? '',
      vehicle_type: vehicle.vehicle_type,
      vin: vehicle.vin ?? '',
      notes: vehicle.notes ?? '',
      is_flagged_stolen: vehicle.is_flagged_stolen,
      owner_id: vehicle.owner_id,
    })
    setEditing(vehicle)
  }

  async function handleSave(event) {
    event.preventDefault()
    if (editing === 'new' && (!vehicleImage || ownerFaces.length === 0)) {
      setError('A vehicle image and at least one clear owner face photo are required.')
      return
    }

    setSaving(true)
    setError('')
    let createdVehicle = null
    try {
      if (editing === 'new' && vehicleImage && form.plate_display) {
        const verification = await api.verifyImagePlate(vehicleImage, form.plate_display)
        if (verification.verification === 'no_plate_detected') {
          throw new Error(
            'Plate could not be read from the uploaded vehicle image. Use a clearer image.'
          )
        }
        if (verification.verification === 'mismatch') {
          throw new Error(
            `Image plate (${verification.detected_plate ?? 'unreadable'}) does not match typed plate (${form.plate_display}).`
          )
        }
      }

      // Empty strings must become null, not "" — the backend validates year as
      // an integer and would reject an empty string.
      const payload = Object.fromEntries(
        Object.entries(form).map(([key, value]) => [
          key,
          value === '' ? null : key === 'year' ? Number(value) : value,
        ]),
      )
      if (editing === 'new') {
        createdVehicle = await api.createVehicle({
          ...payload,
          owner_id: payload.owner_id ?? user.id,
        })
        await api.uploadVehicleImage(createdVehicle.id, vehicleImage)
        const faceResult = await api.enrollFaces(createdVehicle.owner_id, ownerFaces)
        if (faceResult.enrolled === 0) {
          throw new Error(faceResult.messages.join('; ') || 'no owner face was enrolled')
        }
      } else {
        await api.updateVehicle(editing.id, payload)
      }
      setEditing(null)
      await load()
    } catch (err) {
      if (createdVehicle) {
        setEditing(null)
        await load()
        setError(
          `Vehicle ${createdVehicle.plate_display} was registered, but its evidence upload did not complete: ${err.message}`,
        )
      } else {
        setError(err.message)
      }
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete(vehicle) {
    if (!window.confirm(`Remove ${vehicle.plate_display} permanently?`)) return
    try {
      await api.deleteVehicle(vehicle.id)
      await load()
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleImage(vehicle, file) {
    if (!file) return
    try {
      await api.uploadVehicleImage(vehicle.id, file)
      await load()
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleOwnerFaces(vehicle, files) {
    const selected = Array.from(files ?? []).slice(0, 5)
    if (selected.length === 0) return
    setError('')
    try {
      const result = await api.enrollFaces(vehicle.owner_id, selected)
      if (result.enrolled === 0) {
        throw new Error(result.messages.join('; ') || 'No owner face was enrolled')
      }
      if (result.failed > 0) {
        setError(result.messages.join('; '))
      }
      await load()
    } catch (err) {
      setError(err.message)
    }
  }

  if (loading) return <Spinner label="Loading vehicles…" />

  return (
    <div>
      <header className="mb-5 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Vehicles</h1>
          <p className="text-sm text-slate-500">
            {isAdmin ? 'All registered vehicles' : 'Your registered vehicles'}
          </p>
        </div>
        <button onClick={openCreate} className="btn-primary">
          Register vehicle
        </button>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError('')} />

      <div className="mb-4">
        <input
          className="input max-w-sm"
          placeholder="Search plate, make or model…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>

      <div className="card overflow-x-auto p-0">
        {vehicles.length === 0 ? (
          <EmptyState
            title="No vehicles registered"
            hint="Register a vehicle so the system can recognise it as authorised"
          />
        ) : (
          <table className="w-full">
            <thead className="border-b border-slate-200 bg-slate-50">
              <tr>
                <th className="th">Vehicle photo</th>
                <th className="th">Owner face</th>
                <th className="th">Plate</th>
                <th className="th">Vehicle</th>
                {isAdmin && <th className="th">Owner</th>}
                <th className="th">Registered</th>
                <th className="th">Status</th>
                <th className="th text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {vehicles.map((vehicle) => (
                <tr key={vehicle.id} className="hover:bg-slate-50">
                  <td className="td">
                    {vehicle.vehicle_image ? (
                      <img
                        src={mediaUrl(vehicle.vehicle_image)}
                        alt={vehicle.plate_display}
                        className="h-10 w-14 rounded object-cover ring-1 ring-slate-200"
                      />
                    ) : (
                      <label className="flex h-10 w-14 cursor-pointer items-center justify-center rounded bg-slate-100 text-[10px] text-slate-400 hover:bg-slate-200">
                        Upload
                        <input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(event) =>
                            handleImage(vehicle, event.target.files?.[0])
                          }
                        />
                      </label>
                    )}
                  </td>
                  <td className="td">
                    <div className="flex min-w-40 items-center gap-2">
                      {vehicle.owner_face_image ? (
                        <img
                          src={mediaUrl(vehicle.owner_face_image)}
                          alt={`${vehicle.owner_name ?? 'Owner'} enrolled face`}
                          className="h-10 w-10 rounded object-cover ring-1 ring-slate-200"
                        />
                      ) : (
                        <div className="flex h-10 w-10 items-center justify-center rounded bg-amber-50 text-[10px] text-amber-700 ring-1 ring-amber-200">
                          None
                        </div>
                      )}
                      <div>
                        <div className="text-xs font-medium">
                          {vehicle.owner_name ?? 'Vehicle owner'}
                        </div>
                        <div
                          className={`text-[10px] ${
                            vehicle.owner_face_count > 0 ? 'text-green-600' : 'text-amber-600'
                          }`}
                        >
                          {vehicle.owner_face_count > 0
                            ? `${vehicle.owner_face_count} face${
                                vehicle.owner_face_count === 1 ? '' : 's'
                              } linked`
                            : 'No face linked'}
                        </div>
                        <label className="cursor-pointer text-[10px] text-blue-600 hover:underline">
                          {vehicle.owner_face_count > 0 ? 'Add face' : 'Enroll face'}
                          <input
                            type="file"
                            accept="image/*"
                            multiple
                            className="hidden"
                            onChange={(event) => {
                              handleOwnerFaces(vehicle, event.target.files)
                              event.target.value = ''
                            }}
                          />
                        </label>
                      </div>
                    </div>
                  </td>
                  <td className="td font-mono font-semibold">
                    {vehicle.plate_display}
                  </td>
                  <td className="td">
                    <div>
                      {[vehicle.year, vehicle.make, vehicle.model]
                        .filter(Boolean)
                        .join(' ') || '—'}
                    </div>
                    <div className="text-xs capitalize text-slate-400">
                      {vehicle.color} · {vehicle.vehicle_type}
                    </div>
                  </td>
                  {isAdmin && (
                    <td className="td">
                      <div>{vehicle.owner_name ?? '—'}</div>
                      <div className="text-xs text-slate-400">
                        {vehicle.owner_phone}
                      </div>
                    </td>
                  )}
                  <td className="td whitespace-nowrap text-xs text-slate-500">
                    {formatDateTime(vehicle.registration_date)}
                  </td>
                  <td className="td">
                    {vehicle.is_flagged_stolen ? (
                      <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs font-semibold text-red-700 ring-1 ring-red-200">
                        STOLEN
                      </span>
                    ) : (
                      <span className="text-xs text-green-600">Active</span>
                    )}
                  </td>
                  <td className="td text-right">
                    <button
                      onClick={() => openEdit(vehicle)}
                      className="text-xs text-blue-600 hover:underline"
                    >
                      Edit
                    </button>
                    <button
                      onClick={() => handleDelete(vehicle)}
                      className="ml-3 text-xs text-red-600 hover:underline"
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {editing && (
        <Modal
          title={editing === 'new' ? 'Register a vehicle' : `Edit ${editing.plate_display}`}
          onClose={() => setEditing(null)}
        >
          <form onSubmit={handleSave} className="space-y-3">
            <div>
              <label className="label">License plate *</label>
              <input
                className="input font-mono"
                value={form.plate_display}
                onChange={(event) =>
                  setForm({ ...form, plate_display: event.target.value })
                }
                placeholder="ABC-123XY"
                required
              />
            </div>

            {editing === 'new' && (
              <div className="grid gap-3 border-y border-slate-200 py-3 sm:grid-cols-2">
                <div>
                  <label className="label">Vehicle image *</label>
                  <input
                    className="input block text-xs"
                    type="file"
                    accept="image/*"
                    onChange={(event) => setVehicleImage(event.target.files?.[0] ?? null)}
                    required
                  />
                  <p className="mt-1 text-xs text-slate-400">
                    Use a clear image showing the vehicle and plate.
                  </p>
                </div>
                <div>
                  <label className="label">Owner face photos *</label>
                  <input
                    className="input block text-xs"
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={(event) =>
                      setOwnerFaces(Array.from(event.target.files ?? []).slice(0, 5))
                    }
                    required
                  />
                  <p className="mt-1 text-xs text-slate-400">
                    Select up to 5 clear photos containing one owner face each.
                  </p>
                </div>
              </div>
            )}

            {isAdmin && owners.length > 0 && (
              <div>
                <label className="label">Owner</label>
                <select
                  className="input"
                  value={form.owner_id ?? ''}
                  onChange={(event) =>
                    setForm({ ...form, owner_id: Number(event.target.value) })
                  }
                >
                  {owners.map((owner) => (
                    <option key={owner.id} value={owner.id}>
                      {owner.full_name} ({owner.email})
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="label">Make</label>
                <input
                  className="input"
                  value={form.make}
                  onChange={(event) => setForm({ ...form, make: event.target.value })}
                  placeholder="Toyota"
                />
              </div>
              <div>
                <label className="label">Model</label>
                <input
                  className="input"
                  value={form.model}
                  onChange={(event) => setForm({ ...form, model: event.target.value })}
                  placeholder="Camry"
                />
              </div>
              <div>
                <label className="label">Year</label>
                <input
                  className="input"
                  type="number"
                  value={form.year}
                  onChange={(event) => setForm({ ...form, year: event.target.value })}
                  placeholder="2019"
                />
              </div>
              <div>
                <label className="label">Colour</label>
                <input
                  className="input"
                  value={form.color}
                  onChange={(event) => setForm({ ...form, color: event.target.value })}
                  placeholder="Silver"
                />
              </div>
            </div>

            <div>
              <label className="label">Type</label>
              <select
                className="input"
                value={form.vehicle_type}
                onChange={(event) =>
                  setForm({ ...form, vehicle_type: event.target.value })
                }
              >
                <option value="car">Car</option>
                <option value="motorcycle">Motorcycle</option>
                <option value="truck">Truck</option>
                <option value="bus">Bus</option>
                <option value="other">Other</option>
              </select>
            </div>

            {editing !== 'new' && (
              <label className="flex items-center gap-2 rounded-lg bg-red-50 px-3 py-2 text-sm ring-1 ring-red-200">
                <input
                  type="checkbox"
                  checked={Boolean(form.is_flagged_stolen)}
                  onChange={(event) =>
                    setForm({ ...form, is_flagged_stolen: event.target.checked })
                  }
                />
                <span className="text-red-800">
                  Report this vehicle stolen — every sighting alerts as CRITICAL
                </span>
              </label>
            )}

            {editing === 'new' && (
              <div className="rounded bg-blue-50 px-3 py-2 text-xs text-blue-800 ring-1 ring-blue-200">
                The plate is linked to the selected owner. Verification succeeds only
                when this plate is read and an enrolled face for that same owner is matched.
              </div>
            )}

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setEditing(null)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button type="submit" className="btn-primary" disabled={saving}>
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </form>
        </Modal>
      )}
    </div>
  )
}
