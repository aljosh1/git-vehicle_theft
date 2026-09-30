/**
 * API client.
 *
 * One place that knows how to talk to the backend: it attaches the JWT to every
 * request and force-logs-out on a 401. Centralising that means no component has
 * to think about tokens, and an expired session cannot leave the UI in a broken
 * half-authenticated state.
 */

const TOKEN_KEY = 'vtds_token'
const USER_KEY = 'vtds_user'

/**
 * Base URL every request is prefixed with.
 *
 * Empty by default, which keeps relative URLs (`/api/...`). That is what you
 * want locally, because `vite.config.js` proxies `/api` and `/media` to the
 * backend so the browser only ever sees one origin.
 *
 * On a static host such as Vercel there is no dev-server proxy, so set
 * `VITE_API_BASE_URL` (Vercel -> Settings -> Environment Variables, or a
 * `.env.production` file) to the public origin of the backend, e.g.
 * `https://api.example.com`. Leave it empty if you rely on the `vercel.json`
 * rewrites instead.
 */
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')

/** Prefix a root-relative path with the configured API base. */
function apiUrl(path) {
  if (!path) return path
  if (/^https?:\/\//i.test(path)) return path
  return `${API_BASE}${path}`
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function getStoredUser() {
  try {
    const raw = localStorage.getItem(USER_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function storeSession(token, user) {
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

/**
 * Thin fetch wrapper.
 *
 * Throws an Error carrying the backend's `detail` message, so components can
 * surface a useful message instead of a bare status code.
 */
async function request(path, { method = 'GET', body, isForm = false } = {}) {
  const headers = {}
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  if (body && !isForm) headers['Content-Type'] = 'application/json'

  const response = await fetch(apiUrl(path), {
    method,
    headers,
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  })

  // An expired or revoked token must drop the session, or every subsequent
  // request fails silently and the UI looks merely "empty".
  if (response.status === 401) {
    clearSession()
    window.location.href = '/login'
    throw new Error('Session expired - please sign in again')
  }

  if (response.status === 204) return null

  const text = await response.text()
  const data = text ? JSON.parse(text) : null

  if (!response.ok) {
    const detail = data?.detail
    throw new Error(
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg ?? JSON.stringify(d)).join('; ')
          : `Request failed (${response.status})`,
    )
  }
  return data
}

export const api = {
  // --- auth --------------------------------------------------------------
  login: (email, password) =>
    request('/api/auth/login', { method: 'POST', body: { email, password } }),
  register: (payload) =>
    request('/api/auth/register', { method: 'POST', body: payload }),
  me: () => request('/api/auth/me'),
  updateMe: (payload) => request('/api/auth/me', { method: 'PATCH', body: payload }),

  // --- owners ------------------------------------------------------------
  listOwners: (search = '') =>
    request(`/api/owners${search ? `?search=${encodeURIComponent(search)}` : ''}`),
  createOwner: (payload) => request('/api/owners', { method: 'POST', body: payload }),
  updateOwner: (id, payload) =>
    request(`/api/owners/${id}`, { method: 'PATCH', body: payload }),
  deleteOwner: (id) => request(`/api/owners/${id}`, { method: 'DELETE' }),

  // --- vehicles ----------------------------------------------------------
  listVehicles: (search = '') =>
    request(`/api/vehicles${search ? `?search=${encodeURIComponent(search)}` : ''}`),
  createVehicle: (payload) => request('/api/vehicles', { method: 'POST', body: payload }),
  updateVehicle: (id, payload) =>
    request(`/api/vehicles/${id}`, { method: 'PATCH', body: payload }),
  deleteVehicle: (id) => request(`/api/vehicles/${id}`, { method: 'DELETE' }),
  uploadVehicleImage: (id, file) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/vehicles/${id}/image`, {
      method: 'POST',
      body: form,
      isForm: true,
    })
  },

  // --- faces -------------------------------------------------------------
  listFaces: (ownerId) => request(`/api/faces/${ownerId}`),
  enrollFaces: (ownerId, files) => {
    const form = new FormData()
    for (const file of files) form.append('files', file)
    return request(`/api/faces/${ownerId}/enroll`, {
      method: 'POST',
      body: form,
      isForm: true,
    })
  },
  deleteFace: (embeddingId) =>
    request(`/api/faces/embedding/${embeddingId}`, { method: 'DELETE' }),

  // --- detections + alerts -----------------------------------------------
  listDetections: (params = '') => request(`/api/detections${params}`),
  liveDetections: (cameraId = 'cam-0') =>
    request(`/api/detections/live?camera_id=${encodeURIComponent(cameraId)}`),
  listAlerts: (params = '') => request(`/api/alerts${params}`),
  updateAlert: (id, payload) =>
    request(`/api/alerts/${id}`, { method: 'PATCH', body: payload }),
  testEmail: () => request('/api/alerts/test-email', { method: 'POST' }),
  testSms: () => request('/api/alerts/test-sms', { method: 'POST' }),
  alertChannels: () => request('/api/alerts/channels'),

  // --- membership cards / barcodes ---------------------------------------
  getBarcode: (ownerId) => request(`/api/barcode/${ownerId}`),
  regenerateBarcode: (ownerId) =>
    request(`/api/barcode/regenerate/${ownerId}`, { method: 'POST' }),
  cardDownloadUrl: (ownerId) => `/api/barcode/${ownerId}/card`,
  lookupCode: (code) =>
    request(`/api/barcode/lookup/${encodeURIComponent(code)}`),
  scanBarcode: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/api/barcode/scan', { method: 'POST', body: form, isForm: true })
  },
  verifyCode: (code) => {
    const form = new FormData()
    form.append('code_override', code)
    return request('/api/barcode/scan', { method: 'POST', body: form, isForm: true })
  },

  // --- stream control ----------------------------------------------------
  startStream: (payload) => request('/api/stream/start', { method: 'POST', body: payload }),
  stopStream: (cameraId) =>
    request('/api/stream/stop', { method: 'POST', body: { camera_id: cameraId } }),
  streamStatus: () => request('/api/stream/status'),
  analyseImage: (file, dispatchAlert = false) => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/stream/analyse-image?dispatch_alert=${dispatchAlert}`, {
      method: 'POST',
      body: form,
      isForm: true,
    })
  },
  verifyImagePlate: (file, plateInput) => {
    const form = new FormData()
    form.append('file', file)
    form.append('plate_input', plateInput)
    return request('/api/stream/verify-image-plate', {
      method: 'POST',
      body: form,
      isForm: true,
    })
  },
  verifyFaceImage: (file, expectedOwnerId = null) => {
    const form = new FormData()
    form.append('file', file)
    if (expectedOwnerId != null) {
      form.append('expected_owner_id', String(expectedOwnerId))
    }
    return request('/api/stream/verify-face-image', {
      method: 'POST',
      body: form,
      isForm: true,
    })
  },
  verifyPlateInput: (plateInput) => {
    const form = new FormData()
    form.append('plate_input', plateInput)
    return request('/api/stream/verify-plate-input', {
      method: 'POST',
      body: form,
      isForm: true,
    })
  },
  verifyVehicleImage: (file, expectedPlate = null) => {
    const form = new FormData()
    form.append('file', file)
    if (expectedPlate) {
      form.append('expected_plate', expectedPlate)
    }
    return request('/api/stream/verify-vehicle-image', {
      method: 'POST',
      body: form,
      isForm: true,
    })
  },
  uploadVideo: (file, cameraId = 'upload-0') => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/stream/upload?camera_id=${encodeURIComponent(cameraId)}`, {
      method: 'POST',
      body: form,
      isForm: true,
    })
  },

  // --- stats + system ----------------------------------------------------
  stats: () => request('/api/stats'),
  timeseries: (days = 7) => request(`/api/stats/timeseries?days=${days}`),
  trainingStatus: () => request('/api/training/status'),
  info: () => request('/api/info'),
}

/**
 * URL for the MJPEG live feed.
 *
 * The token goes in the query string because a browser will not attach an
 * Authorization header to an `<img src>` request. See the matching note on
 * `get_user_from_query_token` in backend/api/deps.py.
 */
export function mjpegUrl(cameraId = 'cam-0') {
  return apiUrl(
    `/api/stream/mjpeg?camera_id=${encodeURIComponent(cameraId)}&token=${getToken()}`,
  )
}

/** Resolve a stored media path ("evidence/foo.jpg") to a servable URL. */
export function mediaUrl(path) {
  if (!path) return null
  return apiUrl(`/media/${path.replace(/^\/+/, '')}`)
}

/**
 * Authenticated URL for a binary endpoint the browser fetches directly
 * (a card download). The token rides in the query string because an `<a href>`
 * cannot carry an Authorization header.
 */
export function authUrl(path) {
  return `${apiUrl(path)}${path.includes('?') ? '&' : '?'}token=${encodeURIComponent(getToken() ?? '')}`
}
