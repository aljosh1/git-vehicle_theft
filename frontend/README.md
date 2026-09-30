# Frontend - React + Tailwind dashboard

Placeholder for Step 8. Scaffolding command:

```powershell
cd frontend
npm create vite@latest . -- --template react
npm install
npm install -D tailwindcss postcss autoprefixer && npx tailwindcss init -p
npm install react-router-dom axios recharts lucide-react
npm run dev          # http://localhost:5173
```

`http://localhost:5173` is already in the backend's `CORS_ORIGINS` default.

## Planned structure

```
src/
├── api/client.js          Axios instance; attaches the JWT, handles 401 -> login
├── context/AuthContext.jsx
├── components/
│   ├── Layout.jsx         Sidebar + header shell
│   ├── StatCard.jsx       Dashboard KPI tile
│   ├── ThreatBadge.jsx    Colour-coded threat level
│   ├── LiveFeed.jsx       <img src=".../api/stream/mjpeg"> MJPEG viewer
│   └── DataTable.jsx      Sortable/filterable table
├── pages/
│   ├── Login.jsx
│   ├── Dashboard.jsx      Live feed, vehicle count, threat count, recent alerts
│   ├── LiveMonitoring.jsx Full-screen feed + real-time detection panel
│   ├── Vehicles.jsx       Add / edit / remove vehicles
│   ├── Owners.jsx         Owner management + face enrolment
│   └── Alerts.jsx         Alert history, acknowledge / resolve
└── dashboard/             Chart widgets (detections & alerts over time)
```

## Why MJPEG for the live feed

The backend streams `multipart/x-mixed-replace` JPEG frames, which a plain
`<img>` tag renders with no client-side decoding library and no WebRTC
signalling. For a single-viewer surveillance dashboard on a LAN that is the
simplest thing that works; WebRTC would only be worth its complexity for
many concurrent remote viewers.
