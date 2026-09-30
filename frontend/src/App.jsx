/** Route table + auth guard. */
import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import { useAuth } from './context/AuthContext.jsx'
import Alerts from './pages/Alerts.jsx'
import Dashboard from './pages/Dashboard.jsx'
import LiveMonitoring from './pages/LiveMonitoring.jsx'
import Login from './pages/Login.jsx'
import Owners from './pages/Owners.jsx'
import TrainingMonitor from './pages/TrainingMonitor.jsx'
import Verification from './pages/Verification.jsx'
import Vehicles from './pages/Vehicles.jsx'

/** Redirects to /login when there is no session; admin-only routes add a role check. */
function Protected({ children, adminOnly = false }) {
  const { isAuthenticated, isAdmin } = useAuth()
  if (!isAuthenticated) return <Navigate to="/login" replace />
  if (adminOnly && !isAdmin) return <Navigate to="/" replace />
  return children
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/"
        element={
          <Protected>
            <Layout />
          </Protected>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="live" element={<LiveMonitoring />} />
        <Route path="live-camera" element={<LiveMonitoring />} />
        <Route path="verify" element={<Verification />} />
        <Route path="vehicles" element={<Vehicles />} />
        <Route
          path="owners"
          element={
            <Protected adminOnly>
              <Owners />
            </Protected>
          }
        />
        <Route path="alerts" element={<Alerts />} />
        <Route
          path="training"
          element={
            <Protected adminOnly>
              <TrainingMonitor />
            </Protected>
          }
        />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
