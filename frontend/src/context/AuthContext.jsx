/**
 * Authentication context.
 *
 * Holds the current user for the whole app and exposes login/logout. The token
 * itself lives in localStorage (see api/client.js) so a page refresh does not
 * sign the user out.
 */
import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import { api, clearSession, getStoredUser, getToken, storeSession } from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => getStoredUser())

  const login = useCallback(async (email, password) => {
    const result = await api.login(email, password)
    storeSession(result.access_token, result.user)
    setUser(result.user)
    return result.user
  }, [])

  const logout = useCallback(() => {
    clearSession()
    setUser(null)
  }, [])

  const value = useMemo(
    () => ({
      user,
      setUser,
      login,
      logout,
      isAuthenticated: Boolean(user && getToken()),
      isAdmin: user?.role === 'admin',
    }),
    [user, login, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside an AuthProvider')
  return context
}
