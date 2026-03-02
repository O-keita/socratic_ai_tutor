import { createContext, useContext, useState, type ReactNode } from 'react'
import type { AuthResponse } from '../api/auth'

const AUTH_KEY = 'auth'

interface AuthState {
  id: string
  username: string
  token: string
  expiresAt: number
}

interface AuthContextValue {
  user: AuthState | null
  login: (data: AuthResponse) => void
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

function loadStoredAuth(): AuthState | null {
  try {
    const raw = localStorage.getItem(AUTH_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as AuthState
    if (Date.now() > parsed.expiresAt) {
      localStorage.removeItem(AUTH_KEY)
      return null
    }
    return parsed
  } catch {
    return null
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthState | null>(loadStoredAuth)

  const login = (data: AuthResponse) => {
    const state: AuthState = {
      id: data.id,
      username: data.username,
      token: data.token,
      expiresAt: Date.now() + 24 * 60 * 60 * 1000,
    }
    localStorage.setItem(AUTH_KEY, JSON.stringify(state))
    setUser(state)
  }

  const logout = () => {
    localStorage.removeItem(AUTH_KEY)
    setUser(null)
  }

  return <AuthContext.Provider value={{ user, login, logout }}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
