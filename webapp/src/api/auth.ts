import { apiClient } from './client'

export interface LoginPayload {
  email: string
  password: string
}

export interface RegisterPayload {
  username: string
  email: string
  password: string
}

export interface AuthResponse {
  id: string
  username: string
  token: string
}

export const login = (p: LoginPayload): Promise<AuthResponse> =>
  apiClient.post<AuthResponse>('/login', p).then((r) => r.data)

export const register = (p: RegisterPayload): Promise<{ id: string; username: string; email: string }> =>
  apiClient.post('/register', p).then((r) => r.data)
