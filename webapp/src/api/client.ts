import axios from 'axios'

const BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  (import.meta.env.DEV ? '/api' : '')

export const apiClient = axios.create({ baseURL: BASE })

apiClient.interceptors.request.use((config) => {
  try {
    const raw = localStorage.getItem('auth')
    if (raw) {
      const { token } = JSON.parse(raw) as { token: string }
      if (token) config.headers.Authorization = `Bearer ${token}`
    }
  } catch {
    // ignore malformed storage
  }
  return config
})

apiClient.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('auth')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  },
)
