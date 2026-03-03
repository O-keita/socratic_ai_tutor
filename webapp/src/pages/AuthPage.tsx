import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { BookOpen } from 'lucide-react'
import { login, register } from '../api/auth'
import { useAuth } from '../context/AuthContext'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import ErrorBanner from '../components/ui/ErrorBanner'

type Tab = 'login' | 'register'

interface AuthPageProps {
  defaultTab?: Tab
}

function getErrorMessage(err: unknown): string {
  if (err && typeof err === 'object' && 'response' in err) {
    const resp = (err as { response?: { data?: { detail?: string } } }).response
    return resp?.data?.detail ?? 'An error occurred'
  }
  if (err instanceof Error) return err.message
  return 'An error occurred'
}

export default function AuthPage({ defaultTab = 'login' }: AuthPageProps) {
  const [tab, setTab] = useState<Tab>(defaultTab)
  const [error, setError] = useState<string | null>(null)
  const { login: saveAuth } = useAuth()
  const navigate = useNavigate()

  // Login form state
  const [loginEmail, setLoginEmail] = useState('')
  const [loginPassword, setLoginPassword] = useState('')

  // Register form state
  const [regUsername, setRegUsername] = useState('')
  const [regEmail, setRegEmail] = useState('')
  const [regPassword, setRegPassword] = useState('')

  const loginMut = useMutation({
    mutationFn: () => login({ email: loginEmail, password: loginPassword }),
    onSuccess: (data) => {
      saveAuth(data)
      navigate('/')
    },
    onError: (err) => setError(getErrorMessage(err)),
  })

  const registerMut = useMutation({
    mutationFn: () => register({ username: regUsername, email: regEmail, password: regPassword }),
    onSuccess: () => {
      setTab('login')
      setLoginEmail(regEmail)
      setError(null)
    },
    onError: (err) => setError(getErrorMessage(err)),
  })

  const handleTabChange = (newTab: Tab) => {
    setTab(newTab)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-warm-100 to-amber-50 flex items-center justify-center p-4">
      {/* Decorative blobs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-orange-200/30 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-amber-200/20 rounded-full blur-3xl" />
      </div>

      <div className="w-full max-w-md relative z-10">
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center mx-auto mb-4 shadow-lg shadow-orange-500/25">
            <BookOpen className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-slate-800">Bantaba AI</h1>
          <p className="text-slate-500 mt-1.5">Learn through guided discovery</p>
        </div>

        <div className="bg-white/80 backdrop-blur-sm rounded-3xl border border-white/60 shadow-xl shadow-orange-900/5 p-7 space-y-6">
          <div className="flex bg-warm-100 rounded-2xl p-1.5 gap-1">
            {(['login', 'register'] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => handleTabChange(t)}
                className={`flex-1 py-2.5 text-sm font-semibold rounded-xl transition-all capitalize ${
                  tab === t
                    ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white shadow-md shadow-orange-500/25'
                    : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                {t === 'login' ? 'Sign In' : 'Sign Up'}
              </button>
            ))}
          </div>

          {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

          {tab === 'login' ? (
            <form
              onSubmit={(e) => { e.preventDefault(); loginMut.mutate() }}
              className="space-y-4"
            >
              <Input
                label="Email or username"
                type="text"
                value={loginEmail}
                onChange={(e) => setLoginEmail(e.target.value)}
                placeholder="you@example.com"
                required
              />
              <Input
                label="Password"
                type="password"
                value={loginPassword}
                onChange={(e) => setLoginPassword(e.target.value)}
                placeholder="••••••••"
                required
              />
              <Button type="submit" className="w-full" loading={loginMut.isPending}>
                Sign In
              </Button>
            </form>
          ) : (
            <form
              onSubmit={(e) => { e.preventDefault(); registerMut.mutate() }}
              className="space-y-4"
            >
              <Input
                label="Username"
                type="text"
                value={regUsername}
                onChange={(e) => setRegUsername(e.target.value)}
                placeholder="your_username"
                required
              />
              <Input
                label="Email"
                type="email"
                value={regEmail}
                onChange={(e) => setRegEmail(e.target.value)}
                placeholder="you@example.com"
                required
              />
              <Input
                label="Password"
                type="password"
                value={regPassword}
                onChange={(e) => setRegPassword(e.target.value)}
                placeholder="••••••••"
                required
              />
              <Button type="submit" className="w-full" loading={registerMut.isPending}>
                Create Account
              </Button>
            </form>
          )}
        </div>

        <p className="text-center text-xs text-slate-400 mt-6">
          Powered by Bantaba AI &middot; Learn by questioning
        </p>
      </div>
    </div>
  )
}
