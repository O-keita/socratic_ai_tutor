import { Settings, Moon, Sun, Server, Info } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

export default function SettingsPage() {
  const { isDark, toggle } = useTheme()

  return (
    <div className="max-w-2xl mx-auto p-4 md:p-8 space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 to-slate-800 rounded-2xl p-6 md:p-8 text-white">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center">
            <Settings className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Settings</h1>
            <p className="text-slate-300 text-sm">Customize your experience</p>
          </div>
        </div>
      </div>

      {/* Appearance */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-100 dark:border-slate-700 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-700">
          <h2 className="font-bold text-slate-800 dark:text-white">Appearance</h2>
        </div>
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {isDark ? <Moon className="w-5 h-5 text-indigo-400" /> : <Sun className="w-5 h-5 text-amber-500" />}
              <div>
                <p className="font-medium text-slate-700 dark:text-slate-200">Dark Mode</p>
                <p className="text-sm text-slate-500 dark:text-slate-400">Switch between light and dark themes</p>
              </div>
            </div>
            <button
              onClick={toggle}
              className={`relative w-12 h-7 rounded-full transition-colors ${
                isDark ? 'bg-orange-500' : 'bg-slate-300'
              }`}
            >
              <div
                className={`absolute top-0.5 w-6 h-6 rounded-full bg-white shadow transition-transform ${
                  isDark ? 'translate-x-5' : 'translate-x-0.5'
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Server Info */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-100 dark:border-slate-700 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-700">
          <h2 className="font-bold text-slate-800 dark:text-white">AI Engine</h2>
        </div>
        <div className="px-6 py-4 space-y-4">
          <div className="flex items-start gap-3">
            <Server className="w-5 h-5 text-slate-400 mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="font-medium text-slate-700 dark:text-slate-200">Backend Server</p>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                The web app connects to the remote Socratic AI backend for all conversations.
              </p>
              <code className="mt-2 block text-xs bg-slate-50 dark:bg-slate-900 text-orange-500 px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 break-all">
                {import.meta.env.VITE_API_BASE_URL || window.location.origin + '/api'}
              </code>
            </div>
          </div>
        </div>
      </div>

      {/* About */}
      <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-100 dark:border-slate-700 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-700">
          <h2 className="font-bold text-slate-800 dark:text-white">About</h2>
        </div>
        <div className="px-6 py-4">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-slate-400 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-slate-700 dark:text-slate-200">Bantaba AI</p>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                An AI-powered learning platform that teaches data science and machine learning using the Socratic method.
                The AI never gives direct answers — it guides you with questions to develop critical thinking.
              </p>
              <p className="text-xs text-slate-400 mt-3">Web App v1.0 &middot; Capstone Project</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
