import { Link, NavLink } from 'react-router-dom'
import { Brain, BookOpen, MessageCircle, Code, User, LogOut, Settings, GraduationCap, Sparkles, ChevronDown } from 'lucide-react'
import { useAuth } from '../../context/AuthContext'
import { cn } from '../../utils/cn'
import { useState, useRef, useEffect } from 'react'

const NAV_LINKS = [
  { to: '/', label: 'Home', icon: GraduationCap, exact: true },
  { to: '/browse', label: 'Explore', icon: Sparkles, exact: false },
  { to: '/chat', label: 'AI Tutor', icon: MessageCircle, exact: false },
]

export default function TopBar() {
  const { user, logout } = useAuth()
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setUserMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <header className="bg-white border-b border-slate-200 shrink-0 z-30 sticky top-0">
      <div className="max-w-7xl mx-auto px-4 md:px-6 h-16 flex items-center gap-6">

        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 shrink-0 group">
          <div className="w-9 h-9 rounded-lg bg-hero-gradient flex items-center justify-center shadow-brand">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div className="hidden sm:block">
            <span className="text-sm font-bold text-slate-900 group-hover:text-brand-600 transition-colors">Socratic</span>
            <span className="text-xs text-slate-500 block -mt-0.5">AI Tutor</span>
          </div>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-1 flex-1">
          {NAV_LINKS.map(({ to, label, icon: Icon, exact }) => (
            <NavLink
              key={to}
              to={to}
              end={exact}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'text-brand-600 bg-brand-50'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50',
                )
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Mobile nav */}
        <nav className="flex md:hidden items-center gap-0.5 flex-1">
          {[
            { to: '/', icon: GraduationCap, exact: true },
            { to: '/chat', icon: MessageCircle, exact: false },
            { to: '/profile', icon: User, exact: false },
          ].map(({ to, icon: Icon, exact }) => (
            <NavLink
              key={to}
              to={to}
              end={exact}
              className={({ isActive }) =>
                cn(
                  'p-2 rounded-lg transition-colors',
                  isActive ? 'text-brand-600 bg-brand-50' : 'text-slate-400 hover:text-slate-600',
                )
              }
            >
              <Icon className="w-5 h-5" />
            </NavLink>
          ))}
        </nav>

        {/* Right section - User Menu */}
        <div className="ml-auto flex items-center gap-3">
          {user && (
            <div className="relative" ref={menuRef}>
              <button
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                className="flex items-center gap-2 hover:bg-slate-50 rounded-lg px-2 py-1.5 transition-colors"
              >
                <div className="w-8 h-8 rounded-full bg-hero-gradient flex items-center justify-center shadow-sm">
                  <span className="text-white text-xs font-bold">{user.username[0]?.toUpperCase()}</span>
                </div>
                <div className="hidden lg:block text-left">
                  <p className="text-xs font-semibold text-slate-700 leading-none">{user.username}</p>
                  <p className="text-[10px] text-slate-400 mt-0.5">Student</p>
                </div>
                <ChevronDown className={cn('w-4 h-4 text-slate-400 transition-transform hidden sm:block', userMenuOpen && 'rotate-180')} />
              </button>

              {/* User Dropdown Menu */}
              {userMenuOpen && (
                <div className="absolute top-full right-0 mt-2 w-56 bg-white rounded-xl shadow-lg border border-slate-200 py-1.5 z-50">
                  {/* User Info */}
                  <div className="px-4 py-3 border-b border-slate-100">
                    <p className="text-sm font-semibold text-slate-900">{user.username}</p>
                    <p className="text-xs text-slate-500 mt-0.5">Learning with Socratic AI</p>
                  </div>

                  {/* Menu Items */}
                  <div className="py-1">
                    {[
                      { to: '/profile', icon: User, label: 'My Learning' },
                      { to: '/playground', icon: Code, label: 'Playground' },
                      { to: '/quiz', icon: Brain, label: 'Practice Quiz' },
                      { to: '/glossary', icon: BookOpen, label: 'Glossary' },
                    ].map(({ to, icon: Icon, label }) => (
                      <Link
                        key={to}
                        to={to}
                        onClick={() => setUserMenuOpen(false)}
                        className="flex items-center gap-3 px-4 py-2.5 text-sm text-slate-700 hover:bg-brand-50 hover:text-brand-600 transition-colors"
                      >
                        <Icon className="w-4 h-4" />
                        {label}
                      </Link>
                    ))}
                    <div className="border-t border-slate-100 my-1" />
                    <Link
                      to="/settings"
                      onClick={() => setUserMenuOpen(false)}
                      className="flex items-center gap-3 px-4 py-2.5 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
                    >
                      <Settings className="w-4 h-4" />
                      Settings
                    </Link>
                    <button
                      onClick={() => {
                        setUserMenuOpen(false)
                        logout()
                      }}
                      className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                    >
                      <LogOut className="w-4 h-4" />
                      Sign Out
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  )
}