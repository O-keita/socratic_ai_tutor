import { NavLink } from 'react-router-dom'
import { MessageCircle, User, Home } from 'lucide-react'
import { cn } from '../../utils/cn'

const NAV_ITEMS = [
  { to: '/', label: 'Browse', icon: Home, exact: true },
  { to: '/chat', label: 'AI Tutor', icon: MessageCircle, exact: false },
  { to: '/profile', label: 'Profile', icon: User, exact: false },
]

export default function Sidebar() {
  return (
    <aside className="hidden md:flex flex-col w-60 shrink-0 bg-white/60 backdrop-blur-sm border-r border-slate-200/60 py-6 px-3">
      <nav className="space-y-1">
        {NAV_ITEMS.map(({ to, label, icon: Icon, exact }) => (
          <NavLink
            key={to}
            to={to}
            end={exact}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition-all',
                isActive
                  ? 'bg-orange-50 text-orange-600 shadow-sm border border-orange-100'
                  : 'text-slate-500 hover:text-slate-700 hover:bg-warm-50',
              )
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Decorative bottom card */}
      <div className="mt-auto mx-1 p-4 rounded-2xl bg-gradient-to-br from-orange-50 to-warm-200 border border-orange-100">
        <p className="text-xs font-semibold text-orange-700 mb-1">Socratic Method</p>
        <p className="text-[11px] text-orange-600/70 leading-relaxed">Learn by discovering answers through guided questions.</p>
      </div>
    </aside>
  )
}
