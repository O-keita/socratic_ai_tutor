import { cn } from '../../utils/cn'

interface DifficultyBadgeProps {
  level: string
  className?: string
}

function getColor(level: string): string {
  const lower = level.toLowerCase()
  if (lower.includes('beginner')) return 'bg-emerald-50 text-emerald-600 border-emerald-200'
  if (lower.includes('intermediate')) return 'bg-amber-50 text-amber-600 border-amber-200'
  if (lower.includes('advanced')) return 'bg-red-50 text-red-600 border-red-200'
  return 'bg-slate-50 text-slate-500 border-slate-200'
}

export default function DifficultyBadge({ level, className }: DifficultyBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2.5 py-0.5 text-xs font-medium rounded-full border',
        getColor(level),
        className,
      )}
    >
      {level}
    </span>
  )
}
