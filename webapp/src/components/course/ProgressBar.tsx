import { cn } from '../../utils/cn'

interface ProgressBarProps {
  value: number // 0-1
  className?: string
  showLabel?: boolean
}

export default function ProgressBar({ value, className, showLabel }: ProgressBarProps) {
  const pct = Math.round(value * 100)
  return (
    <div className={cn('w-full', className)}>
      {showLabel && (
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>Progress</span>
          <span className="font-medium text-orange-500">{pct}%</span>
        </div>
      )}
      <div className="h-1.5 bg-orange-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-orange-500 to-amber-500 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
