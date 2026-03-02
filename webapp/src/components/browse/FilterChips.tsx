import { cn } from '../../utils/cn'

interface ChipOption {
  label: string
  value: string
}

interface FilterChipsProps {
  label: string
  options: ChipOption[]
  value: string
  onChange: (value: string) => void
}

function Chip({
  label,
  active,
  onClick,
}: {
  label: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'px-3 py-1.5 rounded-full text-sm font-medium border transition-all',
        active
          ? 'bg-gradient-to-r from-orange-500 to-amber-500 text-white border-orange-500 shadow-sm'
          : 'bg-white text-slate-500 border-slate-200 hover:border-orange-300 hover:text-slate-700',
      )}
    >
      {label}
    </button>
  )
}

export default function FilterChips({ label, options, value, onChange }: FilterChipsProps) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">{label}</span>
      {options.map((opt) => (
        <Chip
          key={opt.value}
          label={opt.label}
          active={value === opt.value}
          onClick={() => onChange(opt.value === value ? 'All' : opt.value)}
        />
      ))}
    </div>
  )
}
