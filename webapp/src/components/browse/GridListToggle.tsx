import { LayoutGrid, List } from 'lucide-react'
import { cn } from '../../utils/cn'

interface GridListToggleProps {
  layout: 'grid' | 'list'
  onChange: (layout: 'grid' | 'list') => void
}

export default function GridListToggle({ layout, onChange }: GridListToggleProps) {
  return (
    <div className="flex items-center gap-1 bg-white border border-slate-200 rounded-lg p-1 shadow-sm">
      <button
        onClick={() => onChange('grid')}
        className={cn(
          'p-1.5 rounded transition-colors',
          layout === 'grid' ? 'bg-orange-50 text-orange-600' : 'text-slate-400 hover:text-slate-600',
        )}
        title="Grid view"
      >
        <LayoutGrid className="w-4 h-4" />
      </button>
      <button
        onClick={() => onChange('list')}
        className={cn(
          'p-1.5 rounded transition-colors',
          layout === 'list' ? 'bg-orange-50 text-orange-600' : 'text-slate-400 hover:text-slate-600',
        )}
        title="List view"
      >
        <List className="w-4 h-4" />
      </button>
    </div>
  )
}
