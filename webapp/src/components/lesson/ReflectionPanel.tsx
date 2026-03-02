import { useState } from 'react'
import { ChevronDown, Lightbulb, HelpCircle } from 'lucide-react'
import { cn } from '../../utils/cn'

interface ReflectionPanelProps {
  keyPoints?: string
  reflectionQuestions?: string[]
}

export default function ReflectionPanel({ keyPoints, reflectionQuestions }: ReflectionPanelProps) {
  const [open, setOpen] = useState(false)

  if (!keyPoints && (!reflectionQuestions || reflectionQuestions.length === 0)) return null

  return (
    <div className="mt-8 border border-orange-200 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-3 px-4 py-3 bg-orange-50 hover:bg-orange-100/70 transition-colors text-left"
      >
        <Lightbulb className="w-4 h-4 text-orange-500 shrink-0" />
        <span className="font-medium text-orange-600 text-sm">Key Points & Reflection</span>
        <ChevronDown
          className={cn('w-4 h-4 text-orange-500 ml-auto transition-transform', open && 'rotate-180')}
        />
      </button>
      {open && (
        <div className="px-4 py-4 space-y-4 bg-orange-50/50">
          {keyPoints && (
            <div>
              <h4 className="text-xs font-semibold text-orange-600 uppercase tracking-wider mb-2">
                Key Points
              </h4>
              <p className="text-sm text-slate-600">{keyPoints}</p>
            </div>
          )}
          {reflectionQuestions && reflectionQuestions.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-orange-600 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <HelpCircle className="w-3.5 h-3.5" />
                Reflection Questions
              </h4>
              <ul className="space-y-2">
                {reflectionQuestions.map((q, i) => (
                  <li key={i} className="flex gap-2 text-sm text-slate-600">
                    <span className="text-orange-500 font-semibold shrink-0">{i + 1}.</span>
                    {q}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
