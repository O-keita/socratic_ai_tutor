import { type InputHTMLAttributes, forwardRef } from 'react'
import { cn } from '../../utils/cn'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, className, ...props }, ref) => (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-sm font-medium text-slate-600">{label}</label>
      )}
      <input
        ref={ref}
        className={cn(
          'bg-white border border-slate-200 text-slate-800 placeholder-slate-400 rounded-2xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-orange-500/30 focus:border-orange-400 transition-all w-full shadow-sm',
          error && 'border-red-400 focus:ring-red-400/30',
          className,
        )}
        {...props}
      />
      {error && <p className="text-sm text-red-500">{error}</p>}
    </div>
  ),
)

Input.displayName = 'Input'
export default Input
