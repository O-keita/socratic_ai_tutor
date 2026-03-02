export default function TypingIndicator() {
  return (
    <div className="flex gap-3 items-start">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-400 to-amber-400 flex items-center justify-center shrink-0 shadow-sm">
        <span className="text-white text-xs font-bold">AI</span>
      </div>
      <div className="bg-white rounded-2xl rounded-tl-sm px-4 py-3 border border-slate-100 shadow-sm">
        <div className="flex gap-1 items-center h-5">
          <div className="w-2 h-2 bg-orange-300 rounded-full animate-bounce [animation-delay:-0.3s]" />
          <div className="w-2 h-2 bg-orange-300 rounded-full animate-bounce [animation-delay:-0.15s]" />
          <div className="w-2 h-2 bg-orange-300 rounded-full animate-bounce" />
        </div>
      </div>
    </div>
  )
}
