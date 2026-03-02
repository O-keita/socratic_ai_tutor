import { Code, Info } from 'lucide-react'

export default function PlaygroundPage() {
  return (
    <div className="flex flex-col h-[calc(100vh-56px)]">
      {/* Header */}
      <div className="px-6 py-5 bg-gradient-to-r from-blue-500 to-cyan-500 text-white">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
            <Code className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold">Python Playground</h1>
            <p className="text-sm text-blue-100">Write and run Python code in your browser</p>
          </div>
        </div>
      </div>

      {/* Info banner */}
      <div className="px-6 py-3 bg-blue-50 dark:bg-blue-900/20 border-b border-blue-100 dark:border-blue-800 flex items-center gap-2">
        <Info className="w-4 h-4 text-blue-500 shrink-0" />
        <p className="text-xs text-blue-600 dark:text-blue-400">
          Powered by Pyodide (Python in WebAssembly). First load downloads ~8 MB. Supports numpy, pandas, sklearn, and more.
        </p>
      </div>

      {/* Iframe */}
      <iframe
        src="/playground/index.html"
        title="Python Playground"
        className="flex-1 w-full border-0"
        sandbox="allow-scripts allow-same-origin allow-popups"
        allow="clipboard-write"
      />
    </div>
  )
}
