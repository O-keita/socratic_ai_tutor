import { useEffect, useRef, useState } from 'react'
import { sendMessage, type ChatMessage } from '../../api/chat'
import ChatMsg from './ChatMessage'
import ChatInput from './ChatInput'
import TypingIndicator from './TypingIndicator'
import ErrorBanner from '../ui/ErrorBanner'

interface ChatInterfaceProps {
  topicContext?: string
  starterPrompts?: string[]
  className?: string
}

export default function ChatInterface({
  topicContext,
  starterPrompts,
  className,
}: ChatInterfaceProps) {
  const [history, setHistory] = useState<ChatMessage[]>([])
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history, pending])

  const handleSend = async (text: string) => {
    setError(null)

    const userMsg: ChatMessage = { role: 'user', content: text }
    const historyToSend = topicContext && history.length === 0
      ? [
          { role: 'user' as const, content: `I am currently studying: "${topicContext}". I'd like your help understanding this topic using the Socratic method.` },
          ...history,
        ]
      : history

    setHistory((h) => [...h, userMsg])
    setPending(true)

    try {
      const data = await sendMessage(text, historyToSend)
      setHistory((h) => [...h, { role: 'assistant', content: data.response }])
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to get a response. Please try again.'
      setError(msg)
    } finally {
      setPending(false)
    }
  }

  return (
    <div className={`flex flex-col h-full bg-warm-100/50 ${className ?? ''}`}>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {history.length === 0 && starterPrompts && (
          <div className="text-center space-y-4 py-8">
            <p className="text-slate-500 text-sm">
              {topicContext
                ? `Ask me anything about "${topicContext}"`
                : 'What would you like to explore?'}
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {starterPrompts.map((p) => (
                <button
                  key={p}
                  onClick={() => handleSend(p)}
                  className="px-3 py-2 text-sm bg-white hover:bg-orange-50 border border-slate-200 hover:border-orange-300 text-slate-600 rounded-xl transition-all shadow-sm"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}

        {history.map((msg, i) => (
          <ChatMsg key={i} message={msg} />
        ))}

        {pending && <TypingIndicator />}

        {error && (
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        )}

        <div ref={bottomRef} />
      </div>

      <ChatInput onSend={handleSend} disabled={pending} />
    </div>
  )
}
