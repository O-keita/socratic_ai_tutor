import { User } from 'lucide-react'
import LessonReader from '../lesson/LessonReader'
import type { ChatMessage as ChatMsg } from '../../api/chat'

interface ChatMessageProps {
  message: ChatMsg
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex gap-3 items-end justify-end">
        <div className="max-w-[75%] bg-gradient-to-br from-orange-500 to-amber-500 rounded-2xl rounded-br-sm px-4 py-3 shadow-sm">
          <p className="text-sm text-white whitespace-pre-wrap">{message.content}</p>
        </div>
        <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center shrink-0">
          <User className="w-4 h-4 text-slate-500" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3 items-start">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-400 to-amber-400 flex items-center justify-center shrink-0 shadow-sm">
        <span className="text-white text-xs font-bold">AI</span>
      </div>
      <div className="max-w-[85%] bg-white rounded-2xl rounded-tl-sm px-4 py-3 border border-slate-100 shadow-sm">
        <div className="text-sm text-slate-700">
          <LessonReader content={message.content} />
        </div>
      </div>
    </div>
  )
}
