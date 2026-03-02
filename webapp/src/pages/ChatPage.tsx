import { MessageCircle, Sparkles } from 'lucide-react'
import ChatInterface from '../components/chat/ChatInterface'

const STARTER_PROMPTS = [
  'What is gradient descent?',
  'Explain overfitting',
  'What is a neural network?',
  'How does backpropagation work?',
  'What is the difference between supervised and unsupervised learning?',
  'Explain the bias-variance tradeoff',
]

export default function ChatPage() {
  return (
    <div className="flex flex-col h-[calc(100vh-56px)]">
      <div className="px-6 py-5 bg-gradient-to-r from-orange-500 to-amber-500 text-white">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
            <MessageCircle className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold flex items-center gap-2">
              AI Tutor
              <Sparkles className="w-4 h-4 text-amber-200" />
            </h1>
            <p className="text-sm text-orange-100">
              I'll guide you with questions, not answers
            </p>
          </div>
        </div>
      </div>
      <ChatInterface
        starterPrompts={STARTER_PROMPTS}
        className="flex-1 min-h-0"
      />
    </div>
  )
}
