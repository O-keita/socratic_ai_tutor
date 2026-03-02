import { apiClient } from './client'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatResponse {
  response: string
  socratic_index: number
  scaffolding_level: string
  sentiment: string
}

export const sendMessage = (
  message: string,
  history: ChatMessage[],
  maxTokens?: number,
): Promise<ChatResponse> =>
  apiClient
    .post<ChatResponse>('/chat', { message, history, max_tokens: maxTokens })
    .then((r) => r.data)
