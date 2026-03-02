export interface QuizMeta {
  id: string
  title: string
  description: string
  topic: string
}

export interface QuizManifest {
  quizzes: QuizMeta[]
}

export interface QuizQuestion {
  id: string
  topic: string
  question: string
  options: string[]
  correctIndex: number
  explanation: string
}
