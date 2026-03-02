import { useState } from 'react'
import { Brain, CheckCircle2, XCircle, RotateCcw, Trophy, TrendingUp } from 'lucide-react'

interface QuizQuestion {
  id: string
  topic: string
  question: string
  options: string[]
  correctIndex: number
  explanation: string
}

// Sample quiz data - in production this would come from the backend
const SAMPLE_QUESTIONS: QuizQuestion[] = [
  {
    id: 'ml_01',
    topic: 'Fundamentals',
    question: "In a linear regression model, what happens to the model's performance if we add too many irrelevant features?",
    options: [
      'It always improves the accuracy',
      'It may lead to overfitting',
      'It reduces the complexity',
      'It eliminates bias completely'
    ],
    correctIndex: 1,
    explanation: "Adding irrelevant features increases the model's flexibility to fit noise in the training data, leading to generalization issues."
  },
  {
    id: 'ml_02',
    topic: 'Supervised Learning',
    question: "Which algorithm is most likely to be used for predicting whether an email is 'Spam' or 'Not Spam'?",
    options: [
      'K-Means Clustering',
      'Linear Regression',
      'Logistic Regression',
      'Principal Component Analysis'
    ],
    correctIndex: 2,
    explanation: 'Spam detection is a binary classification problem, for which Logistic Regression is a fundamental supervised learning algorithm.'
  },
  {
    id: 'ml_03',
    topic: 'Validation',
    question: "What is the primary purpose of a 'test set' in machine learning?",
    options: [
      "To train the model's weights",
      'To tune the hyperparameters',
      'To provide an unbiased evaluation of the final model',
      'To increase the size of the training data'
    ],
    correctIndex: 2,
    explanation: 'The test set is used to evaluate how well the model generalizes to completely unseen data after training and tuning are complete.'
  },
  {
    id: 'ml_04',
    topic: 'Bias-Variance Tradeoff',
    question: 'A model that is very simple and fails to capture the underlying trend of the data is said to have:',
    options: [
      'High Variance',
      'High Bias',
      'Low Bias',
      'Zero Error'
    ],
    correctIndex: 1,
    explanation: 'High bias occurs when the model makes strong assumptions about the data, often leading to underfitting.'
  },
  {
    id: 'ml_05',
    topic: 'Decision Trees',
    question: 'Which of the following is a common technique used to prevent Decision Trees from overfitting?',
    options: [
      'Exploding the tree',
      'Pruning',
      'Increasing the depth infinitely',
      'Removing labels'
    ],
    correctIndex: 1,
    explanation: 'Pruning involves removing branches that provide little power to classify instances, thereby simplifying the model.'
  }
]

export default function QuizPage() {
  const [currentQ, setCurrentQ] = useState(0)
  const [selected, setSelected] = useState<number | null>(null)
  const [showExplanation, setShowExplanation] = useState(false)
  const [score, setScore] = useState({ correct: 0, total: 0 })
  const [quizComplete, setQuizComplete] = useState(false)

  const question = SAMPLE_QUESTIONS[currentQ]

  const handleSelect = (idx: number) => {
    if (showExplanation) return
    setSelected(idx)
  }

  const handleSubmit = () => {
    if (selected === null) return
    setShowExplanation(true)
    const isCorrect = selected === question.correctIndex
    setScore(prev => ({ correct: prev.correct + (isCorrect ? 1 : 0), total: prev.total + 1 }))
  }

  const handleNext = () => {
    if (currentQ + 1 < SAMPLE_QUESTIONS.length) {
      setCurrentQ(currentQ + 1)
      setSelected(null)
      setShowExplanation(false)
    } else {
      setQuizComplete(true)
    }
  }

  const handleRestart = () => {
    setCurrentQ(0)
    setSelected(null)
    setShowExplanation(false)
    setScore({ correct: 0, total: 0 })
    setQuizComplete(false)
  }

  if (quizComplete) {
    const pct = Math.round((score.correct / score.total) * 100)
    return (
      <div className="max-w-2xl mx-auto p-4 md:p-8 space-y-8">
        <div className="bg-gradient-to-r from-purple-500 to-indigo-500 rounded-2xl p-8 text-white text-center">
          <div className="w-20 h-20 rounded-full bg-white/20 mx-auto flex items-center justify-center mb-4">
            <Trophy className="w-10 h-10" />
          </div>
          <h1 className="text-3xl font-bold mb-2">Quiz Complete!</h1>
          <p className="text-purple-100">Great work testing your knowledge</p>
        </div>

        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-8 text-center">
          <div className="text-6xl font-bold text-purple-500 mb-2">{pct}%</div>
          <p className="text-slate-600 mb-6">
            {score.correct} out of {score.total} questions correct
          </p>

          <div className="flex gap-3 justify-center">
            <button
              onClick={handleRestart}
              className="flex items-center gap-2 px-6 py-2.5 bg-purple-500 text-white font-semibold rounded-lg hover:bg-purple-600 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Try Again
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-3xl mx-auto p-4 md:p-8 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-500 to-indigo-500 rounded-2xl p-6 text-white">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center">
              <Brain className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Practice Quiz</h1>
              <p className="text-purple-100 text-sm">Machine Learning Fundamentals</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">{score.correct}/{score.total}</div>
            <div className="text-purple-100 text-xs">Score</div>
          </div>
        </div>

        {/* Progress */}
        <div className="flex items-center gap-3 text-sm">
          <span className="text-purple-100">Question {currentQ + 1}/{SAMPLE_QUESTIONS.length}</span>
          <div className="flex-1 h-2 bg-white/20 rounded-full overflow-hidden">
            <div
              className="h-full bg-white transition-all duration-300"
              style={{ width: `${((currentQ + 1) / SAMPLE_QUESTIONS.length) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Question Card */}
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 md:p-8">
        <div className="inline-block px-3 py-1 bg-purple-50 text-purple-600 text-xs font-semibold rounded-full mb-4">
          {question.topic}
        </div>
        <h2 className="text-xl font-bold text-slate-800 mb-6">{question.question}</h2>

        <div className="space-y-3">
          {question.options.map((option, idx) => {
            const isSelected = selected === idx
            const isCorrect = idx === question.correctIndex
            const showResult = showExplanation

            let className = 'block w-full text-left p-4 rounded-xl border-2 transition-all '

            if (!showResult) {
              className += isSelected
                ? 'border-purple-500 bg-purple-50 text-slate-800'
                : 'border-slate-200 hover:border-purple-200 hover:bg-slate-50 text-slate-700'
            } else {
              if (isCorrect) {
                className += 'border-emerald-500 bg-emerald-50 text-slate-800'
              } else if (isSelected && !isCorrect) {
                className += 'border-red-500 bg-red-50 text-slate-800'
              } else {
                className += 'border-slate-200 bg-slate-50 text-slate-500'
              }
            }

            return (
              <button
                key={idx}
                onClick={() => handleSelect(idx)}
                disabled={showResult}
                className={className}
              >
                <div className="flex items-center gap-3">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center">
                    {showResult && isCorrect && <CheckCircle2 className="w-5 h-5 text-emerald-600" />}
                    {showResult && isSelected && !isCorrect && <XCircle className="w-5 h-5 text-red-600" />}
                  </div>
                  <span className="flex-1 font-medium text-sm">{option}</span>
                </div>
              </button>
            )
          })}
        </div>

        {showExplanation && (
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-xl">
            <div className="flex items-start gap-2">
              <TrendingUp className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-blue-900 mb-1">Explanation</h3>
                <p className="text-sm text-blue-800">{question.explanation}</p>
              </div>
            </div>
          </div>
        )}

        <div className="mt-6 flex gap-3">
          {!showExplanation ? (
            <button
              onClick={handleSubmit}
              disabled={selected === null}
              className="flex-1 py-2.5 bg-purple-500 text-white font-semibold rounded-lg hover:bg-purple-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Submit Answer
            </button>
          ) : (
            <button
              onClick={handleNext}
              className="flex-1 py-2.5 bg-purple-500 text-white font-semibold rounded-lg hover:bg-purple-600 transition-colors"
            >
              {currentQ + 1 < SAMPLE_QUESTIONS.length ? 'Next Question' : 'Finish Quiz'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
