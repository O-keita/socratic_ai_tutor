import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { getProgress, saveProgress } from '../api/progress'
import { useAuth } from './AuthContext'

interface ProgressContextValue {
  completed: Set<string>
  markComplete: (lessonId: string) => void
}

const ProgressContext = createContext<ProgressContextValue | null>(null)

export function ProgressProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth()
  const [completed, setCompleted] = useState<Set<string>>(new Set())
  const syncTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const completedRef = useRef(completed)
  completedRef.current = completed

  useEffect(() => {
    if (!user) {
      setCompleted(new Set())
      return
    }
    getProgress()
      .then((data) => {
        setCompleted(new Set(data.completedLessons ?? []))
      })
      .catch(() => {
        // offline or no progress yet — start empty
      })
  }, [user?.id])

  const markComplete = useCallback((lessonId: string) => {
    setCompleted((prev) => {
      if (prev.has(lessonId)) return prev
      const next = new Set(prev)
      next.add(lessonId)

      clearTimeout(syncTimer.current)
      syncTimer.current = setTimeout(() => {
        saveProgress({
          completedLessons: [...completedRef.current],
        }).catch(() => {
          // silent fail — will sync next time
        })
      }, 500)

      return next
    })
  }, [])

  return (
    <ProgressContext.Provider value={{ completed, markComplete }}>
      {children}
    </ProgressContext.Provider>
  )
}

export function useProgress(): ProgressContextValue {
  const ctx = useContext(ProgressContext)
  if (!ctx) throw new Error('useProgress must be used within ProgressProvider')
  return ctx
}
