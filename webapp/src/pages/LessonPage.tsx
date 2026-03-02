import { useParams, Link } from 'react-router-dom'
import { useNavigate } from 'react-router-dom'
import { ChevronRight, CheckCircle2, MessageCircle } from 'lucide-react'
import { useCourse } from '../hooks/useCourse'
import { useLesson } from '../hooks/useLesson'
import { useProgress } from '../context/ProgressContext'
import { findLesson } from '../utils/lessonUrl'
import { getLessonContentPath } from '../api/courses'
import LessonReader from '../components/lesson/LessonReader'
import LessonNav from '../components/lesson/LessonNav'
import ReflectionPanel from '../components/lesson/ReflectionPanel'
import ChatInterface from '../components/chat/ChatInterface'
import Spinner from '../components/ui/Spinner'
import Button from '../components/ui/Button'
import ErrorBanner from '../components/ui/ErrorBanner'
import { useState } from 'react'

const STARTER_PROMPTS = [
  'Can you explain this concept further?',
  'Why is this important?',
  'Can you give me an example?',
  'What should I understand first?',
]

export default function LessonPage() {
  const { courseId, lessonId } = useParams<{ courseId: string; lessonId: string }>()
  const navigate = useNavigate()
  const { completed, markComplete } = useProgress()
  const [chatOpen, setChatOpen] = useState(false)

  const { data: course, isLoading: courseLoading, error: courseError } = useCourse(courseId)

  const location = course && lessonId ? findLesson(course, lessonId) : null
  const lesson = location?.lesson
  const contentPath = lesson ? getLessonContentPath(lesson) : null

  const { data: content, isLoading: contentLoading } = useLesson(courseId, contentPath)

  const prevLesson = location && location.lessonIndex > 0
    ? location.allLessons[location.lessonIndex - 1]
    : null
  const nextLesson = location && location.lessonIndex < location.allLessons.length - 1
    ? location.allLessons[location.lessonIndex + 1]
    : null

  const isDone = lesson ? completed.has(lesson.id) : false

  const handleComplete = () => {
    if (lesson) markComplete(lesson.id)
    if (nextLesson && courseId) {
      navigate(`/courses/${courseId}/lessons/${nextLesson.id}`)
    }
  }

  if (courseLoading) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <Spinner size="lg" />
      </div>
    )
  }

  if (courseError || !course || !lesson) {
    return (
      <div className="p-8">
        <ErrorBanner message="Lesson not found." />
      </div>
    )
  }

  return (
    <div className="flex h-[calc(100vh-56px)] overflow-hidden">
      {/* Lesson content */}
      <div className={`flex-1 overflow-y-auto transition-all bg-warm-100 ${chatOpen ? 'hidden md:block' : ''}`}>
        <div className="max-w-3xl mx-auto p-4 md:p-8">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-6 flex-wrap">
            <Link to="/" className="hover:text-orange-500">Courses</Link>
            <ChevronRight className="w-3 h-3" />
            <Link to={`/courses/${courseId}`} className="hover:text-orange-500">
              {course.title}
            </Link>
            <ChevronRight className="w-3 h-3" />
            <span className="text-slate-700 font-medium">{lesson.title}</span>
          </div>

          <div className="flex items-start justify-between gap-4 mb-6">
            <h1 className="text-2xl font-bold text-slate-800">{lesson.title}</h1>
            <div className="flex items-center gap-2 shrink-0">
              {isDone && <CheckCircle2 className="w-5 h-5 text-emerald-500" />}
              <button
                onClick={() => setChatOpen((v) => !v)}
                className="md:hidden flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-slate-200 text-sm text-slate-600 shadow-sm"
              >
                <MessageCircle className="w-4 h-4 text-orange-500" />
                Tutor
              </button>
            </div>
          </div>

          {contentLoading ? (
            <div className="flex justify-center py-12">
              <Spinner />
            </div>
          ) : content ? (
            <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 md:p-8">
              <LessonReader content={content} />
            </div>
          ) : (
            <p className="text-slate-400">Content not available</p>
          )}

          <ReflectionPanel
            keyPoints={lesson.keyPoints}
            reflectionQuestions={lesson.reflectionQuestions}
          />

          <div className="mt-8">
            <Button onClick={handleComplete} className="w-full" size="lg">
              {isDone
                ? nextLesson
                  ? 'Next Lesson →'
                  : '✓ Course Complete!'
                : nextLesson
                ? 'Mark Complete & Continue →'
                : 'Mark Complete'}
            </Button>
          </div>

          <LessonNav courseId={courseId!} prevLesson={prevLesson} nextLesson={nextLesson} />
        </div>
      </div>

      {/* Chat panel */}
      <div
        className={`
          border-l border-slate-200 bg-white flex flex-col
          ${chatOpen ? 'flex-1 md:w-96 md:flex-none' : 'hidden md:flex md:w-96'}
        `}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-gradient-to-r from-orange-50 to-amber-50">
          <div>
            <h3 className="text-sm font-bold text-slate-700">AI Tutor</h3>
            <p className="text-xs text-slate-500">Socratic guidance on this lesson</p>
          </div>
          <button
            onClick={() => setChatOpen(false)}
            className="md:hidden text-slate-400 hover:text-slate-600 text-xs font-medium"
          >
            Close
          </button>
        </div>
        <ChatInterface
          topicContext={lesson.title}
          starterPrompts={STARTER_PROMPTS}
          className="flex-1 min-h-0"
        />
      </div>
    </div>
  )
}
