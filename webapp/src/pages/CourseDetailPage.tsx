import { useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Clock, BookOpen, ChevronRight, PlayCircle } from 'lucide-react'
import { useCourse } from '../hooks/useCourse'
import { useProgress } from '../context/ProgressContext'
import { courseCompletionRatio } from '../utils/progressSchema'
import { flattenLessons } from '../utils/lessonUrl'
import SyllabusTree from '../components/course/SyllabusTree'
import DifficultyBadge from '../components/course/DifficultyBadge'
import Spinner from '../components/ui/Spinner'
import Button from '../components/ui/Button'
import ErrorBanner from '../components/ui/ErrorBanner'

export default function CourseDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { data: course, isLoading, error } = useCourse(id)
  const { completed } = useProgress()

  const ratio = course
    ? courseCompletionRatio([...completed], course.id, course.totalLessons)
    : 0

  const resumeLesson = useMemo(() => {
    if (!course) return null
    const all = flattenLessons(course)
    return all.find((l) => !completed.has(l.id)) ?? all[all.length - 1] ?? null
  }, [course, completed])

  if (isLoading) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <Spinner size="lg" />
      </div>
    )
  }

  if (error || !course) {
    return (
      <div className="p-8">
        <ErrorBanner message="Failed to load course. Please try again." />
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto p-4 md:p-8 space-y-8">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-slate-400">
        <Link to="/" className="hover:text-orange-500 transition-colors">Courses</Link>
        <ChevronRight className="w-4 h-4" />
        <span className="text-slate-700 font-medium">{course.title}</span>
      </div>

      {/* Hero */}
      <div className="bg-gradient-to-br from-orange-500 to-amber-500 rounded-2xl p-6 md:p-8 text-white space-y-4">
        <div className="flex flex-wrap gap-2">
          <DifficultyBadge level={course.difficulty} className="!bg-white/20 !text-white !border-white/30" />
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 text-xs font-medium rounded-full bg-white/20 text-white border border-white/30">
            <Clock className="w-3.5 h-3.5" />
            {course.duration}
          </span>
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 text-xs font-medium rounded-full bg-white/20 text-white border border-white/30">
            <BookOpen className="w-3.5 h-3.5" />
            {course.totalLessons} lessons
          </span>
        </div>

        <h1 className="text-3xl font-bold">{course.title}</h1>
        <p className="text-orange-100">{course.description}</p>

        {ratio > 0 && (
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-orange-100">
              <span>Progress</span>
              <span>{Math.round(ratio * 100)}%</span>
            </div>
            <div className="h-2 bg-white/20 rounded-full overflow-hidden">
              <div
                className="h-full bg-white rounded-full transition-all duration-300"
                style={{ width: `${Math.round(ratio * 100)}%` }}
              />
            </div>
          </div>
        )}

        {resumeLesson && (
          <Link to={`/courses/${course.id}/lessons/${resumeLesson.id}`}>
            <Button size="lg" className="!bg-white !text-orange-600 hover:!bg-orange-50 flex items-center gap-2 shadow-lg">
              <PlayCircle className="w-5 h-5" />
              {ratio === 0 ? 'Start Course' : ratio >= 1 ? 'Review Course' : 'Resume'}
            </Button>
          </Link>
        )}
      </div>

      {/* Syllabus */}
      <div>
        <h2 className="text-lg font-bold text-slate-800 mb-4">Course Content</h2>
        <SyllabusTree course={course} />
      </div>
    </div>
  )
}
