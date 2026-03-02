import { Link } from 'react-router-dom'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import type { Lesson } from '../../api/courses'

interface LessonNavProps {
  courseId: string
  prevLesson: Lesson | null
  nextLesson: Lesson | null
}

export default function LessonNav({ courseId, prevLesson, nextLesson }: LessonNavProps) {
  return (
    <div className="flex justify-between items-center pt-6 mt-6 border-t border-slate-200">
      {prevLesson ? (
        <Link
          to={`/courses/${courseId}/lessons/${prevLesson.id}`}
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-orange-500 transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          <span className="line-clamp-1 max-w-[200px]">{prevLesson.title}</span>
        </Link>
      ) : (
        <div />
      )}
      {nextLesson ? (
        <Link
          to={`/courses/${courseId}/lessons/${nextLesson.id}`}
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-orange-500 transition-colors"
        >
          <span className="line-clamp-1 max-w-[200px]">{nextLesson.title}</span>
          <ChevronRight className="w-4 h-4" />
        </Link>
      ) : (
        <div />
      )}
    </div>
  )
}
