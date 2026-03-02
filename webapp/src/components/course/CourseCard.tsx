import { Link } from 'react-router-dom'
import { Clock, BookOpen, Eye, Heart } from 'lucide-react'
import { cn } from '../../utils/cn'
import { useProgress } from '../../context/ProgressContext'
import { courseCompletionRatio } from '../../utils/progressSchema'
import ProgressBar from './ProgressBar'
import DifficultyBadge from './DifficultyBadge'
import Button from '../ui/Button'
import type { CourseManifestItem } from '../../api/courses'

interface CourseCardProps {
  course: CourseManifestItem
  layout: 'grid' | 'list'
  onPreview: (courseId: string) => void
  isFavorite?: boolean
  onToggleFavorite?: (courseId: string) => void
}

export default function CourseCard({ course, layout, onPreview, isFavorite, onToggleFavorite }: CourseCardProps) {
  const { completed } = useProgress()
  const ratio = courseCompletionRatio([...completed], course.id, course.totalLessons)
  const hasStarted = ratio > 0
  const ctaLabel = ratio >= 1 ? 'Review' : hasStarted ? 'Resume' : 'Start'

  const thumbnail = (
    <div className={cn('bg-gradient-to-br from-orange-100 to-amber-50 overflow-hidden shrink-0 relative', layout === 'grid' ? 'aspect-video' : 'w-40 h-28 rounded-xl')}>
      {course.thumbnail ? (
        <img
          src={course.thumbnail}
          alt={course.title}
          className="w-full h-full object-cover"
          onError={(e) => {
            e.currentTarget.style.display = 'none'
          }}
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          <BookOpen className="w-8 h-8 text-orange-300" />
        </div>
      )}
      {onToggleFavorite && (
        <button
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            onToggleFavorite(course.id)
          }}
          className="absolute top-2 right-2 w-8 h-8 rounded-full bg-white/90 backdrop-blur-sm flex items-center justify-center hover:bg-white hover:scale-110 transition-all shadow-md"
          aria-label={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
        >
          <Heart
            className={cn('w-4 h-4 transition-all', isFavorite ? 'fill-red-500 text-red-500' : 'text-slate-400')}
          />
        </button>
      )}
    </div>
  )

  if (layout === 'list') {
    return (
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm flex gap-4 p-4 hover:shadow-md hover:border-orange-200 transition-all group">
        {thumbnail}
        <div className="flex-1 min-w-0 flex flex-col gap-2">
          <div className="flex items-start gap-2 flex-wrap">
            <DifficultyBadge level={course.difficulty} />
          </div>
          <h3 className="font-semibold text-slate-800 leading-snug group-hover:text-orange-600 transition-colors line-clamp-2">
            {course.title}
          </h3>
          <p className="text-sm text-slate-500 line-clamp-2 flex-1">{course.description}</p>
          <div className="flex items-center gap-3 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <Clock className="w-3.5 h-3.5" />
              {course.duration}
            </span>
            <span className="flex items-center gap-1">
              <BookOpen className="w-3.5 h-3.5" />
              {course.totalLessons} lessons
            </span>
          </div>
          {hasStarted && <ProgressBar value={ratio} className="max-w-xs" />}
        </div>
        <div className="flex flex-col gap-2 justify-center shrink-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPreview(course.id)}
            className="flex items-center gap-1"
          >
            <Eye className="w-4 h-4" />
            Preview
          </Button>
          <Link to={`/courses/${course.id}`}>
            <Button size="sm" className="w-full">
              {ctaLabel}
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden hover:shadow-md hover:border-orange-200 transition-all group flex flex-col">
      <div className="aspect-video bg-gradient-to-br from-orange-100 to-amber-50 overflow-hidden">
        {course.thumbnail ? (
          <img
            src={course.thumbnail}
            alt={course.title}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            onError={(e) => {
              e.currentTarget.style.display = 'none'
            }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <BookOpen className="w-12 h-12 text-orange-300" />
          </div>
        )}
      </div>

      <div className="p-4 flex flex-col gap-3 flex-1">
        <DifficultyBadge level={course.difficulty} />
        <h3 className="font-semibold text-slate-800 leading-snug line-clamp-2 group-hover:text-orange-600 transition-colors">
          {course.title}
        </h3>
        <p className="text-sm text-slate-500 line-clamp-2 flex-1">{course.description}</p>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span className="flex items-center gap-1">
            <Clock className="w-3.5 h-3.5" />
            {course.duration}
          </span>
          <span className="flex items-center gap-1">
            <BookOpen className="w-3.5 h-3.5" />
            {course.totalLessons} lessons
          </span>
        </div>
        {hasStarted && <ProgressBar value={ratio} />}
        <div className="flex gap-2 pt-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onPreview(course.id)}
            className="flex items-center gap-1"
          >
            <Eye className="w-4 h-4" />
            Preview
          </Button>
          <Link to={`/courses/${course.id}`} className="flex-1">
            <Button size="sm" className="w-full">
              {ctaLabel}
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
