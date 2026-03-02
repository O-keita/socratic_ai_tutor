import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ChevronRight, CheckCircle2, Circle, BookOpen } from 'lucide-react'
import { cn } from '../../utils/cn'
import { useProgress } from '../../context/ProgressContext'
import type { CourseDetail } from '../../api/courses'

interface SyllabusTreeProps {
  course: CourseDetail
  interactive?: boolean // if false, no navigation links
  defaultExpandedModule?: number
}

export default function SyllabusTree({
  course,
  interactive = true,
  defaultExpandedModule = 0,
}: SyllabusTreeProps) {
  const { completed } = useProgress()
  const [expandedModule, setExpandedModule] = useState<number>(defaultExpandedModule)
  const [expandedChapters, setExpandedChapters] = useState<Set<string>>(
    () => new Set(course.modules[0]?.chapters.map((c) => c.id) ?? []),
  )

  const toggleChapter = (chapterId: string) => {
    setExpandedChapters((prev) => {
      const next = new Set(prev)
      if (next.has(chapterId)) next.delete(chapterId)
      else next.add(chapterId)
      return next
    })
  }

  return (
    <div className="space-y-2">
      {course.modules.map((mod, modIdx) => {
        const modCompleted = mod.chapters
          .flatMap((c) => c.lessons)
          .every((l) => completed.has(l.id))

        return (
          <div key={mod.id} className="border border-slate-200 rounded-xl overflow-hidden bg-white">
            <button
              onClick={() => setExpandedModule(expandedModule === modIdx ? -1 : modIdx)}
              className="w-full flex items-center gap-3 px-4 py-3 bg-white hover:bg-orange-50/50 transition-colors text-left"
            >
              <ChevronRight
                className={cn(
                  'w-4 h-4 text-slate-400 transition-transform',
                  expandedModule === modIdx && 'rotate-90',
                )}
              />
              <span className="flex-1 font-medium text-slate-700 text-sm">{mod.title}</span>
              {modCompleted && <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />}
            </button>

            {expandedModule === modIdx && (
              <div className="bg-warm-100/50">
                {mod.chapters.map((ch) => {
                  const chCompleted = ch.lessons.every((l) => completed.has(l.id))
                  const expanded = expandedChapters.has(ch.id)

                  return (
                    <div key={ch.id} className="border-t border-slate-100">
                      <button
                        onClick={() => toggleChapter(ch.id)}
                        className="w-full flex items-center gap-3 px-6 py-2.5 hover:bg-orange-50/50 transition-colors text-left"
                      >
                        <ChevronRight
                          className={cn(
                            'w-3.5 h-3.5 text-slate-400 transition-transform',
                            expanded && 'rotate-90',
                          )}
                        />
                        <span className="flex-1 text-sm text-slate-600">{ch.title}</span>
                        {chCompleted && (
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                        )}
                      </button>

                      {expanded && (
                        <div className="bg-white/60">
                          {ch.lessons.map((lesson) => {
                            const isDone = completed.has(lesson.id)
                            const inner = (
                              <div className="flex items-center gap-3 px-8 py-2 group">
                                {isDone ? (
                                  <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                                ) : (
                                  <Circle className="w-4 h-4 text-slate-300 shrink-0 group-hover:text-orange-400" />
                                )}
                                <span
                                  className={cn(
                                    'text-sm',
                                    isDone ? 'text-slate-400 line-through' : 'text-slate-600',
                                    interactive && 'group-hover:text-orange-500',
                                  )}
                                >
                                  {lesson.title}
                                </span>
                              </div>
                            )

                            return (
                              <div key={lesson.id} className="border-t border-slate-100/60">
                                {interactive ? (
                                  <Link
                                    to={`/courses/${course.id}/lessons/${lesson.id}`}
                                    className="block hover:bg-orange-50/50 transition-colors"
                                  >
                                    {inner}
                                  </Link>
                                ) : (
                                  <div>{inner}</div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )
      })}

      {course.modules.length === 0 && (
        <div className="text-center py-8 text-slate-400">
          <BookOpen className="w-8 h-8 mx-auto mb-2 opacity-40" />
          <p className="text-sm">No modules available</p>
        </div>
      )}
    </div>
  )
}
