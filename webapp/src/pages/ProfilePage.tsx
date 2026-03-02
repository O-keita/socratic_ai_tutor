import { BookOpen, CheckCircle2, TrendingUp } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useProgress } from '../context/ProgressContext'
import { useManifest } from '../hooks/useManifest'
import { courseCompletionRatio } from '../utils/progressSchema'
import ProgressBar from '../components/course/ProgressBar'
import DifficultyBadge from '../components/course/DifficultyBadge'
import Spinner from '../components/ui/Spinner'
import { Link } from 'react-router-dom'

export default function ProfilePage() {
  const { user } = useAuth()
  const { completed } = useProgress()
  const { data, isLoading } = useManifest()

  const courses = data?.courses ?? []
  const completedArr = [...completed]
  const coursesStarted = courses.filter((c) => completedArr.some((id) => id.startsWith(c.id))).length
  const coursesCompleted = courses.filter(
    (c) => courseCompletionRatio(completedArr, c.id, c.totalLessons) >= 1,
  ).length

  const initial = user?.username?.charAt(0)?.toUpperCase() ?? 'U'

  return (
    <div className="max-w-3xl mx-auto p-4 md:p-8 space-y-8">
      {/* Hero header */}
      <div className="bg-gradient-to-r from-orange-500 to-amber-500 rounded-2xl p-6 md:p-8 text-white">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center text-2xl font-bold">
            {initial}
          </div>
          <div>
            <h1 className="text-2xl font-bold">{user?.username}</h1>
            <p className="text-orange-100 text-sm">Learning with the Socratic method</p>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { icon: CheckCircle2, label: 'Lessons Done', value: completed.size, color: 'text-emerald-500', bg: 'bg-emerald-50' },
          { icon: BookOpen, label: 'Courses Started', value: coursesStarted, color: 'text-orange-500', bg: 'bg-orange-50' },
          { icon: TrendingUp, label: 'Completed', value: coursesCompleted, color: 'text-blue-500', bg: 'bg-blue-50' },
        ].map(({ icon: Icon, label, value, color, bg }) => (
          <div key={label} className="bg-white rounded-2xl border border-slate-100 shadow-sm p-4 text-center">
            <div className={`w-10 h-10 rounded-xl ${bg} flex items-center justify-center mx-auto mb-2`}>
              <Icon className={`w-5 h-5 ${color}`} />
            </div>
            <div className="text-2xl font-bold text-slate-800">{value}</div>
            <div className="text-xs text-slate-500 mt-0.5">{label}</div>
          </div>
        ))}
      </div>

      {/* Course progress */}
      <div>
        <h2 className="text-lg font-bold text-slate-800 mb-4">Course Progress</h2>
        {isLoading ? (
          <div className="flex justify-center py-8">
            <Spinner />
          </div>
        ) : courses.length === 0 ? (
          <p className="text-slate-400 text-sm">No courses available</p>
        ) : (
          <div className="space-y-3">
            {courses.map((c) => {
              const ratio = courseCompletionRatio(completedArr, c.id, c.totalLessons)
              const done = Math.round(ratio * c.totalLessons)
              return (
                <Link
                  key={c.id}
                  to={`/courses/${c.id}`}
                  className="block bg-white rounded-2xl border border-slate-100 shadow-sm p-4 hover:shadow-md hover:border-orange-200 transition-all"
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-xl bg-orange-50 flex items-center justify-center shrink-0">
                      <BookOpen className="w-5 h-5 text-orange-500" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-slate-700 text-sm line-clamp-1">{c.title}</h3>
                      <div className="flex items-center gap-2 mt-0.5">
                        <DifficultyBadge level={c.difficulty} />
                      </div>
                    </div>
                    <span className="text-sm font-bold text-orange-500 shrink-0">
                      {done}/{c.totalLessons}
                    </span>
                  </div>
                  <ProgressBar value={ratio} showLabel />
                </Link>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
