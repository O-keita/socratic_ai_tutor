import { Link } from 'react-router-dom'
import { Brain, BookOpen, MessageCircle, Code, ChevronRight, TrendingUp, CheckCircle2, Zap, ArrowRight, GraduationCap, Sparkles, Clock, BarChart } from 'lucide-react'
import { useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { useProgress } from '../context/ProgressContext'
import { useManifest } from '../hooks/useManifest'
import { courseCompletionRatio } from '../utils/progressSchema'
import CourseCard from '../components/course/CourseCard'
import Spinner from '../components/ui/Spinner'

function getGreeting(): string {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 17) return 'Good afternoon'
  return 'Good evening'
}

const QUICK_TOOLS = [
  {
    to: '/chat',
    label: 'AI Tutor',
    desc: 'Learn through guided questions',
    icon: MessageCircle,
    color: 'bg-brand-600',
    lightColor: 'bg-brand-50',
    textColor: 'text-brand-600',
  },
  {
    to: '/playground',
    label: 'Playground',
    desc: 'Run Python in your browser',
    icon: Code,
    color: 'bg-emerald-600',
    lightColor: 'bg-emerald-50',
    textColor: 'text-emerald-600',
  },
  {
    to: '/quiz',
    label: 'Practice Quiz',
    desc: 'Test your knowledge',
    icon: Brain,
    color: 'bg-purple-600',
    lightColor: 'bg-purple-50',
    textColor: 'text-purple-600',
  },
  {
    to: '/glossary',
    label: 'Glossary',
    desc: 'Key ML/DS terms',
    icon: BookOpen,
    color: 'bg-accent-600',
    lightColor: 'bg-accent-50',
    textColor: 'text-accent-600',
  },
]

export default function HomePage() {
  const { user } = useAuth()
  const { completed } = useProgress()
  const { data, isLoading } = useManifest()

  // Favorites stored in localStorage
  const [favorites, setFavorites] = useState<string[]>(() => {
    const stored = localStorage.getItem('favorite_courses')
    return stored ? JSON.parse(stored) : []
  })

  const toggleFavorite = (courseId: string) => {
    const updated = favorites.includes(courseId)
      ? favorites.filter((id) => id !== courseId)
      : [...favorites, courseId]
    setFavorites(updated)
    localStorage.setItem('favorite_courses', JSON.stringify(updated))
  }

  const courses = data?.courses ?? []
  const completedArr = [...completed]
  
  // Calculate stats
  const coursesStarted = courses.filter((c) => completedArr.some((id) => id.startsWith(c.id))).length
  const coursesCompleted = courses.filter(
    (c) => courseCompletionRatio(completedArr, c.id, c.totalLessons) >= 1,
  ).length

  // Get course sections
  const continueCourse = courses.find((c) => {
    const ratio = courseCompletionRatio(completedArr, c.id, c.totalLessons)
    return ratio > 0 && ratio < 1
  })

  const inProgress = courses.filter((c) => {
    const ratio = courseCompletionRatio(completedArr, c.id, c.totalLessons)
    return ratio > 0 && ratio < 1
  })

  // Mock "most popular" - first 3 courses
  const mostPopular = courses.slice(0, 3)
  
  // Mock "just added" - last 2 courses
  const justAdded = courses.slice(-2)

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-6 py-6 md:py-8 space-y-8">

      {/* ── Hero Section with Featured Course ────────────────────── */}
      {courses.length > 0 && (
        <div className="relative bg-gradient-to-r from-purple-600 via-pink-500 to-orange-500 rounded-2xl overflow-hidden min-h-[280px] md:min-h-[320px]">
          {/* Decorative Background */}
          <div className="absolute inset-0 bg-gradient-to-br from-black/20 to-transparent" />
          <div className="absolute -top-20 -right-20 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-16 -left-16 w-72 h-72 bg-white/10 rounded-full blur-3xl" />
          
          <div className="relative z-10 p-6 md:p-10 flex flex-col md:flex-row items-start md:items-center gap-6">
            <div className="flex-1">
              <div className="inline-flex items-center gap-1.5 bg-white/20 backdrop-blur-sm text-white/95 text-xs font-bold px-3 py-1.5 rounded-full mb-4 border border-white/20">
                <Sparkles className="w-3.5 h-3.5" />
                PROFESSIONAL CERTIFICATE
              </div>
              <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white leading-tight mb-3">
                {courses[0]?.title || 'Machine Learning Fundamentals'}
              </h1>
              <p className="text-white/80 text-sm md:text-base mb-6 max-w-2xl leading-relaxed">
                {courses[0]?.description || 'Learn the core principles of building, optimizing, and deploying deep learning models using PyTorch.'}
              </p>
              
              <div className="flex flex-wrap items-center gap-4 mb-6">
                <div className="flex items-center gap-2 text-white/90 text-sm">
                  <Clock className="w-4 h-4" />
                  <span className="font-medium">{courses[0]?.duration || '10-15 hours'}</span>
                </div>
                <div className="flex items-center gap-2 text-white/90 text-sm">
                  <BarChart className="w-4 h-4" />
                  <span className="font-medium">{courses[0]?.difficulty || 'Intermediate'}</span>
                </div>
              </div>

              <div className="flex flex-wrap gap-3">
                <Link
                  to={courses[0] ? `/courses/${courses[0].id}` : '/browse'}
                  className="inline-flex items-center gap-2 bg-white text-purple-700 font-bold px-6 py-3 rounded-xl hover:bg-white/95 shadow-lg hover:shadow-xl transition-all text-sm"
                >
                  {continueCourse ? 'Continue Learning' : 'Enroll Now'}
                  <ArrowRight className="w-4 h-4" />
                </Link>
                <Link
                  to="/browse"
                  className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm text-white font-semibold px-6 py-3 rounded-xl hover:bg-white/20 border border-white/20 transition-all text-sm"
                >
                  Explore All Courses
                </Link>
              </div>
            </div>

            {/* Stats Card */}
            <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl p-6 min-w-[240px]">
              <p className="text-white/70 text-xs font-semibold uppercase tracking-wide mb-4">Your Progress</p>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5 text-emerald-300" />
                    <span className="text-white text-sm">Lessons</span>
                  </div>
                  <span className="text-2xl font-bold text-white">{completed.size}</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-blue-300" />
                    <span className="text-white text-sm">In Progress</span>
                  </div>
                  <span className="text-2xl font-bold text-white">{coursesStarted}</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <GraduationCap className="w-5 h-5 text-purple-300" />
                    <span className="text-white text-sm">Completed</span>
                  </div>
                  <span className="text-2xl font-bold text-white">{coursesCompleted}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Quick Tools ─────────────────────────────────────────── */}
      <div>
        <h2 className="text-xl font-bold text-slate-900 mb-5 flex items-center gap-2">
          <Zap className="w-5 h-5 text-brand-600" />
          Learning Tools
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { to: '/chat', label: 'AI Tutor', desc: 'Learn through guided questions', icon: MessageCircle, color: 'from-brand-500 to-orange-500' },
            { to: '/playground', label: 'Playground', desc: 'Run Python in your browser', icon: Code, color: 'from-emerald-500 to-teal-500' },
            { to: '/quiz', label: 'Practice Quiz', desc: 'Test your knowledge', icon: Brain, color: 'from-purple-500 to-pink-500' },
            { to: '/glossary', label: 'Glossary', desc: 'Key ML/DS terms', icon: BookOpen, color: 'from-blue-500 to-indigo-500' },
          ].map(({ to, label, desc, icon: Icon, color }) => (
            <Link
              key={to}
              to={to}
              className="group bg-white rounded-xl border border-slate-200 p-5 hover:shadow-lg hover:border-brand-200 hover:-translate-y-0.5 transition-all"
            >
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center mb-4 group-hover:scale-105 transition-transform shadow-lg`}>
                <Icon className="w-6 h-6 text-white" />
              </div>
              <p className="font-bold text-slate-800 mb-1">{label}</p>
              <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
            </Link>
          ))}
        </div>
      </div>

      {/* ── Most Popular ─────────────────────────────────────────── */}
      {mostPopular.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-brand-600" />
              Most Popular
            </h2>
            <Link to="/browse" className="text-sm text-brand-600 hover:text-brand-700 font-semibold flex items-center gap-1 group">
              See all
              <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
          </div>
          {isLoading ? (
            <div className="flex justify-center py-16"><Spinner size="lg" /></div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {mostPopular.map((c) => (
                <CourseCard
                  key={c.id}
                  course={c}
                  layout="grid"
                  onPreview={() => {}}
                  isFavorite={favorites.includes(c.id)}
                  onToggleFavorite={toggleFavorite}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Continue Learning ─────────────────────────────────────── */}
      {inProgress.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <GraduationCap className="w-5 h-5 text-brand-600" />
              Continue Learning
            </h2>
            <Link to="/profile" className="text-sm text-brand-600 hover:text-brand-700 font-semibold flex items-center gap-1 group">
              View progress
              <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {inProgress.slice(0, 3).map((c) => (
              <CourseCard
                key={c.id}
                course={c}
                layout="grid"
                onPreview={() => {}}
                isFavorite={favorites.includes(c.id)}
                onToggleFavorite={toggleFavorite}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Just Added ─────────────────────────────────────────── */}
      {justAdded.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-brand-600" />
              Just Added
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {justAdded.map((c) => (
              <CourseCard
                key={c.id}
                course={c}
                layout="list"
                onPreview={() => {}}
                isFavorite={favorites.includes(c.id)}
                onToggleFavorite={toggleFavorite}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Empty State ─────────────────────────────────────────── */}
      {courses.length === 0 && !isLoading && (
        <div className="text-center py-16">
          <div className="w-20 h-20 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-4">
            <BookOpen className="w-10 h-10 text-slate-400" />
          </div>
          <h3 className="text-xl font-bold text-slate-800 mb-2">No courses available</h3>
          <p className="text-slate-500 mb-6">Check back soon for new learning content!</p>
        </div>
      )}
    </div>
  )
}
