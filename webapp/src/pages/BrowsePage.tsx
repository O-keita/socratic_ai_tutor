import { useMemo, useState } from 'react'
import { Search as SearchIcon, Sparkles, BookOpen, Heart, Clock } from 'lucide-react'
import { useManifest } from '../hooks/useManifest'
import { useProgress } from '../context/ProgressContext'
import SearchBar from '../components/browse/SearchBar'
import FilterChips from '../components/browse/FilterChips'
import GridListToggle from '../components/browse/GridListToggle'
import CourseCard from '../components/course/CourseCard'
import CoursePreviewModal from '../components/course/CoursePreviewModal'
import Spinner from '../components/ui/Spinner'
import ErrorBanner from '../components/ui/ErrorBanner'
import type { CourseManifestItem } from '../api/courses'

type Tab = 'discover' | 'all' | 'favorites' | 'in-progress'

const DIFFICULTY_OPTIONS = [
  { label: 'Beginner', value: 'Beginner' },
  { label: 'Intermediate', value: 'Intermediate' },
  { label: 'Advanced', value: 'Advanced' },
  { label: 'All levels', value: 'Beginner to Advanced' },
]

const DURATION_OPTIONS = [
  { label: '< 5h', value: 'short' },
  { label: '5–20h', value: 'medium' },
  { label: '> 20h', value: 'long' },
]

function matchesDuration(duration: string, filter: string): boolean {
  const lower = duration.toLowerCase()
  const hours = parseFloat(lower)

  if (filter === 'short') {
    if (lower.includes('min') || lower.includes('minute')) return true
    return !isNaN(hours) && hours < 5
  }
  if (filter === 'medium') return !isNaN(hours) && hours >= 5 && hours <= 20
  if (filter === 'long') {
    if (lower.includes('+')) return true
    return !isNaN(hours) && hours > 20
  }
  return true
}

export default function BrowsePage() {
  const { data, isLoading, error, refetch } = useManifest()
  const { progress } = useProgress()
  const [activeTab, setActiveTab] = useState<Tab>('discover')
  const [search, setSearch] = useState('')
  const [diffFilter, setDiffFilter] = useState('All')
  const [durationFilter, setDurationFilter] = useState('All')
  const [layout, setLayout] = useState<'grid' | 'list'>('grid')
  const [previewId, setPreviewId] = useState<string | null>(null)

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

  // Filter logic based on active tab
  const filtered = useMemo<CourseManifestItem[]>(() => {
    let list = data?.courses ?? []

    // Tab filtering
    if (activeTab === 'favorites') {
      list = list.filter((c) => favorites.includes(c.id))
    } else if (activeTab === 'in-progress') {
      list = list.filter((c) => {
        const p = progress[c.id]
        return p && p.completed > 0 && p.completed < p.total
      })
    } else if (activeTab === 'discover') {
      // Show featured/recommended courses (for now, just top 6)
      list = list.slice(0, 6)
    }

    // Search filter
    if (search) {
      const q = search.toLowerCase()
      list = list.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          c.description.toLowerCase().includes(q),
      )
    }

    // Difficulty filter
    if (diffFilter !== 'All') {
      list = list.filter((c) => c.difficulty === diffFilter)
    }

    // Duration filter
    if (durationFilter !== 'All') {
      list = list.filter((c) => matchesDuration(c.duration, durationFilter))
    }

    return list
  }, [data, search, diffFilter, durationFilter, activeTab, favorites, progress])

  const tabs: { id: Tab; label: string; icon: typeof Sparkles }[] = [
    { id: 'discover', label: 'Discover', icon: Sparkles },
    { id: 'all', label: 'All Courses', icon: BookOpen },
    { id: 'favorites', label: 'Favorites', icon: Heart },
    { id: 'in-progress', label: 'In Progress', icon: Clock },
  ]

  return (
    <div className="p-4 md:p-8 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 to-amber-500 rounded-2xl p-6 md:p-8 text-white">
        <h1 className="text-2xl md:text-3xl font-bold mb-1">Explore Courses</h1>
        <p className="text-orange-100 text-sm md:text-base">
          Master data science and ML through guided discovery
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 overflow-x-auto scrollbar-hide border-b border-slate-200">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2 px-4 py-3 font-semibold text-sm whitespace-nowrap border-b-2 transition-colors ${
              activeTab === id
                ? 'border-brand-600 text-brand-600'
                : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      <SearchBar value={search} onChange={setSearch} />

      <div className="space-y-3">
        <FilterChips
          label="Level"
          options={DIFFICULTY_OPTIONS}
          value={diffFilter}
          onChange={setDiffFilter}
        />
        <FilterChips
          label="Duration"
          options={DURATION_OPTIONS}
          value={durationFilter}
          onChange={setDurationFilter}
        />
      </div>

      {isLoading ? (
        <div className="flex justify-center py-20">
          <Spinner size="lg" />
        </div>
      ) : error ? (
        <div className="py-8">
          <ErrorBanner
            message="Failed to load courses. Make sure the backend is running."
            onDismiss={() => refetch()}
          />
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between">
            <p className="text-sm text-slate-500 font-medium">
              {filtered.length} {filtered.length === 1 ? 'course' : 'courses'}
            </p>
            <GridListToggle layout={layout} onChange={setLayout} />
          </div>

          {filtered.length === 0 ? (
            <div className="text-center py-20 text-slate-400">
              <SearchIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p className="font-medium text-slate-600">
                {activeTab === 'favorites'
                  ? 'No favorite courses yet'
                  : activeTab === 'in-progress'
                  ? 'No courses in progress'
                  : 'No courses match your filters'}
              </p>
              {(search || diffFilter !== 'All' || durationFilter !== 'All') && (
                <button
                  onClick={() => {
                    setSearch('')
                    setDiffFilter('All')
                    setDurationFilter('All')
                  }}
                  className="text-sm text-orange-500 hover:text-orange-600 mt-2 font-medium"
                >
                  Clear all filters
                </button>
              )}
            </div>
          ) : layout === 'grid' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {filtered.map((c) => (
                <CourseCard
                  key={c.id}
                  course={c}
                  layout="grid"
                  onPreview={setPreviewId}
                  isFavorite={favorites.includes(c.id)}
                  onToggleFavorite={toggleFavorite}
                />
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {filtered.map((c) => (
                <CourseCard
                  key={c.id}
                  course={c}
                  layout="list"
                  onPreview={setPreviewId}
                  isFavorite={favorites.includes(c.id)}
                  onToggleFavorite={toggleFavorite}
                />
              ))}
            </div>
          )}
        </>
      )}

      <CoursePreviewModal courseId={previewId} onClose={() => setPreviewId(null)} />
    </div>
  )
}
