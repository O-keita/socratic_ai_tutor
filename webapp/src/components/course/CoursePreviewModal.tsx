import { BookOpen } from 'lucide-react'
import Modal from '../ui/Modal'
import SyllabusTree from './SyllabusTree'
import Spinner from '../ui/Spinner'
import LessonReader from '../lesson/LessonReader'
import { useCourse } from '../../hooks/useCourse'
import { useLesson } from '../../hooks/useLesson'
import { flattenLessons } from '../../utils/lessonUrl'
import type { CourseDetail } from '../../api/courses'
import { getLessonContentPath as getPath } from '../../api/courses'

interface CoursePreviewModalProps {
  courseId: string | null
  onClose: () => void
}

function PreviewContent({ course }: { course: CourseDetail }) {
  const allLessons = flattenLessons(course)
  const firstLesson = allLessons[0]
  const contentPath = firstLesson ? getPath(firstLesson) : null

  const { data: lessonContent, isLoading: lessonLoading } = useLesson(
    course.id,
    contentPath,
  )

  return (
    <div className="flex h-full max-h-[70vh]">
      <div className="w-72 shrink-0 border-r border-slate-100 overflow-y-auto p-4 bg-warm-100/50">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Syllabus
        </h3>
        <SyllabusTree course={course} interactive={false} />
      </div>
      <div className="flex-1 overflow-y-auto p-6">
        {firstLesson && (
          <div className="mb-4 pb-4 border-b border-slate-100">
            <p className="text-xs text-orange-500 font-medium mb-1">First Lesson Preview</p>
            <h3 className="text-lg font-semibold text-slate-800">{firstLesson.title}</h3>
          </div>
        )}
        {lessonLoading ? (
          <div className="flex justify-center py-12">
            <Spinner />
          </div>
        ) : lessonContent ? (
          <LessonReader content={lessonContent} />
        ) : (
          <div className="text-center py-12 text-slate-400">
            <BookOpen className="w-8 h-8 mx-auto mb-2 opacity-40" />
            <p className="text-sm">Preview not available</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default function CoursePreviewModal({ courseId, onClose }: CoursePreviewModalProps) {
  const { data: course, isLoading } = useCourse(courseId)

  return (
    <Modal
      open={!!courseId}
      onClose={onClose}
      title={course?.title ?? 'Course Preview'}
      size="xl"
    >
      {isLoading ? (
        <div className="flex justify-center items-center py-16">
          <Spinner size="lg" />
        </div>
      ) : course ? (
        <PreviewContent course={course} />
      ) : null}
    </Modal>
  )
}
