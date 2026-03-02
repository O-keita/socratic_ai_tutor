export interface ProgressData {
  completedLessons: string[]
  lastVisited?: { courseId: string; lessonId: string }
}

export const EMPTY_PROGRESS: ProgressData = { completedLessons: [] }

export function courseCompletionRatio(
  completedLessons: string[],
  courseId: string,
  totalLessons: number,
): number {
  if (!totalLessons) return 0
  return completedLessons.filter((id) => id.startsWith(courseId)).length / totalLessons
}
