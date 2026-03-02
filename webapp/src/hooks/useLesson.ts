import { useQuery } from '@tanstack/react-query'
import { fetchLesson } from '../api/courses'

export function useLesson(courseId: string | null | undefined, contentPath: string | null | undefined) {
  return useQuery({
    queryKey: ['lesson', courseId, contentPath],
    queryFn: () => fetchLesson(courseId!, contentPath!),
    enabled: !!courseId && !!contentPath,
    staleTime: 30 * 60 * 1000,
  })
}
