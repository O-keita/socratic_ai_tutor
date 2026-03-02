import { useQuery } from '@tanstack/react-query'
import { fetchCourse } from '../api/courses'

export function useCourse(courseId: string | null | undefined) {
  return useQuery({
    queryKey: ['course', courseId],
    queryFn: () => fetchCourse(courseId!),
    enabled: !!courseId,
    staleTime: 10 * 60 * 1000,
  })
}
