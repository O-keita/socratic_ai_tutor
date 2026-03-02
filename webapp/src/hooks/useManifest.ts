import { useQuery } from '@tanstack/react-query'
import { fetchManifest } from '../api/courses'

export function useManifest() {
  return useQuery({
    queryKey: ['manifest'],
    queryFn: fetchManifest,
    staleTime: 10 * 60 * 1000,
  })
}
