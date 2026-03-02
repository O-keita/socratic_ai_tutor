import { apiClient } from './client'
import type { ProgressData } from '../utils/progressSchema'

export const getProgress = (): Promise<ProgressData> =>
  apiClient.get<ProgressData>('/user/progress').then((r) => r.data)

export const saveProgress = (data: ProgressData): Promise<void> =>
  apiClient.post('/user/progress', data).then(() => undefined)
