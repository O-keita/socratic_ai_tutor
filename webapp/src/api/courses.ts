import { apiClient } from './client'

export interface CourseManifestItem {
  id: string
  title: string
  description: string
  thumbnail: string
  difficulty: string
  duration: string
  totalLessons: number
}

export interface ManifestResponse {
  courses: CourseManifestItem[]
}

export interface Lesson {
  id: string
  title: string
  orderIndex: number
  contentPath?: string
  contentFile?: string
  keyPoints?: string
  reflectionQuestions?: string[]
  isCompleted?: boolean
}

export interface Chapter {
  id: string
  title: string
  orderIndex: number
  lessons: Lesson[]
}

export interface Module {
  id: string
  title: string
  description?: string
  orderIndex: number
  chapters: Chapter[]
}

export interface CourseDetail {
  id: string
  title: string
  description: string
  thumbnail?: string
  difficulty: string
  duration: string
  totalLessons: number
  modules: Module[]
}

export const fetchManifest = (): Promise<ManifestResponse> =>
  apiClient.get<ManifestResponse>('/content/manifest').then((r) => r.data)

export const fetchCourse = (id: string): Promise<CourseDetail> =>
  apiClient.get<CourseDetail>(`/content/${id}/course.json`).then((r) => r.data)

export const fetchLesson = (courseId: string, filename: string): Promise<string> =>
  apiClient
    .get<string>(`/content/${courseId}/lessons/${filename}`, {
      transformResponse: [(d) => d],
    })
    .then((r) => r.data)

export function getLessonContentPath(lesson: Lesson): string {
  return lesson.contentPath ?? lesson.contentFile ?? ''
}
