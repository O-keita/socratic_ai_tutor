import type { CourseDetail, Module, Chapter, Lesson } from '../api/courses'

export interface LessonLocation {
  lesson: Lesson
  chapter: Chapter
  module: Module
  lessonIndex: number
  allLessons: Lesson[]
}

export function flattenLessons(course: CourseDetail): Lesson[] {
  const all: Lesson[] = []
  for (const mod of course.modules) {
    for (const ch of mod.chapters) {
      all.push(...ch.lessons)
    }
  }
  return all
}

export function findLesson(course: CourseDetail, lessonId: string): LessonLocation | null {
  const allLessons = flattenLessons(course)
  const lessonIndex = allLessons.findIndex((l) => l.id === lessonId)
  if (lessonIndex === -1) return null

  for (const mod of course.modules) {
    for (const ch of mod.chapters) {
      const lesson = ch.lessons.find((l) => l.id === lessonId)
      if (lesson) return { lesson, chapter: ch, module: mod, lessonIndex, allLessons }
    }
  }
  return null
}
