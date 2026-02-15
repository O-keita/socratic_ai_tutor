import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import '../models/course.dart';
import 'progress_service.dart';

class CourseService {
  static final CourseService _instance = CourseService._internal();
  factory CourseService() => _instance;
  CourseService._internal();

  List<Course>? _cachedCourses;
  static const String manifestUrl = 'http://10.0.2.2:8000/content/manifest';

  /// Clear the internal course cache
  void clearCache() {
    _cachedCourses = null;
  }

  /// Refresh courses from remote manifest and then load them
  Future<List<Course>> getCourses({bool forceRefresh = false}) async {
    if (_cachedCourses != null && !forceRefresh) {
      return _cachedCourses!;
    }

    print('CourseService: Loading courses...');
    List<Course> courses = [];
    
    // 1. Try to load from assets (bundled courses)
    try {
      final String jsonString = await rootBundle.loadString('assets/courses/courses.json');
      final Map<String, dynamic> data = json.decode(jsonString);
      final List<dynamic> assetCourses = data['courses'];
      print('CourseService: Found ${assetCourses.length} courses in assets');
      
      for (var info in assetCourses) {
        final course = await loadCourse(info['id']);
        if (course != null) {
          courses.add(course);
          print('CourseService: Successfully loaded asset course: ${course.title}');
        } else {
          print('CourseService: ⚠️ Failed to load asset course details for: ${info['id']}');
        }
      }
    } catch (e) {
      print('CourseService: Error loading asset courses: $e');
    }

    // 2. Try to fetch remote manifest for updates/new courses
    if (courses.isEmpty) {
      print('CourseService: ⚠️ Asset courses failed, trying remote manifest as fallback...');
      try {
        final response = await http.get(Uri.parse(manifestUrl)).timeout(const Duration(seconds: 3));
        if (response.statusCode == 200) {
          final manifest = json.decode(response.body);
          final List<dynamic> remoteCourses = manifest['courses'];
          
          for (var remoteInfo in remoteCourses) {
            if (!courses.any((c) => c.id == remoteInfo['id'])) {
              courses.add(Course(
                id: remoteInfo['id'],
                title: remoteInfo['title'],
                description: remoteInfo['description'],
                thumbnail: remoteInfo['thumbnail'] ?? '',
                totalLessons: remoteInfo['totalLessons'] ?? 0,
                difficulty: remoteInfo['difficulty'] ?? 'Online',
                duration: remoteInfo['duration'] ?? '-',
                modules: [],
              ));
            }
          }
          print('CourseService: Loaded ${courses.length} courses from remote');
        }
      } catch (e) {
        print('CourseService: Skipping remote manifest (Backend might be offline): $e');
      }
    }

    _cachedCourses = courses;
    return courses;
  }

  /// Load a full course, checking local storage first then assets
  Future<Course?> loadCourse(String courseId) async {
    try {
      String? jsonString;
      
      // 1. Check local storage (downloaded content)
      // For the prototype, we want to prioritize assets if local is old/broken
      try {
        final dir = await getApplicationSupportDirectory().timeout(const Duration(milliseconds: 500));
        final localFile = File(p.join(dir.path, 'courses', courseId, 'course.json'));
        if (await localFile.exists()) {
          // Only use local if it exists and we aren't forcing a refresh
          jsonString = await localFile.readAsString();
          
          // Basic validation: if it doesn't have modules, it's probably a stub
          if (!jsonString.contains('"modules"')) {
            jsonString = null;
          }
        }
      } catch (e) {
        print('CourseService: Local storage check skipped/failed for $courseId');
      }
      
      if (jsonString == null) {
        try {
          // 2. Fallback to assets
          jsonString = await rootBundle.loadString('assets/courses/$courseId/course.json');
        } catch (e) {
          // 3. Last fallback: Try to fetch from remote (for online-only courses)
          final remoteUrl = 'http://10.0.2.2:8000/content/$courseId/course.json';
          final response = await http.get(Uri.parse(remoteUrl)).timeout(const Duration(seconds: 5));
          
          if (response.statusCode == 200) {
            jsonString = response.body;
            // Best effort caching
            try {
              final dir = await getApplicationSupportDirectory().timeout(const Duration(milliseconds: 500));
              final localFile = File(p.join(dir.path, 'courses', courseId, 'course.json'));
              await localFile.parent.create(recursive: true);
              await localFile.writeAsString(jsonString);
            } catch (_) {}
          } else {
            throw Exception('Course $courseId not found in storage, assets, or remote server.');
          }
        }
      }

      final Map<String, dynamic> data = json.decode(jsonString);
      final course = Course.fromJson(data);
      await _loadCompletionStatus(course);
      return course;
    } catch (e) {
      print('CourseService: Error loading course $courseId: $e');
      return null;
    }
  }

  /// Download a course manifest and content
  Future<bool> downloadCourse(String courseId, String manifestUrl) async {
    try {
      final response = await http.get(Uri.parse(manifestUrl));
      if (response.statusCode == 200) {
        final dir = await getApplicationSupportDirectory();
        final courseDir = Directory(p.join(dir.path, 'courses', courseId));
        if (!await courseDir.exists()) await courseDir.create(recursive: true);
        
        final localFile = File(p.join(courseDir.path, 'course.json'));
        await localFile.writeAsString(response.body);
        
        // Clear cache so it reloads
        _cachedCourses = null;
        return true;
      }
    } catch (e) {
      print('CourseService: Error downloading course $courseId: $e');
    }
    return false;
  }

  /// Load the content of a lesson from its markdown file
  Future<String> loadLessonContent(Lesson lesson, {String? courseId}) async {
    if (lesson.content.isNotEmpty) {
      return lesson.content;
    }

    if (lesson.contentFile == null) {
      return 'No content available.';
    }

    try {
      // 1. Check local storage
      if (courseId != null) {
        try {
          final dir = await getApplicationSupportDirectory().timeout(const Duration(milliseconds: 500));
          final localFile = File(p.join(dir.path, 'courses', courseId, lesson.contentFile!));
          if (await localFile.exists()) {
            final content = await localFile.readAsString();
            lesson.content = content;
            return content;
          }
        } catch (e) {
          print('CourseService: Local lesson content check skipped for ${lesson.id}');
        }
      }

      // 2. Try assets
      String path;
      if (courseId != null) {
        path = 'assets/courses/$courseId/${lesson.contentFile}';
      } else {
        path = 'assets/courses/${lesson.contentFile}';
      }
      
      try {
        final String content = await rootBundle.loadString(path);
        lesson.content = content;
        return content;
      } catch (e) {
        // 3. Try Remote
        if (courseId != null) {
          final fileName = p.basename(lesson.contentFile!);
          final remoteUrl = 'http://10.0.2.2:8000/content/$courseId/lessons/$fileName';
          final response = await http.get(Uri.parse(remoteUrl)).timeout(const Duration(seconds: 5));
          
          if (response.statusCode == 200) {
            lesson.content = response.body;
            return response.body;
          }
        }
        rethrow;
      }
    } catch (e, stack) {
      print('Error loading lesson content: $e\n$stack');
      return 'Error loading content. Please check your internet connection.';
    }
  }

  /// Load completion status for all lessons in a course
  Future<void> _loadCompletionStatus(Course course) async {
    try {
      final completedLessons = await ProgressService.getCompletedLessons();

      for (var module in course.modules) {
        for (var chapter in module.chapters) {
          for (var lesson in chapter.lessons) {
            lesson.isCompleted = completedLessons.contains(lesson.id);
          }
        }
      }
    } catch (e) {
      // If progress service fails, just mark all as not completed
      print('CourseService: Could not load completion status: $e');
    }
  }

  /// Refresh a course's data
  Future<Course?> refreshCourse(String courseId) async {
    _cachedCourses = null;
    return loadCourse(courseId);
  }
}
