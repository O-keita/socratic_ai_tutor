import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import '../models/course.dart';
import '../utils/app_config.dart';
import 'progress_service.dart';

class CourseService {
  static final CourseService _instance = CourseService._internal();
  factory CourseService() => _instance;
  CourseService._internal();

  List<Course>? _cachedCourses;

  String get _remoteManifestUrl => '${AppConfig.backendUrl}/content/manifest';

  /// Clear the internal course cache
  void clearCache() {
    _cachedCourses = null;
  }

  /// Load only bundled (asset + locally cached) courses. Fast — no network.
  Future<List<Course>> getCourses({bool forceRefresh = false}) async {
    if (_cachedCourses != null && !forceRefresh) {
      return _cachedCourses!;
    }

    debugPrint('CourseService: Loading local courses...');
    List<Course> courses = [];

    try {
      final String jsonString =
          await rootBundle.loadString('assets/courses/courses.json');
      final Map<String, dynamic> data = json.decode(jsonString);
      final List<dynamic> assetCourses = data['courses'];
      debugPrint('CourseService: Found ${assetCourses.length} courses in assets');

      for (var info in assetCourses) {
        final course = await loadCourse(info['id']);
        if (course != null) {
          courses.add(course);
        }
      }
    } catch (e) {
      debugPrint('CourseService: Error loading asset courses: $e');
    }

    _cachedCourses = courses;
    return courses;
  }

  /// Fetch courses from the admin backend and merge any that aren't already
  /// in [existing]. Returns only the newly added courses.
  /// Throws on network error so the caller can show feedback.
  Future<List<Course>> fetchRemoteCourses(List<Course> existing) async {
    debugPrint('CourseService: Fetching remote manifest...');
    final response = await http
        .get(Uri.parse(_remoteManifestUrl))
        .timeout(const Duration(seconds: 10));

    if (response.statusCode != 200) {
      throw Exception('Server returned ${response.statusCode}');
    }

    final manifest = json.decode(response.body);
    final List<dynamic> remoteList = manifest['courses'] ?? [];
    final List<Course> newCourses = [];

    for (var info in remoteList) {
      if (!existing.any((c) => c.id == info['id'])) {
        final course = await loadCourse(info['id']);
        if (course != null) {
          newCourses.add(course);
          debugPrint('CourseService: Fetched remote course: ${course.title}');
        }
      }
    }

    // Merge into the in-memory cache
    if (newCourses.isNotEmpty && _cachedCourses != null) {
      _cachedCourses = [..._cachedCourses!, ...newCourses];
    }

    return newCourses;
  }

  /// Load a full course, checking local storage first then assets
  Future<Course?> loadCourse(String courseId) async {
    try {
      String? jsonString;

      // 1. Check local storage (previously downloaded content)
      try {
        final dir = await getApplicationSupportDirectory()
            .timeout(const Duration(milliseconds: 500));
        final localFile =
            File(p.join(dir.path, 'courses', courseId, 'course.json'));
        if (await localFile.exists()) {
          jsonString = await localFile.readAsString();
          // Discard stubs that lack modules
          if (!jsonString.contains('"modules"')) jsonString = null;
        }
      } catch (_) {
        debugPrint('CourseService: Local storage check skipped for $courseId');
      }

      // 2. Fallback to bundled assets
      if (jsonString == null) {
        try {
          jsonString = await rootBundle
              .loadString('assets/courses/$courseId/course.json');
        } catch (e) {
          // 3. Last resort: remote fetch
          final remoteUrl =
              '${AppConfig.backendUrl}/content/$courseId/course.json';
          final response = await http
              .get(Uri.parse(remoteUrl))
              .timeout(const Duration(seconds: 5));

          if (response.statusCode == 200) {
            jsonString = response.body;
            // Best-effort local cache
            try {
              final dir = await getApplicationSupportDirectory()
                  .timeout(const Duration(milliseconds: 500));
              final localFile =
                  File(p.join(dir.path, 'courses', courseId, 'course.json'));
              await localFile.parent.create(recursive: true);
              await localFile.writeAsString(jsonString);
            } catch (_) {}
          } else {
            throw Exception(
                'Course $courseId not found in storage, assets, or remote.');
          }
        }
      }

      final Map<String, dynamic> data = json.decode(jsonString);
      final course = Course.fromJson(data);
      await _loadCompletionStatus(course);
      return course;
    } catch (e) {
      debugPrint('CourseService: Error loading course $courseId: $e');
      return null;
    }
  }

  /// Download a course and save it to local storage
  Future<bool> downloadCourse(String courseId, String manifestUrl) async {
    try {
      final response = await http.get(Uri.parse(manifestUrl));
      if (response.statusCode == 200) {
        final dir = await getApplicationSupportDirectory();
        final courseDir =
            Directory(p.join(dir.path, 'courses', courseId));
        if (!await courseDir.exists()) await courseDir.create(recursive: true);

        await File(p.join(courseDir.path, 'course.json'))
            .writeAsString(response.body);

        _cachedCourses = null;
        return true;
      }
    } catch (e) {
      debugPrint('CourseService: Error downloading course $courseId: $e');
    }
    return false;
  }

  /// Load the markdown content for a lesson
  Future<String> loadLessonContent(Lesson lesson, {String? courseId}) async {
    if (lesson.content.isNotEmpty) return lesson.content;
    if (lesson.contentFile == null) return 'No content available.';

    try {
      // 1. Local storage
      if (courseId != null) {
        try {
          final dir = await getApplicationSupportDirectory()
              .timeout(const Duration(milliseconds: 500));
          final localFile = File(
              p.join(dir.path, 'courses', courseId, lesson.contentFile!));
          if (await localFile.exists()) {
            final content = await localFile.readAsString();
            lesson.content = content;
            return content;
          }
        } catch (_) {
          debugPrint(
              'CourseService: Local lesson content check skipped for ${lesson.id}');
        }
      }

      // 2. Assets
      final assetPath = courseId != null
          ? 'assets/courses/$courseId/${lesson.contentFile}'
          : 'assets/courses/${lesson.contentFile}';

      try {
        final content = await rootBundle.loadString(assetPath);
        lesson.content = content;
        return content;
      } catch (_) {
        // 3. Remote
        if (courseId != null) {
          final fileName = p.basename(lesson.contentFile!);
          final remoteUrl =
              '${AppConfig.backendUrl}/content/$courseId/lessons/$fileName';
          final response = await http
              .get(Uri.parse(remoteUrl))
              .timeout(const Duration(seconds: 5));
          if (response.statusCode == 200) {
            lesson.content = response.body;
            return response.body;
          }
        }
        rethrow;
      }
    } catch (e, stack) {
      debugPrint('CourseService: Error loading lesson content: $e\n$stack');
      return 'Error loading content. Please check your internet connection.';
    }
  }

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
      debugPrint('CourseService: Could not load completion status: $e');
    }
  }

  Future<Course?> refreshCourse(String courseId) async {
    _cachedCourses = null;
    return loadCourse(courseId);
  }
}
