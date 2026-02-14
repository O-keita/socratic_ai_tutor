import 'package:shared_preferences/shared_preferences.dart';

class ProgressService {
  static const String _progressKey = 'lesson_progress';
  static SharedPreferences? _prefs;
  
  // Initialize SharedPreferences with retry logic
  static Future<SharedPreferences?> _getPrefs() async {
    if (_prefs != null) return _prefs;
    
    // Retry up to 5 times with delay for emulator startup
    for (int i = 0; i < 5; i++) {
      try {
        _prefs = await SharedPreferences.getInstance();
        return _prefs;
      } catch (e) {
        if (i == 4) {
          print('ProgressService: Could not initialize SharedPreferences: $e');
          return null;
        }
        await Future.delayed(Duration(seconds: 1));
      }
    }
    return null;
  }
  
  // Save completed lesson
  static Future<void> markLessonComplete(String lessonId) async {
    try {
      final prefs = await _getPrefs();
      if (prefs == null) return;
      
      final progress = prefs.getStringList(_progressKey) ?? [];
      if (!progress.contains(lessonId)) {
        progress.add(lessonId);
        await prefs.setStringList(_progressKey, progress);
      }
    } catch (e) {
      print('ProgressService: Error marking lesson complete: $e');
    }
  }

  // Check if lesson is completed
  static Future<bool> isLessonCompleted(String lessonId) async {
    try {
      final prefs = await _getPrefs();
      if (prefs == null) return false;
      
      final progress = prefs.getStringList(_progressKey) ?? [];
      return progress.contains(lessonId);
    } catch (e) {
      print('ProgressService: Error checking lesson: $e');
      return false;
    }
  }

  // Get all completed lessons
  static Future<List<String>> getCompletedLessons() async {
    try {
      final prefs = await _getPrefs();
      if (prefs == null) return [];
      
      return prefs.getStringList(_progressKey) ?? [];
    } catch (e) {
      print('ProgressService: Error getting completed lessons: $e');
      return [];
    }
  }

  // Reset progress
  static Future<void> resetProgress() async {
    try {
      final prefs = await _getPrefs();
      if (prefs == null) return;
      await prefs.remove(_progressKey);
    } catch (e) {
      print('ProgressService: Error resetting progress: $e');
    }
  }

  // Get course progress percentage
  static Future<double> getCourseProgress(String courseId, int totalLessons) async {
    try {
      final prefs = await _getPrefs();
      if (prefs == null) return 0;
      final progress = prefs.getStringList(_progressKey) ?? [];
      final courseLessons = progress.where((id) => id.startsWith(courseId)).length;
      if (totalLessons == 0) return 0;
      return courseLessons / totalLessons;
    } catch (e) {
      print('ProgressService: Error getting course progress: $e');
      return 0;
    }
  }
}
