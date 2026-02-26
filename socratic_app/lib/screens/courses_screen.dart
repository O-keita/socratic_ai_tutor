import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../models/course.dart';
import '../services/course_service.dart';
import 'course_detail_screen.dart';

class CoursesScreen extends StatefulWidget {
  const CoursesScreen({super.key});

  @override
  State<CoursesScreen> createState() => _CoursesScreenState();
}

class _CoursesScreenState extends State<CoursesScreen> {
  final _courseService = CourseService();

  List<Course> _courses = [];
  bool _isLoading = true;
  bool _isSyncing = false;
  int _newCoursesFound = 0;
  String? _syncError;

  @override
  void initState() {
    super.initState();
    _loadCourses();
  }

  Future<void> _loadCourses({bool forceRefresh = false}) async {
    setState(() {
      _isLoading = true;
      _syncError = null;
      _newCoursesFound = 0;
    });
    try {
      final courses = await _courseService.getCourses(forceRefresh: forceRefresh);
      if (mounted) {
        setState(() {
          _courses = courses;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _syncRemoteCourses() async {
    if (_isSyncing) return;
    setState(() {
      _isSyncing = true;
      _syncError = null;
      _newCoursesFound = 0;
    });
    try {
      final added = await _courseService.fetchRemoteCourses(_courses);
      if (mounted) {
        setState(() {
          _courses = [..._courses, ...added];
          _newCoursesFound = added.length;
          _isSyncing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _syncError = 'Could not reach server. Check your connection.';
          _isSyncing = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: isDark
              ? AppTheme.backgroundGradient
              : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildHeader(),
              Expanded(child: _buildBody()),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'My Courses',
            style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
          ),
          const SizedBox(height: 4),
          Text(
            'Continue learning with AI guidance',
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                  color: AppTheme.textSecondary,
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_courses.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.cloud_off, size: 48, color: AppTheme.textSecondary),
              const SizedBox(height: 16),
              Text(
                'No courses found.',
                textAlign: TextAlign.center,
                style: Theme.of(context)
                    .textTheme
                    .bodyMedium
                    ?.copyWith(color: AppTheme.textSecondary),
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: () => _loadCourses(forceRefresh: true),
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    if (_courses.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.school_outlined,
                  size: 48, color: AppTheme.textSecondary),
              const SizedBox(height: 16),
              Text(
                'No courses available yet.',
                style: Theme.of(context)
                    .textTheme
                    .bodyMedium
                    ?.copyWith(color: AppTheme.textSecondary),
              ),
            ],
          ),
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: () async {
        _courseService.clearCache();
        await _loadCourses();
      },
      child: ListView.builder(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(20, 4, 20, 20),
        itemCount: _courses.length + 1, // +1 for sync button footer
        itemBuilder: (context, index) {
          if (index == _courses.length) return _buildSyncFooter();
          return _buildCourseCard(_courses[index]);
        },
      ),
    );
  }

  Widget _buildSyncFooter() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: Column(
        children: [
          if (_syncError != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                _syncError!,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 12, color: AppTheme.warning),
              ),
            ),
          if (_newCoursesFound > 0)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                '$_newCoursesFound new course${_newCoursesFound > 1 ? 's' : ''} added!',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 12,
                  color: AppTheme.success,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: _isSyncing ? null : _syncRemoteCourses,
              icon: _isSyncing
                  ? const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: AppTheme.accentOrange,
                      ),
                    )
                  : const Icon(Icons.cloud_download_outlined, size: 16),
              label: Text(
                _isSyncing ? 'Checking for new courses...' : 'Fetch new courses',
                style: const TextStyle(fontSize: 13),
              ),
              style: OutlinedButton.styleFrom(
                foregroundColor: AppTheme.accentOrange,
                side: BorderSide(
                  color: AppTheme.accentOrange.withValues(alpha: 0.5),
                ),
                padding: const EdgeInsets.symmetric(vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCourseCard(Course course) {
    return GestureDetector(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => CourseDetailScreen(course: course),
          ),
        );
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        decoration: BoxDecoration(
          color: AppTheme.surfaceCard,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: AppTheme.primaryLight.withValues(alpha: 0.3),
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 56,
                    height: 56,
                    decoration: BoxDecoration(
                      gradient: AppTheme.primaryGradient,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(
                      Icons.school,
                      color: Colors.white,
                      size: 28,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          course.title,
                          style:
                              Theme.of(context).textTheme.titleLarge?.copyWith(
                                    fontWeight: FontWeight.w600,
                                  ),
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            _buildTag(course.difficulty),
                            const SizedBox(width: 8),
                            _buildTag(course.duration),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const Icon(
                    Icons.arrow_forward_ios,
                    color: AppTheme.textSecondary,
                    size: 16,
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Text(
                course.description,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: AppTheme.textSecondary,
                    ),
              ),
              const SizedBox(height: 16),
              _buildProgressBar(course),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTag(String text) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: isDark ? AppTheme.primaryLight : AppTheme.tagBackground,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: isDark ? AppTheme.textPrimary : AppTheme.tagText,
          fontSize: 12,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }

  Widget _buildProgressBar(Course course) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final progress = course.progress;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              '${course.completedLessons}/${course.totalLessons} lessons',
              style: TextStyle(
                color:
                    isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                fontSize: 12,
              ),
            ),
            Text(
              '${(progress * 100).toInt()}%',
              style: TextStyle(
                color:
                    isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: progress,
            backgroundColor:
                isDark ? AppTheme.primaryLight : AppTheme.tagBackground,
            valueColor:
                const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
            minHeight: 6,
          ),
        ),
      ],
    );
  }
}
