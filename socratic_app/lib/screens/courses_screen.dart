import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../models/course.dart';
import '../data/course_data.dart';
import 'course_detail_screen.dart';

class CoursesScreen extends StatefulWidget {
  const CoursesScreen({super.key});

  @override
  State<CoursesScreen> createState() => _CoursesScreenState();
}

class _CoursesScreenState extends State<CoursesScreen> {
  List<Course> _courses = [];

  @override
  void initState() {
    super.initState();
    _courses = CourseData.getCourses();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildHeader(),
              Expanded(
                child: ListView.builder(
                  padding: const EdgeInsets.all(20),
                  itemCount: _courses.length,
                  itemBuilder: (context, index) {
                    return _buildCourseCard(_courses[index]);
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'My Courses',
            style: Theme.of(context).textTheme.headlineLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
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
            color: AppTheme.primaryLight.withOpacity(0.3),
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
                          style: Theme.of(context).textTheme.titleLarge?.copyWith(
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
                color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                fontSize: 12,
              ),
            ),
            Text(
              '${(progress * 100).toInt()}%',
              style: TextStyle(
                color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
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
            backgroundColor: isDark ? AppTheme.primaryLight : AppTheme.tagBackground,
            valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
            minHeight: 6,
          ),
        ),
      ],
    );
  }
}
