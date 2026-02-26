import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../models/course.dart';
import '../services/theme_service.dart';
import 'lesson_screen.dart';

class CourseDetailScreen extends StatefulWidget {
  final Course course;

  const CourseDetailScreen({super.key, required this.course});

  @override
  State<CourseDetailScreen> createState() => _CourseDetailScreenState();
}

class _CourseDetailScreenState extends State<CourseDetailScreen> {
  int _expandedModuleIndex = 0;
  int _expandedChapterIndex = -1;

  // Get all lessons in the course for navigation
  List<Lesson> get _allLessons {
    final lessons = <Lesson>[];
    for (final module in widget.course.modules) {
      for (final chapter in module.chapters) {
        lessons.addAll(chapter.lessons);
      }
    }
    return lessons;
  }

  int _getLessonIndex(Lesson lesson) {
    return _allLessons.indexWhere((l) => l.id == lesson.id);
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildAppBar(),
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildCourseHeader(),
                      const SizedBox(height: 24),
                      _buildModulesList(),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAppBar() {
    final colorScheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_new, size: 20),
            onPressed: () => Navigator.pop(context),
            color: colorScheme.onSurface,
          ),
          const Spacer(),
          IconButton(
            icon: const Icon(Icons.more_vert, size: 24),
            onPressed: () {},
            color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
          ),
        ],
      ),
    );
  }

  Widget _buildCourseHeader() {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 64,
                height: 64,
                decoration: BoxDecoration(
                  gradient: AppTheme.primaryGradient,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Icon(
                  Icons.school,
                  color: Colors.white,
                  size: 32,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.course.title,
                      style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        _buildTag(widget.course.difficulty),
                        const SizedBox(width: 8),
                        _buildTag('${widget.course.modules.length} modules'),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text(
            widget.course.description,
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
              color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 20),
          _buildProgressCard(),
        ],
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

  Widget _buildProgressCard() {
    final progress = widget.course.progress;
    final colorScheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isDark 
              ? AppTheme.primaryLight.withValues(alpha: 0.3)
              : Colors.grey.withValues(alpha: 0.2),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Progress',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                  ),
                ),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: progress,
                    backgroundColor: isDark ? AppTheme.primaryLight : Colors.grey.withValues(alpha: 0.2),
                    valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
                    minHeight: 8,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 16),
          Column(
            children: [
              Text(
                '${(progress * 100).toInt()}%',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                '${widget.course.completedLessons}/${widget.course.totalLessons}',
                style: TextStyle(
                  color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildModulesList() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Course Content',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 16),
          ...widget.course.modules.asMap().entries.map((entry) {
            return _buildModuleItem(entry.key, entry.value);
          }),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _buildModuleItem(int moduleIndex, Module module) {
    final isExpanded = _expandedModuleIndex == moduleIndex;
    final colorScheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isExpanded 
              ? colorScheme.primary
              : (isDark ? AppTheme.primaryLight.withValues(alpha: 0.3) : Colors.grey.withValues(alpha: 0.2)),
        ),
      ),
      child: Column(
        children: [
          InkWell(
            onTap: () {
              setState(() {
                _expandedModuleIndex = isExpanded ? -1 : moduleIndex;
                _expandedChapterIndex = -1;
              });
            },
            borderRadius: BorderRadius.circular(12),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: module.isCompleted 
                          ? AppTheme.success.withValues(alpha: 0.2)
                          : (isDark ? AppTheme.primaryLight : AppTheme.tagBackground),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Center(
                      child: module.isCompleted
                          ? const Icon(Icons.check, color: AppTheme.success, size: 20)
                          : Text(
                              '${moduleIndex + 1}',
                              style: TextStyle(
                                color: isDark ? AppTheme.textPrimary : AppTheme.tagText,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          module.title,
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          '${module.chapters.length} chapters • ${module.completedChapters}/${module.chapters.length} completed',
                          style: TextStyle(
                            color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Icon(
                    isExpanded ? Icons.expand_less : Icons.expand_more,
                    color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                  ),
                ],
              ),
            ),
          ),
          if (isExpanded) ...[
            Divider(height: 1, color: isDark ? AppTheme.primaryLight : Colors.grey.withValues(alpha: 0.2)),
            ...module.chapters.asMap().entries.map((entry) {
              return _buildChapterItem(moduleIndex, entry.key, entry.value);
            }),
          ],
        ],
      ),
    );
  }

  Widget _buildChapterItem(int moduleIndex, int chapterIndex, Chapter chapter) {
    final isExpanded = _expandedModuleIndex == moduleIndex && _expandedChapterIndex == chapterIndex;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    
    return Column(
      children: [
        InkWell(
          onTap: () {
            setState(() {
              _expandedChapterIndex = isExpanded ? -1 : chapterIndex;
            });
          },
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: Row(
              children: [
                const SizedBox(width: 52),
                Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: chapter.isCompleted 
                        ? AppTheme.success.withValues(alpha: 0.2)
                        : (isDark ? AppTheme.primaryLight : Colors.grey.withValues(alpha: 0.1)),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Center(
                    child: chapter.isCompleted
                        ? const Icon(Icons.check, color: AppTheme.success, size: 16)
                        : Text(
                            '${chapterIndex + 1}',
                            style: TextStyle(
                              color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    chapter.title,
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
                Icon(
                  isExpanded ? Icons.expand_less : Icons.expand_more,
                  color: isDark ? AppTheme.textMuted : AppTheme.lightTextSecondary.withValues(alpha: 0.6),
                  size: 20,
                ),
              ],
            ),
          ),
        ),
        if (isExpanded) ...[
          ...chapter.lessons.map((lesson) {
            return _buildLessonItem(lesson, chapter, widget.course.modules[moduleIndex]);
          }),
        ],
      ],
    );
  }

  Widget _buildLessonItem(Lesson lesson, Chapter chapter, Module module) {
    final allLessons = _allLessons;
    final currentIndex = _getLessonIndex(lesson);
    final isDark = Theme.of(context).brightness == Brightness.dark;
    
    return InkWell(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => LessonScreen(
              lesson: lesson,
              chapter: chapter,
              module: module,
              course: widget.course,
              allLessons: allLessons,
              currentIndex: currentIndex,
            ),
          ),
        ).then((_) {
          setState(() {});
        });
      },
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            const SizedBox(width: 96),
            Container(
              width: 24,
              height: 24,
              decoration: BoxDecoration(
                color: lesson.isCompleted 
                    ? AppTheme.success
                    : Colors.transparent,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: lesson.isCompleted 
                      ? AppTheme.success
                      : (isDark ? AppTheme.textMuted : Colors.grey.withValues(alpha: 0.4)),
                  width: 2,
                ),
              ),
              child: lesson.isCompleted
                  ? const Icon(Icons.check, color: Colors.white, size: 14)
                  : null,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                lesson.title,
                style: TextStyle(
                  color: lesson.isCompleted 
                      ? (isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary)
                      : Theme.of(context).textTheme.bodyLarge?.color,
                  fontSize: 14,
                ),
              ),
            ),
            const Icon(
              Icons.play_circle_outline,
              color: AppTheme.accentOrange,
              size: 20,
            ),
          ],
        ),
      ),
    );
  }
}
