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

  List<Lesson> get _allLessons {
    final lessons = <Lesson>[];
    for (final module in widget.course.modules) {
      for (final chapter in module.chapters) lessons.addAll(chapter.lessons);
    }
    return lessons;
  }

  int _getLessonIndex(Lesson lesson) => _allLessons.indexWhere((l) => l.id == lesson.id);

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;
    final bgColor = isDark ? AppTheme.primaryDark : const Color(0xFFFEF6EE);

    return Scaffold(
      body: Container(
        color: bgColor,
        child: SafeArea(
          child: Column(
            children: [
              _buildAppBar(isDark),
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildCourseHeader(isDark),
                      const SizedBox(height: 24),
                      _buildModulesList(isDark),
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

  Widget _buildAppBar(bool isDark) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      child: Row(
        children: [
          IconButton(
            icon: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: isDark ? AppTheme.surfaceCard : Colors.white,
                shape: BoxShape.circle,
                boxShadow: isDark ? null : [BoxShadow(color: Colors.black.withValues(alpha: 0.06), blurRadius: 8, offset: const Offset(0, 2))],
              ),
              child: Icon(Icons.arrow_back_ios_new, size: 16, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary),
            ),
            onPressed: () => Navigator.pop(context),
          ),
          const Spacer(),
          IconButton(
            icon: Icon(Icons.more_vert, size: 24, color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary),
            onPressed: () {},
          ),
        ],
      ),
    );
  }

  Widget _buildCourseHeader(bool isDark) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: isDark
            ? const LinearGradient(colors: [Color(0xFF5B3E96), Color(0xFF3D2473)])
            : const LinearGradient(colors: [AppTheme.accentOrange, AppTheme.accentOrangeDark]),
        borderRadius: BorderRadius.circular(28),
        boxShadow: [
          BoxShadow(color: (isDark ? const Color(0xFF5B3E96) : AppTheme.accentOrange).withValues(alpha: 0.3), blurRadius: 20, offset: const Offset(0, 10)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(color: Colors.white.withValues(alpha: 0.2), borderRadius: BorderRadius.circular(16)),
                child: const Icon(Icons.school_rounded, color: Colors.white, size: 28),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(widget.course.title, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w700, color: Colors.white)),
                    const SizedBox(height: 6),
                    Row(children: [
                      _heroTag(widget.course.difficulty),
                      const SizedBox(width: 8),
                      _heroTag('${widget.course.modules.length} modules'),
                    ]),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text(widget.course.description, style: TextStyle(color: Colors.white.withValues(alpha: 0.85), fontSize: 13, height: 1.5), maxLines: 3, overflow: TextOverflow.ellipsis),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: LinearProgressIndicator(
                    value: widget.course.progress,
                    backgroundColor: Colors.white.withValues(alpha: 0.2),
                    valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                    minHeight: 6,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Text('${(widget.course.progress * 100).toInt()}%', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 14)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _heroTag(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(color: Colors.white.withValues(alpha: 0.2), borderRadius: BorderRadius.circular(10)),
      child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w600)),
    );
  }

  Widget _buildModulesList(bool isDark) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Course Content', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
          const SizedBox(height: 16),
          ...widget.course.modules.asMap().entries.map((entry) => _buildModuleItem(entry.key, entry.value, isDark)),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _buildModuleItem(int moduleIndex, Module module, bool isDark) {
    final isExpanded = _expandedModuleIndex == moduleIndex;
    final cardBg = isDark ? AppTheme.surfaceCard : Colors.white;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: cardBg,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: isExpanded ? AppTheme.accentOrange.withValues(alpha: 0.5) : (isDark ? AppTheme.primaryLight.withValues(alpha: 0.2) : Colors.grey.withValues(alpha: 0.1))),
        boxShadow: isDark ? null : [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 10, offset: const Offset(0, 3))],
      ),
      child: Column(
        children: [
          InkWell(
            onTap: () => setState(() { _expandedModuleIndex = isExpanded ? -1 : moduleIndex; _expandedChapterIndex = -1; }),
            borderRadius: BorderRadius.circular(20),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  Container(
                    width: 44,
                    height: 44,
                    decoration: BoxDecoration(
                      color: module.isCompleted ? AppTheme.success.withValues(alpha: 0.15) : AppTheme.accentOrange.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: Center(
                      child: module.isCompleted
                          ? const Icon(Icons.check_rounded, color: AppTheme.success, size: 22)
                          : Text('${moduleIndex + 1}', style: TextStyle(color: AppTheme.accentOrange, fontWeight: FontWeight.bold, fontSize: 16)),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(module.title, style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
                        const SizedBox(height: 2),
                        Text('${module.chapters.length} chapters • ${module.completedChapters}/${module.chapters.length} completed', style: TextStyle(color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted, fontSize: 12)),
                      ],
                    ),
                  ),
                  Icon(isExpanded ? Icons.expand_less : Icons.expand_more, color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary),
                ],
              ),
            ),
          ),
          if (isExpanded) ...[
            Divider(height: 1, color: isDark ? AppTheme.primaryLight.withValues(alpha: 0.15) : Colors.grey.withValues(alpha: 0.1)),
            ...module.chapters.asMap().entries.map((entry) => _buildChapterItem(moduleIndex, entry.key, entry.value, isDark)),
          ],
        ],
      ),
    );
  }

  Widget _buildChapterItem(int moduleIndex, int chapterIndex, Chapter chapter, bool isDark) {
    final isExpanded = _expandedModuleIndex == moduleIndex && _expandedChapterIndex == chapterIndex;

    return Column(
      children: [
        InkWell(
          onTap: () => setState(() { _expandedChapterIndex = isExpanded ? -1 : chapterIndex; }),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: Row(
              children: [
                const SizedBox(width: 52),
                Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: chapter.isCompleted ? AppTheme.success.withValues(alpha: 0.15) : (isDark ? AppTheme.primaryLight.withValues(alpha: 0.5) : Colors.grey.withValues(alpha: 0.08)),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Center(
                    child: chapter.isCompleted
                        ? const Icon(Icons.check, color: AppTheme.success, size: 16)
                        : Text('${chapterIndex + 1}', style: TextStyle(color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary, fontSize: 12, fontWeight: FontWeight.w500)),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(child: Text(chapter.title, style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary))),
                Icon(isExpanded ? Icons.expand_less : Icons.expand_more, color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted, size: 20),
              ],
            ),
          ),
        ),
        if (isExpanded) ...chapter.lessons.map((lesson) => _buildLessonItem(lesson, chapter, widget.course.modules[moduleIndex], isDark)),
      ],
    );
  }

  Widget _buildLessonItem(Lesson lesson, Chapter chapter, Module module, bool isDark) {
    final allLessons = _allLessons;
    final currentIndex = _getLessonIndex(lesson);

    return InkWell(
      onTap: () {
        Navigator.push(context, MaterialPageRoute(
          builder: (_) => LessonScreen(lesson: lesson, chapter: chapter, module: module, course: widget.course, allLessons: allLessons, currentIndex: currentIndex),
        )).then((_) => setState(() {}));
      },
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            const SizedBox(width: 96),
            Container(
              width: 26,
              height: 26,
              decoration: BoxDecoration(
                color: lesson.isCompleted ? AppTheme.success : Colors.transparent,
                borderRadius: BorderRadius.circular(13),
                border: Border.all(color: lesson.isCompleted ? AppTheme.success : (isDark ? AppTheme.textMuted : Colors.grey.withValues(alpha: 0.4)), width: 2),
              ),
              child: lesson.isCompleted ? const Icon(Icons.check, color: Colors.white, size: 14) : null,
            ),
            const SizedBox(width: 12),
            Expanded(child: Text(lesson.title, style: TextStyle(color: lesson.isCompleted ? (isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary) : (isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary), fontSize: 14))),
            const Icon(Icons.play_circle_outline, color: AppTheme.accentOrange, size: 20),
          ],
        ),
      ),
    );
  }
}
