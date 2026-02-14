import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../models/course.dart';
import '../services/progress_service.dart';
import '../services/course_service.dart';
import '../services/theme_service.dart';
import 'lesson_chat_screen.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

class LessonScreen extends StatefulWidget {
  final Lesson lesson;
  final Chapter chapter;
  final Module module;
  final Course course;
  final List<Lesson>? allLessons; // All lessons in chapter for navigation
  final int? currentIndex; // Current lesson index

  const LessonScreen({
    super.key,
    required this.lesson,
    required this.chapter,
    required this.module,
    required this.course,
    this.allLessons,
    this.currentIndex,
  });

  @override
  State<LessonScreen> createState() => _LessonScreenState();
}

class _LessonScreenState extends State<LessonScreen> {
  final ScrollController _scrollController = ScrollController();
  final CourseService _courseService = CourseService();
  bool _hasReachedEnd = false;
  bool _isLoadingContent = true;
  String _content = '';

  bool get _hasNextLesson {
    if (widget.allLessons == null || widget.currentIndex == null) return false;
    return widget.currentIndex! < widget.allLessons!.length - 1;
  }

  bool get _hasPreviousLesson {
    if (widget.allLessons == null || widget.currentIndex == null) return false;
    return widget.currentIndex! > 0;
  }

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
    _loadProgress();
    _loadContent();
  }

  Future<void> _loadContent() async {
    setState(() => _isLoadingContent = true);
    final content = await _courseService.loadLessonContent(
      widget.lesson, 
      courseId: widget.course.id,
    );
    if (mounted) {
      setState(() {
        _content = content;
        _isLoadingContent = false;
      });
    }
  }

  void _loadProgress() async {
    final isCompleted = await ProgressService.isLessonCompleted(widget.lesson.id);
    if (isCompleted && mounted) {
      setState(() {
        widget.lesson.isCompleted = true;
      });
    }
  }

  void _onScroll() {
    if (_scrollController.position.pixels >= 
        _scrollController.position.maxScrollExtent - 100) {
      if (!_hasReachedEnd) {
        setState(() {
          _hasReachedEnd = true;
        });
      }
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _markComplete() async {
    await ProgressService.markLessonComplete(widget.lesson.id);
    setState(() {
      widget.lesson.isCompleted = true;
    });
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Lesson completed! 🎉'),
          backgroundColor: AppTheme.success,
          duration: Duration(seconds: 2),
        ),
      );
    }
  }

  void _openAITutor() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => LessonChatScreen(
          lesson: widget.lesson,
          course: widget.course,
        ),
      ),
    );
  }

  void _goToNextLesson() {
    if (!_hasNextLesson) return;
    
    final nextIndex = widget.currentIndex! + 1;
    final nextLesson = widget.allLessons![nextIndex];
    
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (context) => LessonScreen(
          lesson: nextLesson,
          chapter: widget.chapter,
          module: widget.module,
          course: widget.course,
          allLessons: widget.allLessons,
          currentIndex: nextIndex,
        ),
      ),
    );
  }

  void _goToPreviousLesson() {
    if (!_hasPreviousLesson) return;
    
    final prevIndex = widget.currentIndex! - 1;
    final prevLesson = widget.allLessons![prevIndex];
    
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (context) => LessonScreen(
          lesson: prevLesson,
          chapter: widget.chapter,
          module: widget.module,
          course: widget.course,
          allLessons: widget.allLessons,
          currentIndex: prevIndex,
        ),
      ),
    );
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
                  controller: _scrollController,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildLessonHeader(),
                      _buildContent(),
                      if (widget.lesson.keyPoints != null) _buildKeyPoints(),
                      if (widget.lesson.reflectionQuestions != null) 
                        _buildReflectionQuestions(),
                      _buildBottomActions(),
                      const SizedBox(height: 32),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _openAITutor,
        backgroundColor: AppTheme.accentOrange,
        icon: const Icon(Icons.psychology),
        label: const Text('Ask AI Tutor'),
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
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.chapter.title,
                  style: TextStyle(
                    color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                    fontSize: 12,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
                Text(
                  widget.lesson.title,
                  style: TextStyle(
                    color: colorScheme.onSurface,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          if (widget.lesson.isCompleted)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: AppTheme.success.withOpacity(0.2),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.check_circle, color: AppTheme.success, size: 16),
                  SizedBox(width: 4),
                  Text(
                    'Done',
                    style: TextStyle(
                      color: AppTheme.success,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildLessonHeader() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: isDark ? AppTheme.primaryLight : AppTheme.tagBackground,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  widget.module.title,
                  style: TextStyle(
                    color: isDark ? AppTheme.textPrimary : AppTheme.tagText,
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildContent() {
    if (_isLoadingContent) {
      return const Padding(
        padding: EdgeInsets.all(40),
        child: Center(
          child: CircularProgressIndicator(color: AppTheme.accentOrange),
        ),
      );
    }

    final colorScheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: MarkdownBody(
        data: _content,
        styleSheet: MarkdownStyleSheet(
          h1: TextStyle(
            color: colorScheme.onSurface,
            fontSize: 28,
            fontWeight: FontWeight.bold,
          ),
          h2: TextStyle(
            color: colorScheme.onSurface,
            fontSize: 22,
            fontWeight: FontWeight.w600,
          ),
          h3: TextStyle(
            color: colorScheme.onSurface,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
          p: TextStyle(
            color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
            fontSize: 16,
            height: 1.7,
          ),
          strong: TextStyle(
            color: colorScheme.onSurface,
            fontWeight: FontWeight.w600,
          ),
          em: TextStyle(
            color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
            fontStyle: FontStyle.italic,
          ),
          code: TextStyle(
            color: colorScheme.onSurface,
            backgroundColor: isDark ? AppTheme.surfaceCard : AppTheme.tagBackground,
            fontSize: 14,
            fontFamily: 'monospace',
          ),
          codeblockDecoration: BoxDecoration(
            color: isDark ? AppTheme.surfaceCard : Colors.grey.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: isDark ? AppTheme.primaryLight.withOpacity(0.3) : Colors.grey.withOpacity(0.2),
            ),
          ),
          codeblockPadding: const EdgeInsets.all(16),
          blockquote: TextStyle(
            color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
            fontStyle: FontStyle.italic,
          ),
          blockquoteDecoration: BoxDecoration(
            border: Border(
              left: BorderSide(
                color: isDark ? AppTheme.primaryLight : AppTheme.lightTextMuted,
                width: 4,
              ),
            ),
          ),
          blockquotePadding: const EdgeInsets.only(left: 16),
          listBullet: TextStyle(
            color: colorScheme.onSurface,
          ),
          tableHead: TextStyle(
            color: colorScheme.onSurface,
            fontWeight: FontWeight.w600,
          ),
          tableBody: TextStyle(
            color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
          ),
          tableBorder: TableBorder.all(
            color: isDark ? AppTheme.primaryLight.withOpacity(0.3) : Colors.grey.withOpacity(0.2),
          ),
          tableHeadAlign: TextAlign.center,
          tableCellsPadding: const EdgeInsets.all(8),
        ),
      ),
    );
  }

  Widget _buildKeyPoints() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      margin: const EdgeInsets.all(20),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark ? AppTheme.surfaceCard : AppTheme.lightSurfaceCard,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isDark ? AppTheme.primaryLight.withOpacity(0.5) : AppTheme.tagBackground,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.lightbulb, color: colorScheme.primary, size: 20),
              const SizedBox(width: 8),
              Text(
                'Key Points',
                style: TextStyle(
                  color: colorScheme.onSurface,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            widget.lesson.keyPoints!,
            style: TextStyle(
              color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
              fontSize: 14,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildReflectionQuestions() {
    final colorScheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isDark ? AppTheme.primaryLight.withOpacity(0.3) : Colors.grey.withOpacity(0.2),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.psychology, color: AppTheme.warning, size: 20),
              SizedBox(width: 8),
              Text(
                'Reflection Questions',
                style: TextStyle(
                  color: AppTheme.warning,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Think about these questions. Ask the AI Tutor for guidance!',
            style: TextStyle(
              color: isDark ? AppTheme.textMuted : AppTheme.lightTextSecondary.withOpacity(0.7),
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 12),
          ...widget.lesson.reflectionQuestions!.asMap().entries.map((entry) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${entry.key + 1}. ',
                    style: const TextStyle(
                      color: AppTheme.warning,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Expanded(
                    child: Text(
                      entry.value,
                      style: TextStyle(
                        color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                        fontSize: 14,
                        height: 1.4,
                      ),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildBottomActions() {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          // Mark Complete Button (if not completed)
          if (!widget.lesson.isCompleted)
            SizedBox(
              width: double.infinity,
              child: Container(
                decoration: BoxDecoration(
                  gradient: _hasReachedEnd 
                      ? AppTheme.buttonGradient
                      : null,
                  color: _hasReachedEnd ? null : (isDark ? AppTheme.primaryLight : Colors.grey.withOpacity(0.2)),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ElevatedButton(
                  onPressed: _hasReachedEnd ? _markComplete : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.transparent,
                    shadowColor: Colors.transparent,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: Text(
                    _hasReachedEnd ? 'Mark as Complete' : 'Read to the end to complete',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: _hasReachedEnd ? Colors.white : (isDark ? AppTheme.textMuted : AppTheme.lightTextSecondary.withOpacity(0.5)),
                    ),
                  ),
                ),
              ),
            ),
          
          const SizedBox(height: 16),
          
          // Navigation Row (Previous / Next)
          if (widget.allLessons != null && widget.currentIndex != null)
            Row(
              children: [
                // Previous Button
                if (_hasPreviousLesson)
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _goToPreviousLesson,
                      icon: const Icon(Icons.arrow_back, size: 18),
                      label: const Text('Previous'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                        side: BorderSide(
                          color: isDark ? AppTheme.primaryLight.withOpacity(0.5) : Colors.grey.withOpacity(0.3),
                        ),
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  )
                else
                  const Spacer(),
                
                const SizedBox(width: 12),
                
                // Next Button
                if (_hasNextLesson)
                  Expanded(
                    child: Container(
                      decoration: BoxDecoration(
                        gradient: AppTheme.buttonGradient,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: ElevatedButton.icon(
                        onPressed: _goToNextLesson,
                        icon: const Text('Next Lesson'),
                        label: const Icon(Icons.arrow_forward, size: 18),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.transparent,
                          shadowColor: Colors.transparent,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ),
                  )
                else
                  Expanded(
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      decoration: BoxDecoration(
                        color: AppTheme.success.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: AppTheme.success.withOpacity(0.5),
                        ),
                      ),
                      child: const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.check_circle, color: AppTheme.success, size: 18),
                          SizedBox(width: 8),
                          Text(
                            'Chapter Complete!',
                            style: TextStyle(
                              color: AppTheme.success,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          
          // Lesson progress indicator
          if (widget.allLessons != null && widget.currentIndex != null)
            Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Text(
                'Lesson ${widget.currentIndex! + 1} of ${widget.allLessons!.length}',
                style: TextStyle(
                  color: isDark ? AppTheme.textMuted : AppTheme.lightTextSecondary.withOpacity(0.5),
                  fontSize: 12,
                ),
              ),
            ),
        ],
      ),
    );
  }
}
