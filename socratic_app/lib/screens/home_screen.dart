import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../models/course.dart';
import '../services/course_service.dart';
import '../services/theme_service.dart';
import 'course_detail_screen.dart';
import 'chat_screen.dart';
import 'settings_screen.dart';
import 'profile_screen.dart';
import 'quiz_screen.dart';
import 'glossary_screen.dart';
import 'playground_screen.dart';
import '../services/hybrid_tutor_service.dart';
import '../services/auth_service.dart';
import '../services/model_download_service.dart';
import '../utils/app_snackbar.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<Course> _courses = [];
  int _currentIndex = 0;
  bool _isLoading = true;
  bool _playgroundActive = false;
  bool _isSyncing = false;
  int _newCoursesFound = 0;
  String? _syncError;
  final CourseService _courseService = CourseService();
  final HybridTutorService _hybridService = HybridTutorService();

  // Starter prompts shown in the AI Tutor tab
  static const _starterPrompts = [
    'What is machine learning?',
    'Explain gradient descent',
    'What is overfitting?',
    'How does backpropagation work?',
    'What is a neural network?',
    'Explain supervised learning',
  ];

  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 500), () {
      if (mounted) _loadCourses();
    });
  }

  Future<void> _loadCourses() async {
    if (!mounted) return;
    setState(() => _isLoading = true);

    try {
      // Load only local/bundled courses — fast, no network wait.
      final courses = await _courseService.getCourses();
      if (mounted) {
        setState(() {
          _courses = courses;
          _isLoading = false;
        });
      }
    } catch (e) {
      debugPrint('HomeScreen: Error loading courses: $e');
      if (mounted) setState(() { _courses = []; _isLoading = false; });
    }
  }

  Future<void> _syncRemoteCourses() async {
    if (_isSyncing || !mounted) return;
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
      debugPrint('HomeScreen: Remote sync error: $e');
      if (mounted) {
        setState(() {
          _syncError = 'Could not reach server. Check your connection.';
          _isSyncing = false;
        });
      }
    }
  }

  String _getGreeting() {
    final hour = DateTime.now().hour;
    if (hour < 12) return 'Good Morning';
    if (hour < 17) return 'Good Afternoon';
    return 'Good Evening';
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;

    return Scaffold(
      body: Stack(
        children: [
          // Main tab shell — always alive
          Container(
            decoration: BoxDecoration(
              gradient: isDark
                  ? AppTheme.backgroundGradient
                  : AppTheme.lightBackgroundGradient,
            ),
            child: SafeArea(
              // IndexedStack keeps all tab subtrees alive so switching tabs does
              // not reload courses or reset UI state.
              child: IndexedStack(
                index: _currentIndex,
                children: [
                  _buildCoursesTab(),
                  _buildChatTab(),
                  const ProfileScreen(),
                  const SettingsScreen(),
                ],
              ),
            ),
          ),

          // Playground overlay — kept alive via Offstage so Pyodide never restarts
          Offstage(
            offstage: !_playgroundActive,
            child: PlaygroundScreen(
              onClose: () => setState(() => _playgroundActive = false),
            ),
          ),
        ],
      ),
      bottomNavigationBar: _playgroundActive ? null : _buildBottomNav(isDark),
    );
  }

  // ── Material 3 NavigationBar ───────────────────────────────────────────────
  Widget _buildBottomNav(bool isDark) {
    return Container(
      decoration: BoxDecoration(
        border: Border(
          top: BorderSide(
            color: isDark
                ? AppTheme.primaryLight.withValues(alpha: 0.15)
                : AppTheme.accentOrange.withValues(alpha: 0.1),
            width: 0.5,
          ),
        ),
      ),
      child: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) => setState(() => _currentIndex = i),
        backgroundColor: isDark ? AppTheme.surfaceDark : AppTheme.lightSurface,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        indicatorColor: AppTheme.accentOrange.withValues(alpha: 0.15),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.school_outlined),
            selectedIcon: Icon(Icons.school),
            label: 'Courses',
          ),
          NavigationDestination(
            icon: Icon(Icons.chat_bubble_outline),
            selectedIcon: Icon(Icons.chat_bubble),
            label: 'AI Tutor',
          ),
          NavigationDestination(
            icon: Icon(Icons.person_outline),
            selectedIcon: Icon(Icons.person),
            label: 'Profile',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }

  // ── Courses Tab ───────────────────────────────────────────────────────────
  Widget _buildCoursesTab() {
    if (_isLoading) {
      return const Center(
        child: CircularProgressIndicator(color: AppTheme.accentOrange),
      );
    }

    if (_courses.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(40),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: AppTheme.accentOrange.withValues(alpha: 0.1),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.wifi_off_rounded,
                    size: 48, color: AppTheme.accentOrange),
              ),
              const SizedBox(height: 20),
              Text(
                'No courses found',
                style: Theme.of(context)
                    .textTheme
                    .titleLarge
                    ?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Text(
                'This can happen on first launch or in offline mode.',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: () {
                  _courseService.clearCache();
                  _loadCourses();
                },
                icon: const Icon(Icons.refresh, size: 18),
                label: const Text('Refresh'),
              ),
            ],
          ),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Non-blocking download progress banner
        _buildModelDownloadBanner(),

        Padding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header row
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _getGreeting(),
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                        const SizedBox(height: 2),
                        Consumer<AuthService>(
                          builder: (context, auth, _) => Text(
                            auth.currentUser?.username ?? 'Learner',
                            style: Theme.of(context)
                                .textTheme
                                .headlineMedium
                                ?.copyWith(fontWeight: FontWeight.w700),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 12),
                  _buildStatusBadge(),
                ],
              ),

              const SizedBox(height: 20),
              _buildQuickTools(),
              const SizedBox(height: 24),

              Text(
                'Continue Learning',
                style: Theme.of(context)
                    .textTheme
                    .titleLarge
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 12),
            ],
          ),
        ),

        Expanded(
          child: RefreshIndicator(
            onRefresh: () async {
              _courseService.clearCache();
              await _loadCourses();
            },
            color: AppTheme.accentOrange,
            child: ListView.builder(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 8),
              itemCount: _courses.length + 1, // +1 for sync button footer
              itemBuilder: (context, index) {
                if (index == _courses.length) {
                  return _buildSyncFooter();
                }
                return _buildCourseCard(_courses[index]);
              },
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSyncFooter() {
    final isDark = context.watch<ThemeService>().isDarkMode;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: Column(
        children: [
          if (_syncError != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  color: AppTheme.warning.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(12),
                  border:
                      Border.all(color: AppTheme.warning.withValues(alpha: 0.35)),
                ),
                child: Row(
                  children: [
                    Icon(Icons.warning_amber_rounded,
                        size: 16, color: AppTheme.warning),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _syncError!,
                        style: TextStyle(
                          fontSize: 12,
                          color: AppTheme.warning,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          if (_newCoursesFound > 0)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                '$_newCoursesFound new course${_newCoursesFound > 1 ? 's' : ''} added!',
                textAlign: TextAlign.center,
                style: TextStyle(
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
                  ? SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: isDark
                            ? AppTheme.accentOrange
                            : AppTheme.accentOrange,
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

  Widget _buildStatusBadge() {
    return StreamBuilder<EngineStatus>(
      stream: _hybridService.statusStream,
      initialData: _hybridService.currentStatus,
      builder: (context, snapshot) {
        final status = snapshot.data ?? EngineStatus.offline;
        final isOnline = status == EngineStatus.online;
        final color = isOnline ? AppTheme.success : AppTheme.warning;
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: color.withValues(alpha: 0.3)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 6,
                height: 6,
                decoration:
                    BoxDecoration(shape: BoxShape.circle, color: color),
              ),
              const SizedBox(width: 6),
              Text(
                isOnline ? 'Online' : 'Offline',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: color,
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildQuickTools() {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: _buildToolCard(
                'Practice Quiz',
                Icons.quiz_outlined,
                AppTheme.accentOrange,
                () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const QuizScreen()),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildToolCard(
                'Key Terms',
                Icons.book_outlined,
                const Color(0xFF6366F1),
                () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const GlossaryScreen()),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        _buildToolCard(
          'Python Playground',
          Icons.code_rounded,
          const Color(0xFF10B981), // emerald green
          () => setState(() => _playgroundActive = true),
        ),
      ],
    );
  }

  Widget _buildToolCard(
      String title, IconData icon, Color color, VoidCallback onTap) {
    final isDark = context.watch<ThemeService>().isDarkMode;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        decoration: BoxDecoration(
          color: isDark
              ? color.withValues(alpha: 0.12)
              : color.withValues(alpha: 0.07),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
            color: color.withValues(alpha: 0.25),
            width: 1.2,
          ),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                title,
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                  color: isDark
                      ? AppTheme.textPrimary
                      : AppTheme.lightTextPrimary,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _handleCourseTap(Course course) async {
    if (course.modules.isEmpty) {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (context) => const Center(
          child: CircularProgressIndicator(color: AppTheme.accentOrange),
        ),
      );

      final fullCourse = await _courseService.loadCourse(course.id);

      if (mounted) {
        Navigator.pop(context);
        if (fullCourse != null) {
          _navigateToDetail(fullCourse);
        } else {
          AppSnackBar.error(
            context,
            'Failed to load course content. Check your connection.',
          );
        }
      }
    } else {
      _navigateToDetail(course);
    }
  }

  void _navigateToDetail(Course course) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => CourseDetailScreen(course: course),
      ),
    ).then((_) => setState(() {}));
  }

  Widget _buildCourseCard(Course course) {
    final isDark = context.watch<ThemeService>().isDarkMode;

    return GestureDetector(
      onTap: () => _handleCourseTap(course),
      child: Container(
        margin: const EdgeInsets.only(bottom: 14),
        decoration: BoxDecoration(
          color: isDark ? AppTheme.surfaceCard : Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
            color: isDark
                ? AppTheme.primaryLight.withValues(alpha: 0.25)
                : AppTheme.tagBackground,
          ),
          boxShadow: isDark
              ? null
              : [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.05),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ],
        ),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 44,
                    height: 44,
                    decoration: BoxDecoration(
                      gradient: AppTheme.primaryGradient,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child:
                        const Icon(Icons.school, color: Colors.white, size: 22),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          course.title,
                          style: Theme.of(context)
                              .textTheme
                              .titleSmall
                              ?.copyWith(fontWeight: FontWeight.w600),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            _buildTag(course.difficulty),
                            const SizedBox(width: 6),
                            _buildTag(course.duration),
                          ],
                        ),
                      ],
                    ),
                  ),
                  Icon(
                    Icons.chevron_right_rounded,
                    color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted,
                    size: 20,
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Text(
                course.description,
                style: Theme.of(context).textTheme.bodySmall,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 12),
              _buildProgressBar(course),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTag(String text) {
    final isDark = context.watch<ThemeService>().isDarkMode;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: isDark ? AppTheme.primaryLight : AppTheme.tagBackground,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: isDark ? AppTheme.textSecondary : AppTheme.tagText,
          fontSize: 10,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }

  Widget _buildProgressBar(Course course) {
    final isDark = context.watch<ThemeService>().isDarkMode;
    final progress = course.progress;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              '${course.completedLessons}/${course.totalLessons} lessons',
              style: Theme.of(context).textTheme.labelSmall,
            ),
            Text(
              '${(progress * 100).toInt()}%',
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: AppTheme.accentOrange,
                  ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: progress,
            backgroundColor: isDark
                ? AppTheme.primaryLight.withValues(alpha: 0.3)
                : AppTheme.tagBackground,
            valueColor:
                const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
            minHeight: 4,
          ),
        ),
      ],
    );
  }

  // ── Model download banner (shown in Courses tab while model is pending) ───
  Widget _buildModelDownloadBanner() {
    return Consumer<ModelDownloadService>(
      builder: (context, dl, _) {
        final isDark = context.watch<ThemeService>().isDarkMode;

        // Hide when already complete
        if (dl.status == DownloadStatus.completed) return const SizedBox.shrink();

        final isDownloading = dl.status == DownloadStatus.downloading ||
            dl.status == DownloadStatus.connecting;
        final pct = dl.progress; // 0.0–1.0

        return GestureDetector(
          onTap: () => setState(() => _currentIndex = 3), // go to Settings tab
          child: Container(
            margin: const EdgeInsets.fromLTRB(20, 12, 20, 0),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: isDark
                  ? AppTheme.accentOrange.withValues(alpha: 0.12)
                  : AppTheme.accentOrange.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                color: AppTheme.accentOrange.withValues(alpha: 0.35),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      isDownloading
                          ? Icons.downloading_rounded
                          : dl.status == DownloadStatus.error
                              ? Icons.error_outline_rounded
                              : Icons.download_for_offline_outlined,
                      size: 16,
                      color: AppTheme.accentOrange,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        isDownloading
                            ? 'Downloading offline AI model… ${(pct * 100).toInt()}%'
                            : dl.status == DownloadStatus.error
                                ? 'Model download failed — tap to retry'
                                : 'Download AI model for offline use →',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: isDark
                              ? AppTheme.textPrimary
                              : AppTheme.lightTextPrimary,
                        ),
                      ),
                    ),
                    Icon(
                      Icons.chevron_right_rounded,
                      size: 16,
                      color: AppTheme.accentOrange,
                    ),
                  ],
                ),
                if (isDownloading) ...[
                  const SizedBox(height: 6),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(2),
                    child: LinearProgressIndicator(
                      value: pct > 0 ? pct : null,
                      backgroundColor:
                          AppTheme.accentOrange.withValues(alpha: 0.15),
                      valueColor: const AlwaysStoppedAnimation<Color>(
                          AppTheme.accentOrange),
                      minHeight: 3,
                    ),
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }

  // ── AI Tutor Tab ──────────────────────────────────────────────────────────
  Widget _buildChatTab() {
    final isDark = context.watch<ThemeService>().isDarkMode;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'AI Tutor',
                style: Theme.of(context)
                    .textTheme
                    .headlineMedium
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 4),
              Text(
                'Ask anything, discover answers through questions',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ),
        ),

        // Scrollable hero + prompts
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                const SizedBox(height: 12),

                // Glowing icon
                Container(
                  width: 88,
                  height: 88,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: AppTheme.primaryGradient,
                    boxShadow: [
                      BoxShadow(
                        color: AppTheme.accentOrange.withValues(alpha: 0.35),
                        blurRadius: 32,
                        spreadRadius: 4,
                      ),
                    ],
                  ),
                  child: const Icon(Icons.psychology,
                      size: 44, color: Colors.white),
                ),

                const SizedBox(height: 20),

                Text(
                  'Socratic AI Tutor',
                  style: Theme.of(context)
                      .textTheme
                      .headlineSmall
                      ?.copyWith(fontWeight: FontWeight.w700),
                ),

                const SizedBox(height: 8),

                Text(
                  'I guide you to discover answers\nthrough thoughtful questions',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),

                const SizedBox(height: 28),

                // CTA button
                SizedBox(
                  width: double.infinity,
                  child: Container(
                    decoration: BoxDecoration(
                      gradient: AppTheme.buttonGradient,
                      borderRadius: BorderRadius.circular(14),
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme.accentOrange.withValues(alpha: 0.35),
                          blurRadius: 16,
                          offset: const Offset(0, 6),
                        ),
                      ],
                    ),
                    child: ElevatedButton.icon(
                      onPressed: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const ChatScreen()),
                      ),
                      icon: const Icon(Icons.chat_bubble_outline, size: 18),
                      label: const Text(
                        'Start Conversation',
                        style: TextStyle(
                            fontSize: 15, fontWeight: FontWeight.w600),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.transparent,
                        shadowColor: Colors.transparent,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 32),

                // Starter prompts
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'Try asking...',
                    style: Theme.of(context).textTheme.labelMedium,
                  ),
                ),
                const SizedBox(height: 10),
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: _starterPrompts
                        .map((p) => _buildPromptChip(p, isDark))
                        .toList(),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildPromptChip(String prompt, bool isDark) {
    return GestureDetector(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ChatScreen(initialMessage: prompt),
        ),
      ),
      child: Container(
        margin: const EdgeInsets.only(right: 8),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
        decoration: BoxDecoration(
          color: isDark ? AppTheme.surfaceCard : Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isDark
                ? AppTheme.primaryLight.withValues(alpha: 0.3)
                : AppTheme.tagBackground,
          ),
          boxShadow: isDark
              ? null
              : [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.04),
                    blurRadius: 8,
                    offset: const Offset(0, 2),
                  ),
                ],
        ),
        child: Text(
          prompt,
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w500,
            color: isDark
                ? AppTheme.textSecondary
                : AppTheme.lightTextSecondary,
          ),
        ),
      ),
    );
  }
}
