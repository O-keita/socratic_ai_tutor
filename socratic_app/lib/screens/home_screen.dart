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
import '../services/model_version_service.dart';
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

  static const _starterPrompts = [
    'What is machine learning?',
    'Explain gradient descent',
    'What is overfitting?',
    'How does backpropagation work?',
    'What is a neural network?',
    'Explain supervised learning',
  ];

  static const _topicCategories = [
    'All',
    'Machine Learning',
    'Data Science',
    'Deep Learning',
    'Statistics',
  ];

  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 500), () {
      if (mounted) _loadCourses();
    });
    // Check for model updates after the UI settles
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) ModelVersionService().checkForUpdate();
    });
  }

  Future<void> _loadCourses() async {
    if (!mounted) return;
    setState(() => _isLoading = true);
    try {
      final courses = await _courseService.getCourses();
      if (mounted) setState(() { _courses = courses; _isLoading = false; });
    } catch (e) {
      debugPrint('HomeScreen: Error loading courses: $e');
      if (mounted) setState(() { _courses = []; _isLoading = false; });
    }
  }

  Future<void> _syncRemoteCourses() async {
    if (_isSyncing || !mounted) return;
    setState(() { _isSyncing = true; _syncError = null; _newCoursesFound = 0; });
    try {
      final added = await _courseService.fetchRemoteCourses(_courses);
      if (mounted) setState(() { _courses = [..._courses, ...added]; _newCoursesFound = added.length; _isSyncing = false; });
    } catch (e) {
      if (mounted) setState(() { _syncError = 'Could not reach server.'; _isSyncing = false; });
    }
  }

  String _getGreeting() {
    final hour = DateTime.now().hour;
    if (hour < 12) return 'Good Morning';
    if (hour < 17) return 'Good Afternoon';
    return 'Good Evening';
  }

  double get _overallProgress {
    if (_courses.isEmpty) return 0;
    int total = 0, completed = 0;
    for (final c in _courses) { total += c.totalLessons; completed += c.completedLessons; }
    return total > 0 ? completed / total : 0;
  }

  int get _totalCompletedLessons {
    int c = 0;
    for (final course in _courses) c += course.completedLessons;
    return c;
  }

  int get _totalLessons {
    int t = 0;
    for (final course in _courses) t += course.totalLessons;
    return t;
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;

    return Scaffold(
      body: Stack(
        children: [
          IndexedStack(
            index: _currentIndex,
            children: [
              _buildCoursesTab(isDark),
              _buildChatTab(isDark),
              const ProfileScreen(),
              const SettingsScreen(),
            ],
          ),
          Offstage(
            offstage: !_playgroundActive,
            child: PlaygroundScreen(
              onClose: () => setState(() => _playgroundActive = false),
            ),
          ),
        ],
      ),
      bottomNavigationBar: _playgroundActive ? null : _buildBottomNav(isDark),
      floatingActionButton: _playgroundActive ? null : _buildCenterFAB(),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
    );
  }

  // ── Center FAB ────────────────────────────────────────────────────────────
  Widget _buildCenterFAB() {
    return Container(
      height: 58,
      width: 58,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: AppTheme.buttonGradient,
        boxShadow: [
          BoxShadow(
            color: AppTheme.accentOrange.withValues(alpha: 0.45),
            blurRadius: 14,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: FloatingActionButton(
        onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ChatScreen())),
        backgroundColor: Colors.transparent,
        elevation: 0,
        shape: const CircleBorder(),
        child: const Icon(Icons.chat_bubble_rounded, color: Colors.white, size: 24),
      ),
    );
  }

  // ── Bottom Navigation ─────────────────────────────────────────────────────
  Widget _buildBottomNav(bool isDark) {
    final bg = isDark ? AppTheme.surfaceDark : Colors.white;
    final active = AppTheme.accentOrange;
    final inactive = isDark ? AppTheme.textMuted : AppTheme.lightTextMuted;

    return Container(
      decoration: BoxDecoration(
        color: bg,
        borderRadius: const BorderRadius.only(topLeft: Radius.circular(28), topRight: Radius.circular(28)),
        boxShadow: [
          BoxShadow(color: Colors.black.withValues(alpha: isDark ? 0.25 : 0.08), blurRadius: 24, offset: const Offset(0, -6)),
        ],
      ),
      child: ClipRRect(
        borderRadius: const BorderRadius.only(topLeft: Radius.circular(28), topRight: Radius.circular(28)),
        child: BottomAppBar(
          color: bg,
          elevation: 0,
          notchMargin: 8,
          shape: const CircularNotchedRectangle(),
          height: 68,
          padding: const EdgeInsets.only(top: 6),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _navItem(Icons.home_outlined, Icons.home_rounded, 'Home', 0, active, inactive, isDark),
              _navItem(Icons.explore_outlined, Icons.explore, 'Explore', 1, active, inactive, isDark),
              const SizedBox(width: 56), // Space for FAB
              _navItem(Icons.person_outline, Icons.person, 'Profile', 2, active, inactive, isDark),
              Consumer<ModelVersionService>(
                builder: (context, versionService, child) {
                  return Stack(
                    clipBehavior: Clip.none,
                    children: [
                      _navItem(Icons.settings_outlined, Icons.settings, 'Settings', 3, active, inactive, isDark),
                      if (versionService.updateAvailable)
                        Positioned(
                          right: 8,
                          top: 2,
                          child: Container(
                            width: 8,
                            height: 8,
                            decoration: const BoxDecoration(
                              color: AppTheme.accentOrange,
                              shape: BoxShape.circle,
                            ),
                          ),
                        ),
                    ],
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _navItem(IconData icon, IconData activeIcon, String label, int idx, Color active, Color inactive, bool isDark) {
    final sel = _currentIndex == idx;
    return InkWell(
      onTap: () => setState(() => _currentIndex = idx),
      borderRadius: BorderRadius.circular(16),
      splashColor: active.withValues(alpha: 0.1),
      highlightColor: active.withValues(alpha: 0.05),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: EdgeInsets.symmetric(horizontal: sel ? 14 : 10, vertical: 6),
        decoration: sel
            ? BoxDecoration(
                color: active.withValues(alpha: isDark ? 0.15 : 0.1),
                borderRadius: BorderRadius.circular(16),
              )
            : null,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(sel ? activeIcon : icon, color: sel ? active : inactive, size: 22),
            const SizedBox(height: 3),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: sel ? FontWeight.w700 : FontWeight.w400,
                color: sel ? active : inactive,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ── HOME / COURSES TAB ────────────────────────────────────────────────────
  // ═══════════════════════════════════════════════════════════════════════════
  Widget _buildCoursesTab(bool isDark) {
    // Full warm background
    final bgColor = isDark ? AppTheme.primaryDark : const Color(0xFFFEF6EE);

    if (_isLoading) {
      return Container(
        color: bgColor,
        child: const Center(child: CircularProgressIndicator(color: AppTheme.accentOrange)),
      );
    }

    if (_courses.isEmpty) {
      return Container(
        color: bgColor,
        child: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(40),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(color: AppTheme.accentOrange.withValues(alpha: 0.12), shape: BoxShape.circle),
                    child: const Icon(Icons.wifi_off_rounded, size: 48, color: AppTheme.accentOrange),
                  ),
                  const SizedBox(height: 20),
                  Text('No courses found', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  Text('This can happen on first launch or in offline mode.', textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodyMedium),
                  const SizedBox(height: 24),
                  ElevatedButton.icon(onPressed: () { _courseService.clearCache(); _loadCourses(); }, icon: const Icon(Icons.refresh, size: 18), label: const Text('Refresh')),
                ],
              ),
            ),
          ),
        ),
      );
    }

    return Container(
      color: bgColor,
      child: SafeArea(
        child: RefreshIndicator(
          onRefresh: () async { _courseService.clearCache(); await _loadCourses(); },
          color: AppTheme.accentOrange,
          child: SingleChildScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // ── Top bar: status + avatar ──────────────────
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 12, 20, 0),
                  child: Row(
                    children: [
                      // Status badge
                      _buildStatusBadge(),
                      const Spacer(),
                      // User avatar
                      Consumer<AuthService>(
                        builder: (context, auth, _) {
                          final initial = (auth.currentUser?.username ?? 'U')[0].toUpperCase();
                          return Container(
                            width: 40,
                            height: 40,
                            decoration: BoxDecoration(
                              gradient: AppTheme.primaryGradient,
                              shape: BoxShape.circle,
                            ),
                            child: Center(child: Text(initial, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white))),
                          );
                        },
                      ),
                    ],
                  ),
                ),

                // ── Big greeting ────────────────────────────
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
                  child: Consumer<AuthService>(
                    builder: (context, auth, _) {
                      final name = auth.currentUser?.username ?? 'Learner';
                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '${_getGreeting()},',
                            style: TextStyle(
                              fontSize: 15,
                              color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            'Hi $name,',
                            style: TextStyle(
                              fontSize: 32,
                              fontWeight: FontWeight.w800,
                              color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary,
                              height: 1.1,
                            ),
                          ),
                          Text(
                            "let's start learning.",
                            style: TextStyle(
                              fontSize: 32,
                              fontWeight: FontWeight.w800,
                              color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary,
                              height: 1.2,
                            ),
                          ),
                        ],
                      );
                    },
                  ),
                ),

                // ── Model download banner ───────────────────
                _buildModelDownloadBanner(isDark),

                // ── Hero progress card (purple/orange gradient) ──
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
                  child: _buildHeroCard(isDark),
                ),

                // ── Quick Tools ────────────────────────────
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
                  child: Row(
                    children: [
                      _buildQuickTool(
                        icon: Icons.quiz_rounded,
                        label: 'Practice\nQuiz',
                        color: AppTheme.accentOrange,
                        isDark: isDark,
                        onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const QuizScreen())),
                      ),
                      const SizedBox(width: 12),
                      _buildQuickTool(
                        icon: Icons.menu_book_rounded,
                        label: 'Key\nTerms',
                        color: const Color(0xFF6366F1),
                        isDark: isDark,
                        onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const GlossaryScreen())),
                      ),
                      const SizedBox(width: 12),
                      _buildQuickTool(
                        icon: Icons.code_rounded,
                        label: 'Python\nPlayground',
                        color: const Color(0xFF10B981),
                        isDark: isDark,
                        onTap: () => setState(() => _playgroundActive = true),
                      ),
                    ],
                  ),
                ),

                // ── Recommended section ─────────────────────
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 28, 20, 0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('Recommended', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
                      GestureDetector(
                        onTap: _isSyncing ? null : _syncRemoteCourses,
                        child: Text(
                          _isSyncing ? 'Syncing...' : 'See all',
                          style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: AppTheme.accentOrange),
                        ),
                      ),
                    ],
                  ),
                ),

                // ── Horizontal course cards ─────────────────
                SizedBox(
                  height: 195,
                  child: ListView.builder(
                    scrollDirection: Axis.horizontal,
                    padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
                    itemCount: _courses.length,
                    itemBuilder: (context, i) => _buildHorizontalCard(_courses[i], isDark),
                  ),
                ),

                // ── Continue Learning ───────────────────────
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 24, 20, 8),
                  child: Text('Continue Learning', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
                ),

                // ── Vertical course list ────────────────────
                ListView.builder(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 0),
                  itemCount: _courses.length + 1,
                  itemBuilder: (context, i) {
                    if (i == _courses.length) return _buildSyncFooter(isDark);
                    return _buildVerticalCard(_courses[i], isDark);
                  },
                ),

                const SizedBox(height: 90),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ── Quick tool card (labeled) ─────────────────────────────────────────────
  Widget _buildQuickTool({
    required IconData icon,
    required String label,
    required Color color,
    required bool isDark,
    required VoidCallback onTap,
  }) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color: isDark ? AppTheme.surfaceCard : Colors.white,
            borderRadius: BorderRadius.circular(18),
            boxShadow: isDark ? null : [
              BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 10, offset: const Offset(0, 3)),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: color, size: 22),
              ),
              const SizedBox(height: 8),
              Text(
                label,
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                  height: 1.2,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Hero progress card ────────────────────────────────────────────────────
  Widget _buildHeroCard(bool isDark) {
    final pct = (_overallProgress * 100).toInt();
    final completed = _totalCompletedLessons;
    final total = _totalLessons;

    // Purple gradient in dark, orange gradient in light (like the reference design)
    final gradient = isDark
        ? const LinearGradient(colors: [Color(0xFF5B3E96), Color(0xFF3D2473)], begin: Alignment.topLeft, end: Alignment.bottomRight)
        : const LinearGradient(colors: [Color(0xFF7C4DFF), Color(0xFF5B3E96)], begin: Alignment.topLeft, end: Alignment.bottomRight);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        gradient: gradient,
        borderRadius: BorderRadius.circular(28),
        boxShadow: [
          BoxShadow(color: const Color(0xFF5B3E96).withValues(alpha: 0.35), blurRadius: 24, offset: const Offset(0, 10)),
        ],
      ),
      child: Row(
        children: [
          // Progress circle
          SizedBox(
            width: 80,
            height: 80,
            child: Stack(
              alignment: Alignment.center,
              children: [
                SizedBox(
                  width: 80,
                  height: 80,
                  child: CircularProgressIndicator(
                    value: _overallProgress,
                    strokeWidth: 7,
                    backgroundColor: Colors.white.withValues(alpha: 0.15),
                    valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                    strokeCap: StrokeCap.round,
                  ),
                ),
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text('$pct%', style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w800)),
                    Text('done', style: TextStyle(color: Colors.white.withValues(alpha: 0.7), fontSize: 10)),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Continue your\nlessons with excited.',
                  style: TextStyle(color: Colors.white.withValues(alpha: 0.9), fontSize: 14, height: 1.4),
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(color: Colors.white.withValues(alpha: 0.2), borderRadius: BorderRadius.circular(12)),
                      child: Text('$completed / $total', style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600)),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                SizedBox(
                  height: 34,
                  child: ElevatedButton(
                    onPressed: _courses.isNotEmpty ? () => _handleCourseTap(_courses.first) : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: const Color(0xFF5B3E96),
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(horizontal: 24),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                    ),
                    child: const Text('Continue', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 13)),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Horizontal course card ────────────────────────────────────────────────
  Widget _buildHorizontalCard(Course course, bool isDark) {
    final progress = course.progress;
    final cardBg = isDark ? AppTheme.surfaceCard : Colors.white;

    return GestureDetector(
      onTap: () => _handleCourseTap(course),
      child: Container(
        width: 155,
        margin: const EdgeInsets.only(right: 14),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: cardBg,
          borderRadius: BorderRadius.circular(22),
          boxShadow: isDark ? null : [
            BoxShadow(color: Colors.black.withValues(alpha: 0.06), blurRadius: 16, offset: const Offset(0, 6)),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(gradient: AppTheme.primaryGradient, borderRadius: BorderRadius.circular(14)),
              child: const Icon(Icons.school_rounded, color: Colors.white, size: 20),
            ),
            const SizedBox(height: 10),
            Text(
              course.title,
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
            const Spacer(),
            Text(
              'Lessons ${course.completedLessons}/${course.totalLessons}',
              style: TextStyle(fontSize: 11, color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted),
            ),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: progress,
                backgroundColor: isDark ? AppTheme.primaryLight.withValues(alpha: 0.3) : AppTheme.tagBackground,
                valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
                minHeight: 4,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Vertical course card ──────────────────────────────────────────────────
  Widget _buildVerticalCard(Course course, bool isDark) {
    final cardBg = isDark ? AppTheme.surfaceCard : Colors.white;

    return GestureDetector(
      onTap: () => _handleCourseTap(course),
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: cardBg,
          borderRadius: BorderRadius.circular(20),
          boxShadow: isDark ? null : [
            BoxShadow(color: Colors.black.withValues(alpha: 0.05), blurRadius: 12, offset: const Offset(0, 4)),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 50,
              height: 50,
              decoration: BoxDecoration(gradient: AppTheme.primaryGradient, borderRadius: BorderRadius.circular(14)),
              child: const Icon(Icons.school_rounded, color: Colors.white, size: 24),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(course.title, style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary), maxLines: 1, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      _tag(course.difficulty, isDark),
                      const SizedBox(width: 6),
                      _tag(course.duration, isDark),
                    ],
                  ),
                  const SizedBox(height: 8),
                  // inline progress
                  Row(
                    children: [
                      Expanded(
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: LinearProgressIndicator(
                            value: course.progress,
                            backgroundColor: isDark ? AppTheme.primaryLight.withValues(alpha: 0.3) : AppTheme.tagBackground,
                            valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
                            minHeight: 4,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text('${(course.progress * 100).toInt()}%', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: AppTheme.accentOrange)),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(width: 6),
            Icon(Icons.chevron_right_rounded, color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted, size: 22),
          ],
        ),
      ),
    );
  }

  Widget _tag(String text, bool isDark) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: isDark ? AppTheme.accentOrange.withValues(alpha: 0.12) : AppTheme.accentOrangeSubtle,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(text, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: isDark ? AppTheme.accentOrangeLight : AppTheme.accentOrangeDark)),
    );
  }

  // ── Status badge ──────────────────────────────────────────────────────────
  Widget _buildStatusBadge() {
    return StreamBuilder<EngineStatus>(
      stream: _hybridService.statusStream,
      initialData: _hybridService.currentStatus,
      builder: (context, snapshot) {
        final isOnline = (snapshot.data ?? EngineStatus.offline) == EngineStatus.online;
        final color = isOnline ? AppTheme.success : AppTheme.warning;
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(color: color.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(20), border: Border.all(color: color.withValues(alpha: 0.3))),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(width: 6, height: 6, decoration: BoxDecoration(shape: BoxShape.circle, color: color)),
              const SizedBox(width: 5),
              Text(isOnline ? 'Online' : 'Offline', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: color)),
            ],
          ),
        );
      },
    );
  }

  // ── Sync footer ───────────────────────────────────────────────────────────
  Widget _buildSyncFooter(bool isDark) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        children: [
          if (_syncError != null)
            Container(
              margin: const EdgeInsets.only(bottom: 10),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(color: AppTheme.warning.withValues(alpha: 0.1), borderRadius: BorderRadius.circular(12), border: Border.all(color: AppTheme.warning.withValues(alpha: 0.35))),
              child: Row(children: [
                Icon(Icons.warning_amber_rounded, size: 16, color: AppTheme.warning),
                const SizedBox(width: 8),
                Expanded(child: Text(_syncError!, style: TextStyle(fontSize: 12, color: AppTheme.warning, fontWeight: FontWeight.w500))),
              ]),
            ),
          if (_newCoursesFound > 0)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text('$_newCoursesFound new course${_newCoursesFound > 1 ? 's' : ''} added!', style: TextStyle(fontSize: 12, color: AppTheme.success, fontWeight: FontWeight.w600)),
            ),
        ],
      ),
    );
  }

  // ── Model download banner ─────────────────────────────────────────────────
  Widget _buildModelDownloadBanner(bool isDark) {
    return Consumer<ModelDownloadService>(
      builder: (context, dl, _) {
        if (dl.status == DownloadStatus.completed) return const SizedBox.shrink();
        final isDownloading = dl.status == DownloadStatus.downloading || dl.status == DownloadStatus.connecting;
        final pct = dl.progress;

        return GestureDetector(
          onTap: () => setState(() => _currentIndex = 3),
          child: Container(
            margin: const EdgeInsets.fromLTRB(20, 12, 20, 0),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: isDark ? AppTheme.accentOrange.withValues(alpha: 0.12) : AppTheme.accentOrange.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppTheme.accentOrange.withValues(alpha: 0.35)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(isDownloading ? Icons.downloading_rounded : dl.status == DownloadStatus.error ? Icons.error_outline_rounded : Icons.download_for_offline_outlined, size: 16, color: AppTheme.accentOrange),
                  const SizedBox(width: 8),
                  Expanded(child: Text(
                    isDownloading ? 'Downloading offline AI model… ${(pct * 100).toInt()}%' : dl.status == DownloadStatus.error ? 'Model download failed — tap to retry' : 'Download AI model for offline use →',
                    style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary),
                  )),
                  Icon(Icons.chevron_right_rounded, size: 16, color: AppTheme.accentOrange),
                ]),
                if (isDownloading) ...[
                  const SizedBox(height: 6),
                  ClipRRect(borderRadius: BorderRadius.circular(3), child: LinearProgressIndicator(value: pct > 0 ? pct : null, backgroundColor: AppTheme.accentOrange.withValues(alpha: 0.15), valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange), minHeight: 3)),
                ],
              ],
            ),
          ),
        );
      },
    );
  }

  // ── Course tap handler ────────────────────────────────────────────────────
  void _handleCourseTap(Course course) async {
    if (course.modules.isEmpty) {
      showDialog(context: context, barrierDismissible: false, builder: (_) => const Center(child: CircularProgressIndicator(color: AppTheme.accentOrange)));
      final full = await _courseService.loadCourse(course.id);
      if (mounted) {
        Navigator.pop(context);
        if (full != null) { _navigateToDetail(full); } else { AppSnackBar.error(context, 'Failed to load course content.'); }
      }
    } else {
      _navigateToDetail(course);
    }
  }

  void _navigateToDetail(Course course) {
    Navigator.push(context, MaterialPageRoute(builder: (_) => CourseDetailScreen(course: course))).then((_) {
      // Force refresh so completion status is re-read from SharedPreferences
      _courseService.clearCache();
      _loadCourses();
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ── EXPLORE / AI TUTOR TAB ────────────────────────────────────────────────
  // ═══════════════════════════════════════════════════════════════════════════
  Widget _buildChatTab(bool isDark) {
    final bgColor = isDark ? AppTheme.primaryDark : const Color(0xFFFEF6EE);

    return Container(
      color: bgColor,
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Header area ─────────────────────────────
            Container(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
              decoration: BoxDecoration(
                gradient: isDark
                    ? const LinearGradient(colors: [Color(0xFF1E293B), Color(0xFF0F172A)], begin: Alignment.topLeft, end: Alignment.bottomRight)
                    : const LinearGradient(colors: [Color(0xFFFEE4C4), Color(0xFFFEF6EE)], begin: Alignment.topCenter, end: Alignment.bottomCenter),
                borderRadius: const BorderRadius.only(bottomLeft: Radius.circular(36), bottomRight: Radius.circular(36)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Learn From\nAnywhere',
                    style: TextStyle(
                      fontSize: 34,
                      fontWeight: FontWeight.w800,
                      color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary,
                      height: 1.1,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Discover answers through questions',
                    style: TextStyle(fontSize: 14, color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary),
                  ),
                ],
              ),
            ),

            // ── Scrollable body ─────────────────────────
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Category chips
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        children: _topicCategories.asMap().entries.map((e) {
                          final isSel = e.key == 0;
                          return Padding(
                            padding: const EdgeInsets.only(right: 8),
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
                              decoration: BoxDecoration(
                                color: isSel ? AppTheme.accentOrange : (isDark ? AppTheme.surfaceCard : Colors.white),
                                borderRadius: BorderRadius.circular(22),
                                boxShadow: !isDark && !isSel ? [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 8, offset: const Offset(0, 2))] : null,
                              ),
                              child: Text(e.value, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: isSel ? Colors.white : (isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary))),
                            ),
                          );
                        }).toList(),
                      ),
                    ),

                    const SizedBox(height: 24),

                    // Featured card
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(24),
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
                          Container(
                            width: 52,
                            height: 52,
                            decoration: BoxDecoration(color: Colors.white.withValues(alpha: 0.2), borderRadius: BorderRadius.circular(16)),
                            child: const Icon(Icons.psychology_rounded, size: 30, color: Colors.white),
                          ),
                          const SizedBox(height: 16),
                          const Text('Socratic AI Tutor', style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w700)),
                          const SizedBox(height: 6),
                          Text('I guide you to discover answers through thoughtful questions', style: TextStyle(color: Colors.white.withValues(alpha: 0.85), fontSize: 14, height: 1.4)),
                          const SizedBox(height: 16),
                          ElevatedButton.icon(
                              onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ChatScreen())),
                              icon: const Icon(Icons.chat_bubble_outline, size: 16),
                              label: const Text('Start Conversation', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.white,
                                foregroundColor: isDark ? const Color(0xFF5B3E96) : AppTheme.accentOrange,
                                elevation: 0,
                                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
                              ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 28),

                    Text('Try asking...', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: _starterPrompts.map((p) => _promptChip(p, isDark)).toList(),
                    ),

                    const SizedBox(height: 90),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _promptChip(String prompt, bool isDark) {
    return GestureDetector(
      onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => ChatScreen(initialMessage: prompt))),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: isDark ? AppTheme.surfaceCard : Colors.white,
          borderRadius: BorderRadius.circular(20),
          boxShadow: isDark ? null : [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 8, offset: const Offset(0, 2))],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.lightbulb_outline_rounded, size: 14, color: AppTheme.accentOrange),
            const SizedBox(width: 6),
            Text(prompt, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500, color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary)),
          ],
        ),
      ),
    );
  }
}
