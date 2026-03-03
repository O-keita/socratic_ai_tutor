import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../services/theme_service.dart';
import '../services/auth_service.dart';
import '../services/session_service.dart';
import '../services/course_service.dart';
import '../models/course.dart';
import 'auth_screen.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => ProfileScreenState();
}

class ProfileScreenState extends State<ProfileScreen> {

  int _sessionCount = 0;
  int _questionCount = 0;
  int _topicCount = 0;
  double _avgSocraticIndex = 0.5;
  String _currentScaffolding = 'Intermediate';
  String _topSentiment = 'Neutral';
  List<Course> _courses = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  /// Public method so the parent (HomeScreen) can trigger a refresh
  /// when the user switches to the Profile tab.
  void refresh() => _loadData();

  Future<void> _loadData() async {
    try {
      final sessions = await SessionService.getSessions();
      final courses = await CourseService().getCourses();

      int msgCount = 0;
      final topics = <String>{};
      double totalSocraticIndex = 0;
      int socraticCount = 0;

      final sentiments = <String, int>{};
      String lastLevel = 'Intermediate';

      for (var s in sessions) {
        msgCount += s.messages.where((m) => m.isUser).length;
        if (s.topic.isNotEmpty) topics.add(s.topic);

        for (var m in s.messages) {
          if (m.metadata != null) {
            if (m.metadata!['socratic_index'] != null) {
              totalSocraticIndex += (m.metadata!['socratic_index'] as num).toDouble();
              socraticCount++;
            }
            if (m.metadata!['scaffolding_level'] != null) {
              lastLevel = m.metadata!['scaffolding_level'].toString();
            }
            if (m.metadata!['sentiment'] != null) {
              final sent = m.metadata!['sentiment'].toString();
              sentiments[sent] = (sentiments[sent] ?? 0) + 1;
            }
          }
        }
      }

      String dominantSentiment = 'Neutral';
      if (sentiments.isNotEmpty) {
        dominantSentiment = sentiments.entries
            .reduce((a, b) => a.value > b.value ? a : b)
            .key;
        dominantSentiment = dominantSentiment[0].toUpperCase() + dominantSentiment.substring(1).replaceAll('_', ' ');
      }

      if (mounted) {
        setState(() {
          _sessionCount = sessions.length;
          _questionCount = msgCount;
          _topicCount = topics.length;
          _avgSocraticIndex = socraticCount > 0 ? totalSocraticIndex / socraticCount : 0.5;
          _currentScaffolding = lastLevel[0].toUpperCase() + lastLevel.substring(1);
          _topSentiment = dominantSentiment;
          _courses = courses;
          _isLoading = false;
        });
      }
    } catch (e) {
      debugPrint('Error loading profile data: $e');
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;
    final authService = context.watch<AuthService>();
    final user = authService.currentUser;

    return CustomScrollView(
      slivers: [
        // ── Gradient header with avatar ──────────────────────────────
        SliverToBoxAdapter(
          child: Container(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
            decoration: BoxDecoration(
              gradient: isDark
                  ? AppTheme.headerGradientDark
                  : AppTheme.headerGradientLight,
              borderRadius: const BorderRadius.only(
                bottomLeft: Radius.circular(32),
                bottomRight: Radius.circular(32),
              ),
            ),
            child: Column(
              children: [
                // Title row
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Profile',
                      style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    IconButton(
                      icon: Icon(isDark ? Icons.light_mode : Icons.dark_mode),
                      onPressed: () => context.read<ThemeService>().toggleTheme(),
                    ),
                  ],
                ),
                const SizedBox(height: 20),

                // Avatar
                Container(
                  width: 90,
                  height: 90,
                  decoration: BoxDecoration(
                    gradient: AppTheme.primaryGradient,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: AppTheme.accentOrange.withValues(alpha: 0.35),
                        blurRadius: 20,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Center(
                    child: Text(
                      (user?.username ?? 'U')[0].toUpperCase(),
                      style: const TextStyle(
                        fontSize: 36,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 14),

                Text(
                  user?.username ?? 'Learner',
                  style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  user?.email ?? 'Learning enthusiast',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                  ),
                ),
              ],
            ),
          ),
        ),

        // ── Stats row ────────────────────────────────────────────────
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
            child: Row(
              children: [
                Expanded(child: _buildStatCircle(context, _sessionCount, 'Sessions', AppTheme.accentOrange, isDark)),
                const SizedBox(width: 12),
                Expanded(child: _buildStatCircle(context, _questionCount, 'Questions', const Color(0xFF6366F1), isDark)),
                const SizedBox(width: 12),
                Expanded(child: _buildStatCircle(context, _topicCount, 'Topics', const Color(0xFF10B981), isDark)),
              ],
            ),
          ),
        ),

        // ── Learning Progress ────────────────────────────────────────
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 24, 20, 0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Learning Progress',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 14),

                if (_isLoading)
                  const Center(child: CircularProgressIndicator())
                else if (_courses.isEmpty)
                  const Text('No course progress yet.')
                else
                  ..._courses.take(3).map((course) => _buildProgressCard(
                    context, course.title, course.progress, isDark,
                  )),
              ],
            ),
          ),
        ),

        // ── Socratic Insights ────────────────────────────────────────
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Socratic Insights',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 14),
                Container(
                  padding: const EdgeInsets.all(18),
                  decoration: BoxDecoration(
                    color: isDark ? AppTheme.surfaceCard : Colors.white,
                    borderRadius: BorderRadius.circular(AppTheme.cardRadiusLarge),
                    border: Border.all(
                      color: isDark
                          ? AppTheme.accentOrange.withValues(alpha: 0.15)
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
                  child: Column(
                    children: [
                      _buildInsightRow(
                        context,
                        'Socratic Index',
                        '${(_avgSocraticIndex * 100).toStringAsFixed(0)}%',
                        'Contribution ratio',
                        Icons.insights,
                        AppTheme.accentOrange,
                        isDark,
                      ),
                      Padding(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        child: Divider(
                          height: 1,
                          color: isDark
                              ? AppTheme.primaryLight.withValues(alpha: 0.2)
                              : AppTheme.tagBackground,
                        ),
                      ),
                      _buildInsightRow(
                        context,
                        'Learning Level',
                        _currentScaffolding,
                        'Dynamic difficulty',
                        Icons.speed,
                        const Color(0xFF0EA5E9),
                        isDark,
                      ),
                      Padding(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        child: Divider(
                          height: 1,
                          color: isDark
                              ? AppTheme.primaryLight.withValues(alpha: 0.2)
                              : AppTheme.tagBackground,
                        ),
                      ),
                      _buildInsightRow(
                        context,
                        'Top Sentiment',
                        _topSentiment,
                        'Confidence indicator',
                        Icons.mood,
                        _topSentiment.contains('Low') ? Colors.orange : AppTheme.success,
                        isDark,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),

        // ── Logout Button ────────────────────────────────────────────
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 100),
            child: SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () async {
                  await authService.logout();
                  if (mounted) {
                    Navigator.of(context).pushAndRemoveUntil(
                      MaterialPageRoute(builder: (_) => const AuthScreen()),
                      (route) => false,
                    );
                  }
                },
                icon: const Icon(Icons.logout, color: Colors.redAccent),
                label: const Text('Logout', style: TextStyle(color: Colors.redAccent)),
                style: OutlinedButton.styleFrom(
                  side: const BorderSide(color: Colors.redAccent),
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  // ── Stat circle card ──────────────────────────────────────────────────────
  Widget _buildStatCircle(BuildContext context, int value, String label, Color color, bool isDark) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 18),
      decoration: BoxDecoration(
        color: isDark ? AppTheme.surfaceCard : Colors.white,
        borderRadius: BorderRadius.circular(AppTheme.cardRadiusLarge),
        border: Border.all(
          color: color.withValues(alpha: 0.2),
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
      child: Column(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: color.withValues(alpha: isDark ? 0.15 : 0.1),
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(
                value.toString(),
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            label,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
            ),
          ),
        ],
      ),
    );
  }

  // ── Progress card with left color border ──────────────────────────────────
  Widget _buildProgressCard(BuildContext context, String title, double progress, bool isDark) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: isDark ? AppTheme.surfaceCard : Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isDark
              ? AppTheme.primaryLight.withValues(alpha: 0.15)
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
      child: Row(
        children: [
          // Left color accent
          Container(
            width: 4,
            height: 56,
            decoration: BoxDecoration(
              color: AppTheme.accentOrange,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(16),
                bottomLeft: Radius.circular(16),
              ),
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          title,
                          overflow: TextOverflow.ellipsis,
                          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        '${(progress * 100).toInt()}%',
                        style: TextStyle(
                          color: AppTheme.accentOrange,
                          fontWeight: FontWeight.w700,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: progress,
                      backgroundColor: isDark
                          ? AppTheme.primaryLight.withValues(alpha: 0.3)
                          : AppTheme.tagBackground,
                      valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
                      minHeight: 5,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Insight row ───────────────────────────────────────────────────────────
  Widget _buildInsightRow(
    BuildContext context,
    String label,
    String value,
    String subtitle,
    IconData icon,
    Color color,
    bool isDark,
  ) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: color.withValues(alpha: isDark ? 0.15 : 0.1),
            borderRadius: BorderRadius.circular(14),
          ),
          child: Icon(icon, color: color, size: 22),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                subtitle,
                style: TextStyle(
                  fontSize: 12,
                  color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                ),
              ),
            ],
          ),
        ),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: color.withValues(alpha: isDark ? 0.12 : 0.08),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            value,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
        ),
      ],
    );
  }
}
