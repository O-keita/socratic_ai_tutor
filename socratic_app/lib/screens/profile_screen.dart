import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../services/theme_service.dart';
import '../services/auth_service.dart';
import '../services/session_service.dart';
import '../services/course_service.dart';
import '../models/course.dart';
import '../widgets/gradient_card.dart';
import 'auth_screen.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
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
        
        // Analyze session-specific metadata
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
        // Capitalize
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
    final colorScheme = Theme.of(context).colorScheme;
    final authService = context.watch<AuthService>();
    final user = authService.currentUser;

    return Container(
      decoration: BoxDecoration(
        gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
      ),
      child: CustomScrollView(
        slivers: [
          // Profile Header
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
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
                        color: colorScheme.onSurface,
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  
                  // Profile Picture
                  Container(
                    width: 100,
                    height: 100,
                    decoration: BoxDecoration(
                      gradient: AppTheme.primaryGradient,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme.accentOrange.withValues(alpha: 0.4),
                          blurRadius: 24,
                          offset: const Offset(0, 8),
                        ),
                      ],
                    ),
                    child: Center(
                      child: Text(
                        (user?.username ?? 'U')[0].toUpperCase(),
                        style: const TextStyle(
                          fontSize: 40,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                  
                  const SizedBox(height: 16),
                  
                  Text(
                    user?.username ?? 'Learner',
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                  
                  const SizedBox(height: 4),
                  
                  Text(
                    user?.email ?? 'Learning enthusiast',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                    ),
                  ),
                  
                  const SizedBox(height: 24),

                  // Stats card
                  Container(
                    padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 8),
                    decoration: BoxDecoration(
                      color: isDark ? AppTheme.surfaceCard : Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: isDark
                            ? AppTheme.primaryLight.withValues(alpha: 0.2)
                            : AppTheme.tagBackground,
                      ),
                      boxShadow: isDark
                          ? null
                          : [
                              BoxShadow(
                                color: Colors.black.withValues(alpha: 0.05),
                                blurRadius: 16,
                                offset: const Offset(0, 4),
                              ),
                            ],
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        _buildStatItem(context, _sessionCount.toString(), 'Sessions'),
                        _buildDivider(context),
                        _buildStatItem(context, _questionCount.toString(), 'Questions'),
                        _buildDivider(context),
                        _buildStatItem(context, _topicCount.toString(), 'Topics'),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Learning Progress
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Learning Progress',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 16),
                  
                  if (_isLoading)
                    const Center(child: CircularProgressIndicator())
                  else if (_courses.isEmpty)
                    const Text('No course progress yet.')
                  else
                    GradientCard(
                      gradient: LinearGradient(
                        colors: isDark 
                            ? [const Color(0xFF1E1B2E), const Color(0xFF151226)]
                            : [Colors.white, Colors.white.withValues(alpha: 0.9)],
                      ),
                      borderGradient: isDark 
                          ? AppTheme.borderGradient 
                          : LinearGradient(colors: [AppTheme.accentOrange.withValues(alpha: 0.2), AppTheme.accentOrange.withValues(alpha: 0.1)]),
                      child: Padding(
                        padding: const EdgeInsets.all(20),
                        child: Column(
                          children: _courses.take(3).map((course) {
                            return Padding(
                              padding: const EdgeInsets.only(bottom: 16),
                              child: _buildProgressItem(
                                context,
                                course.title,
                                course.progress,
                                AppTheme.accentOrange,
                              ),
                            );
                          }).toList(),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),

          // Socratic Insights Section
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Socratic Insights',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: isDark ? AppTheme.surfaceCard : Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: isDark 
                            ? AppTheme.accentOrange.withValues(alpha: 0.2)
                            : AppTheme.tagBackground,
                      ),
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
                        ),
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 12),
                          child: Divider(height: 1, color: Colors.grey),
                        ),
                        _buildInsightRow(
                          context,
                          'Learning Level',
                          _currentScaffolding,
                          'Dynamic difficulty',
                          Icons.speed,
                          Colors.blue,
                        ),
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 12),
                          child: Divider(height: 1, color: Colors.grey),
                        ),
                        _buildInsightRow(
                          context,
                          'Top Sentiment',
                          _topSentiment,
                          'Confidence indicator',
                          Icons.mood,
                          _topSentiment.contains('Low') ? Colors.orange : Colors.green,
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Achievements Section

          // Logout Button
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
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInsightRow(
    BuildContext context,
    String label,
    String value,
    String subtitle,
    IconData icon,
    Color color,
  ) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: color, size: 24),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
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
        Text(
          value,
          style: TextStyle(
            color: color,
            fontWeight: FontWeight.bold,
            fontSize: 16,
          ),
        ),
      ],
    );
  }

  Widget _buildStatItem(BuildContext context, String value, String label) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final colorScheme = Theme.of(context).colorScheme;

    return Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
            color: colorScheme.onSurface,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
          ),
        ),
      ],
    );
  }

  Widget _buildDivider(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      width: 1,
      height: 40,
      color: isDark ? AppTheme.primaryLight.withValues(alpha: 0.3) : Colors.grey.withValues(alpha: 0.3),
    );
  }

  Widget _buildProgressItem(
    BuildContext context,
    String label,
    double progress,
    Color color,
  ) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final colorScheme = Theme.of(context).colorScheme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                label,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(color: colorScheme.onSurface),
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '${(progress * 100).toInt()}%',
              style: TextStyle(
                color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
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
            backgroundColor: isDark ? AppTheme.primaryLight.withValues(alpha: 0.3) : Colors.grey.withValues(alpha: 0.2),
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 6,
          ),
        ),
      ],
    );
  }

}

