import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../services/theme_service.dart';
import '../widgets/gradient_card.dart';

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;
    final colorScheme = Theme.of(context).colorScheme;

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
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      IconButton(
                        icon: Icon(isDark ? Icons.light_mode : Icons.dark_mode),
                        onPressed: () => context.read<ThemeService>().toggleTheme(),
                        color: colorScheme.onSurface,
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  
                  // Profile Picture
                  Container(
                    width: 100,
                    height: 100,
                    decoration: BoxDecoration(
                      gradient: AppTheme.primaryGradient,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme.accentOrange.withOpacity(0.4),
                          blurRadius: 24,
                          offset: const Offset(0, 8),
                        ),
                      ],
                    ),
                    child: const Icon(
                      Icons.person,
                      size: 50,
                      color: Colors.white,
                    ),
                  ),
                  
                  const SizedBox(height: 16),
                  
                  Text(
                    'Student',
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                  
                  const SizedBox(height: 4),
                  
                  Text(
                    'Learning enthusiast',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                    ),
                  ),
                  
                  const SizedBox(height: 24),
                  
                  // Stats Row
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _buildStatItem(context, '12', 'Sessions'),
                      _buildDivider(context),
                      _buildStatItem(context, '48', 'Questions'),
                      _buildDivider(context),
                      _buildStatItem(context, '5', 'Topics'),
                    ],
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
                  
                  GradientCard(
                    gradient: LinearGradient(
                      colors: isDark 
                          ? [const Color(0xFF1E1B2E), const Color(0xFF151226)]
                          : [Colors.white, Colors.white.withOpacity(0.9)],
                    ),
                    borderGradient: isDark 
                        ? AppTheme.borderGradient 
                        : LinearGradient(colors: [AppTheme.accentOrange.withOpacity(0.2), AppTheme.accentOrange.withOpacity(0.1)]),
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        children: [
                          _buildProgressItem(
                            context,
                            'Critical Thinking',
                            0.75,
                            AppTheme.accentOrange,
                          ),
                          const SizedBox(height: 16),
                          _buildProgressItem(
                            context,
                            'Problem Solving',
                            0.60,
                            const Color(0xFF3B82F6),
                          ),
                          const SizedBox(height: 16),
                          _buildProgressItem(
                            context,
                            'Analytical Skills',
                            0.85,
                            const Color(0xFF10B981),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Recent Achievements
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Achievements',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 16),
                  
                  SizedBox(
                    height: 120,
                    child: ListView(
                      scrollDirection: Axis.horizontal,
                      children: [
                        _buildAchievementCard(
                          context,
                          Icons.lightbulb,
                          'Quick Learner',
                          'Completed 5 sessions',
                          const Color(0xFFF59E0B),
                        ),
                        _buildAchievementCard(
                          context,
                          Icons.star,
                          'Deep Thinker',
                          'Asked insightful questions',
                          AppTheme.accentOrange,
                        ),
                        _buildAchievementCard(
                          context,
                          Icons.trending_up,
                          'Consistent',
                          '7 day streak',
                          const Color(0xFF10B981),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Settings Section
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Settings',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 16),
                  
                  _buildSettingsTile(
                    context,
                    Icons.psychology,
                    'Model Settings',
                    'Configure LLM parameters',
                  ),
                  _buildSettingsTile(
                    context,
                    Icons.school,
                    'Difficulty Level',
                    'Adjust scaffolding intensity',
                  ),
                  _buildSettingsTile(
                    context,
                    Icons.notifications_outlined,
                    'Notifications',
                    'Manage alerts and reminders',
                  ),
                  _buildSettingsTile(
                    context,
                    Icons.palette_outlined,
                    'Appearance',
                    'Theme and display settings',
                  ),
                  _buildSettingsTile(
                    context,
                    Icons.info_outline,
                    'About',
                    'App version and info',
                  ),
                  
                  const SizedBox(height: 100),
                ],
              ),
            ),
          ),
        ],
      ),
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
      color: isDark ? AppTheme.primaryLight.withOpacity(0.3) : Colors.grey.withOpacity(0.3),
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
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: TextStyle(color: colorScheme.onSurface),
            ),
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
            backgroundColor: isDark ? AppTheme.primaryLight.withOpacity(0.3) : Colors.grey.withOpacity(0.2),
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 6,
          ),
        ),
      ],
    );
  }

  Widget _buildAchievementCard(
    BuildContext context,
    IconData icon,
    String title,
    String description,
    Color color,
  ) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      width: 140,
      margin: const EdgeInsets.only(right: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: color.withOpacity(0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, color: color, size: 20),
          ),
          const Spacer(),
          Text(
            title,
            style: TextStyle(
              color: colorScheme.onSurface,
              fontWeight: FontWeight.w600,
              fontSize: 13,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            description,
            style: TextStyle(
              color: isDark ? AppTheme.textMuted : AppTheme.lightTextSecondary.withOpacity(0.6),
              fontSize: 11,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }

  Widget _buildSettingsTile(
    BuildContext context,
    IconData icon,
    String title,
    String subtitle,
  ) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: isDark ? null : Border.all(color: Colors.grey.withOpacity(0.1)),
      ),
      child: ListTile(
        leading: Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: AppTheme.accentOrange.withOpacity(0.15),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Icon(icon, color: AppTheme.accentOrange, size: 22),
        ),
        title: Text(
          title,
          style: TextStyle(
            color: colorScheme.onSurface,
            fontWeight: FontWeight.w500,
          ),
        ),
        subtitle: Text(
          subtitle,
          style: TextStyle(
            color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
            fontSize: 12,
          ),
        ),
        trailing: Icon(
          Icons.chevron_right,
          color: isDark ? AppTheme.textMuted : AppTheme.lightTextSecondary.withOpacity(0.4),
        ),
        onTap: () {},
      ),
    );
  }
}
