import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class FeatureTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String description;
  final List<Color> gradientColors;
  final VoidCallback? onTap;

  const FeatureTile({
    super.key,
    required this.icon,
    required this.title,
    required this.description,
    required this.gradientColors,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppTheme.surfaceCard,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: gradientColors[0].withValues(alpha: 0.2),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    gradientColors[0].withValues(alpha: 0.2),
                    gradientColors[1].withValues(alpha: 0.1),
                  ],
                ),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(
                icon,
                color: gradientColors[0],
                size: 22,
              ),
            ),
            const Spacer(),
            Text(
              title,
              style: const TextStyle(
                color: AppTheme.textPrimary,
                fontWeight: FontWeight.w600,
                fontSize: 14,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              description,
              style: const TextStyle(
                color: AppTheme.textMuted,
                fontSize: 11,
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }
}
