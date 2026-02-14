import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../services/theme_service.dart';
import 'model_setup_screen.dart';

class OnboardingScreen extends StatelessWidget {
  const OnboardingScreen({super.key});

  void _navigateToSetup(BuildContext context) {
    Navigator.of(context).pushReplacement(
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) => const ModelSetupScreen(),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          return FadeTransition(opacity: animation, child: child);
        },
        transitionDuration: const Duration(milliseconds: 400),
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
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                const Spacer(flex: 2),
                
                // App icon
                Container(
                  width: 100,
                  height: 100,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: AppTheme.primaryGradient,
                    boxShadow: [
                      BoxShadow(
                        color: AppTheme.accentOrange.withOpacity(0.4),
                        blurRadius: 30,
                        spreadRadius: 5,
                      ),
                    ],
                  ),
                  child: const Icon(
                    Icons.psychology,
                    size: 50,
                    color: Colors.white,
                  ),
                ),
                
                const SizedBox(height: 48),
                
                // Simple title
                Text(
                  'Socratic AI Tutor',
                  style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
                
                const SizedBox(height: 16),
                
                // Simple subtitle
                Text(
                  'Learn through guided questions',
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                  ),
                  textAlign: TextAlign.center,
                ),
                
                const Spacer(flex: 3),
                
                // Get Started button
                SizedBox(
                  width: double.infinity,
                  child: Container(
                    decoration: BoxDecoration(
                      gradient: AppTheme.buttonGradient,
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme.accentOrange.withOpacity(0.4),
                          blurRadius: 20,
                          offset: const Offset(0, 8),
                        ),
                      ],
                    ),
                    child: ElevatedButton(
                      onPressed: () => _navigateToSetup(context),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.transparent,
                        shadowColor: Colors.transparent,
                        padding: const EdgeInsets.symmetric(vertical: 18),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                      ),
                      child: const Text(
                        'Get Started',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ),
                ),
                
                const SizedBox(height: 48),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
