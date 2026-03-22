import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../services/theme_service.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: isDark
              ? AppTheme.backgroundGradient
              : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: Column(
            children: [
              // ── Header ──────────────────────────────────────────────
              Padding(
                padding: const EdgeInsets.fromLTRB(8, 8, 16, 0),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_rounded),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      'About Bantaba AI',
                      style: Theme.of(context)
                          .textTheme
                          .headlineSmall
                          ?.copyWith(fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
              ),

              // ── Body ────────────────────────────────────────────────
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
                  children: [
                    // Logo + tagline
                    Center(
                      child: Column(
                        children: [
                          Container(
                            width: 90,
                            height: 90,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              boxShadow: [
                                BoxShadow(
                                  color: AppTheme.accentOrange
                                      .withValues(alpha: 0.35),
                                  blurRadius: 24,
                                  spreadRadius: 4,
                                ),
                              ],
                            ),
                            child: ClipOval(
                              child: Image.asset(
                                'assets/images/logo.png',
                                fit: BoxFit.cover,
                                errorBuilder: (_, __, ___) => Container(
                                  decoration: const BoxDecoration(
                                    shape: BoxShape.circle,
                                    gradient: AppTheme.primaryGradient,
                                  ),
                                  child: const Icon(Icons.psychology,
                                      size: 44, color: Colors.white),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 16),
                          Text(
                            'Bantaba AI',
                            style: Theme.of(context)
                                .textTheme
                                .headlineMedium
                                ?.copyWith(fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Your offline-first Socratic tutor',
                            style: Theme.of(context)
                                .textTheme
                                .bodyMedium
                                ?.copyWith(fontStyle: FontStyle.italic),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Version 1.0 | March 2026',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 28),

                    // What is Bantaba AI
                    _buildSection(
                      context,
                      icon: Icons.smart_toy_outlined,
                      title: 'What is Bantaba AI?',
                      body:
                          'Bantaba AI is an offline-first Socratic tutoring '
                          'app built to support learners studying data science '
                          'and machine learning. It was designed specifically '
                          'for low-resource environments where internet access '
                          'is unreliable or unavailable.',
                      isDark: isDark,
                    ),

                    // How it works
                    _buildSection(
                      context,
                      icon: Icons.psychology_outlined,
                      title: 'How It Works',
                      body:
                          'The app uses a small AI language model that runs '
                          'directly on your device. When you ask a question, '
                          'instead of giving you a direct answer, the tutor '
                          'responds with guiding questions to help you think '
                          'through the problem yourself. This method is called '
                          'Socratic tutoring and it is backed by educational '
                          'research showing that guided reasoning leads to '
                          'deeper understanding than simply being given answers.',
                      isDark: isDark,
                    ),

                    // Who built it
                    _buildSection(
                      context,
                      icon: Icons.engineering_outlined,
                      title: 'Who Built It?',
                      body:
                          'Bantaba AI was developed by Omar Keita as a final '
                          'year Capstone project for the BSc in Software '
                          'Engineering at African Leadership University. '
                          'The project was supervised by '
                          'Mr. Marvin Muyonga Ogore.',
                      isDark: isDark,
                    ),

                    // Limitations
                    _buildSection(
                      context,
                      icon: Icons.warning_amber_rounded,
                      title: 'Limitations',
                      body:
                          'This is a research prototype. The AI can make '
                          'mistakes. It works best for introductory data '
                          'science and machine learning topics. It is not a '
                          'substitute for a qualified teacher or verified '
                          'learning materials.',
                      isDark: isDark,
                    ),

                    // Technology
                    _buildSection(
                      context,
                      icon: Icons.memory_outlined,
                      title: 'Technology',
                      body:
                          'Built with Flutter, FastAPI, llama.cpp, and the '
                          'Qwen3-0.6B open weight model by Alibaba Cloud.',
                      isDark: isDark,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSection(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String body,
    required bool isDark,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: AppTheme.accentOrange.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Icon(icon, color: AppTheme.accentOrange, size: 20),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    title,
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                body,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      height: 1.55,
                    ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
