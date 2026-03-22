import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../theme/app_theme.dart';
import '../services/theme_service.dart';

class EulaScreen extends StatefulWidget {
  /// When true the screen is shown as a gate — no back button, and "I Agree"
  /// stores acceptance then navigates to /home.  When false (opened from
  /// Settings) the screen is informational: a back button is shown and
  /// "I Agree" simply pops.
  final bool blocking;

  /// The authenticated user's ID used to store acceptance.
  /// Required when [blocking] is true.
  final String? userId;

  const EulaScreen({super.key, this.blocking = false, this.userId});

  // ── Persistence helpers ────────────────────────────────────────────────────
  static const _eulaVersion = 'v1';

  static Future<bool> isAccepted(String userId) async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool('eula_accepted_${_eulaVersion}_$userId') ?? false;
  }

  static Future<void> setAccepted(String userId) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('eula_accepted_${_eulaVersion}_$userId', true);
  }

  @override
  State<EulaScreen> createState() => _EulaScreenState();
}

class _EulaScreenState extends State<EulaScreen> {

  static const _clauses = [
    _Clause(
      icon: Icons.school_outlined,
      title: '1. What This App Is',
      body:
          'Bantaba AI is an AI-powered learning tool built to help you think '
          'through data science and machine learning concepts. It uses a small '
          'language model to ask you guiding questions rather than give you '
          'direct answers. This approach is called Socratic tutoring and it is '
          'intentional. The app is not a human tutor and it can make mistakes. '
          'Do not treat anything it says as the final word on a topic.',
    ),
    _Clause(
      icon: Icons.storage_outlined,
      title: '2. What Data Is Stored',
      body:
          'When you use this app, it stores your username, your conversation '
          'history, your quiz results, and your app settings on your device. '
          'In online mode, some of that data is also sent to a secure server '
          'only to generate a response and is not kept after your session ends. '
          'Your data is never sold or shared with third parties.',
    ),
    _Clause(
      icon: Icons.delete_outline_rounded,
      title: '3. Your Right to Delete Your Data',
      body:
          'You can delete all your stored data at any time by going to Settings '
          'and selecting Clear All Data. Once deleted it cannot be recovered. '
          'If you uninstall the app, all data on your device is permanently '
          'removed.',
    ),
    _Clause(
      icon: Icons.people_outlined,
      title: '4. Shared Devices',
      body:
          'If you share your device with other people, your conversation history '
          'and quiz results may be visible to them unless you clear your data '
          'after each session. We recommend using the Clear All Data option if '
          'your device is shared.',
    ),
    _Clause(
      icon: Icons.psychology_outlined,
      title: '5. How the Tutor Works',
      body:
          'The tutor is designed to ask questions, not give answers. This is on '
          'purpose. If you are expecting the app to solve problems for you '
          'directly, this may not be the right tool for that. The goal is to '
          'help you reason through problems yourself with guidance.',
    ),
    _Clause(
      icon: Icons.code_outlined,
      title: '6. Open Source Attribution',
      body:
          'This app is built using open source tools including the Qwen3-0.6B '
          'model by Alibaba Cloud, llama.cpp, Flutter, and FastAPI. All tools '
          'are used in compliance with their respective licences. Full details '
          'are available in the About screen.',
    ),
  ];

  bool _scrolledToBottom = false;
  final _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(() {
      if (_scrollController.offset >=
          _scrollController.position.maxScrollExtent - 80) {
        if (!_scrolledToBottom) setState(() => _scrolledToBottom = true);
      }
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _onAgree(BuildContext context) async {
    if (widget.blocking && widget.userId != null) {
      await EulaScreen.setAccepted(widget.userId!);
      if (context.mounted) {
        Navigator.of(context).pushReplacementNamed('/home');
      }
    } else {
      Navigator.of(context).pop();
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;

    return PopScope(
      canPop: !widget.blocking,
      child: Scaffold(
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
                    if (!widget.blocking)
                      IconButton(
                        icon: const Icon(Icons.arrow_back_rounded),
                        onPressed: () => Navigator.of(context).pop(),
                      )
                    else
                      const SizedBox(width: 48),
                    const SizedBox(width: 4),
                    Text(
                      'EULA & Privacy Policy',
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
                  controller: _scrollController,
                  padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
                  children: [
                    // Intro card
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        gradient: AppTheme.buttonGradient,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.shield_outlined,
                              color: Colors.white, size: 32),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  'Your Privacy Matters',
                                  style: TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 16,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  'Please read these terms carefully before '
                                  'using Bantaba AI.',
                                  style: TextStyle(
                                    color: Colors.white.withValues(alpha: 0.9),
                                    fontSize: 13,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 24),

                    // Clause cards
                    ..._clauses.map((c) => _buildClauseCard(context, c, isDark)),

                    const SizedBox(height: 16),

                    // Footer
                    Text(
                      'Bantaba AI — Version 1.0 | March 2026\n'
                      'African Leadership University, Rwanda',
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodySmall,
                    ),

                    const SizedBox(height: 8),

                    Text(
                      'By tapping "I Agree" you confirm that you have read '
                      'and understood this agreement.',
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            fontStyle: FontStyle.italic,
                          ),
                    ),

                    const SizedBox(height: 20),

                    // I Agree button
                    SizedBox(
                      width: double.infinity,
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          gradient: AppTheme.buttonGradient,
                          borderRadius: BorderRadius.circular(14),
                        ),
                        child: ElevatedButton(
                          onPressed: () => _onAgree(context),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.transparent,
                            shadowColor: Colors.transparent,
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(14),
                            ),
                          ),
                          child: const Text(
                            'I Agree',
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w700,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
      ),  // PopScope
    );
  }

  Widget _buildClauseCard(BuildContext context, _Clause clause, bool isDark) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
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
                    child: Icon(clause.icon,
                        color: AppTheme.accentOrange, size: 20),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      clause.title,
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w700,
                          ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                clause.body,
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

class _Clause {
  final IconData icon;
  final String title;
  final String body;
  const _Clause({required this.icon, required this.title, required this.body});
}
