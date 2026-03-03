import 'package:flutter/material.dart';
import '../models/quiz_question.dart';
import '../services/quiz_service.dart';
import '../theme/app_theme.dart';
import 'chat_screen.dart';

class QuizScreen extends StatefulWidget {
  const QuizScreen({super.key});

  @override
  State<QuizScreen> createState() => _QuizScreenState();
}

class _QuizScreenState extends State<QuizScreen> {
  final QuizService _quizService = QuizService();
  List<QuizQuestion> _quizQuestions = [];
  int _currentIndex = 0;
  final Map<int, int> _userAnswers = {};
  bool _isFinished = false;

  @override
  void initState() {
    super.initState();
    _loadQuiz();
  }

  Future<void> _loadQuiz() async {
    await _quizService.loadQuizData();
    if (mounted) setState(() { _quizQuestions = _quizService.getRandomQuestions(5); });
  }

  void _handleOptionSelected(int optionIndex) => setState(() { _userAnswers[_currentIndex] = optionIndex; });

  void _nextQuestion() {
    if (_currentIndex < _quizQuestions.length - 1) {
      setState(() => _currentIndex++);
    } else {
      setState(() => _isFinished = true);
    }
  }

  void _previousQuestion() { if (_currentIndex > 0) setState(() => _currentIndex--); }

  void _showHelp(QuizQuestion question) {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => ChatScreen(initialTopic: question.topic, initialMessage: "I'm working on a quiz question: '${question.question}'. Can you help me understand the concept behind this without giving the answer?"),
    ));
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? AppTheme.primaryDark : const Color(0xFFFEF6EE);

    if (_quizQuestions.isEmpty) {
      return Scaffold(
        body: Container(color: bgColor, child: const Center(child: CircularProgressIndicator(color: AppTheme.accentOrange))),
      );
    }

    if (_isFinished) return _buildSummaryScreen(isDark, bgColor);

    final question = _quizQuestions[_currentIndex];
    final selectedOption = _userAnswers[_currentIndex];

    return Scaffold(
      body: Container(
        color: bgColor,
        child: SafeArea(
          child: Column(
            children: [
              // ── App bar ──────────────────────────
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                child: Row(
                  children: [
                    IconButton(
                      icon: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(color: isDark ? AppTheme.surfaceCard : Colors.white, shape: BoxShape.circle, boxShadow: isDark ? null : [BoxShadow(color: Colors.black.withValues(alpha: 0.06), blurRadius: 8, offset: const Offset(0, 2))]),
                        child: Icon(Icons.arrow_back_ios_new, size: 16, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary),
                      ),
                      onPressed: () => Navigator.pop(context),
                    ),
                    Expanded(
                      child: Text('ML Practice Quiz', textAlign: TextAlign.center, style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
                    ),
                    const SizedBox(width: 48),
                  ],
                ),
              ),

              // ── Progress bar ─────────────────────
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: LinearProgressIndicator(
                    value: (_currentIndex + 1) / _quizQuestions.length,
                    backgroundColor: isDark ? Colors.white12 : AppTheme.accentOrange.withValues(alpha: 0.1),
                    valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
                    minHeight: 6,
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 6),
                child: Align(
                  alignment: Alignment.centerRight,
                  child: Text('${_currentIndex + 1}/${_quizQuestions.length}', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: AppTheme.accentOrange)),
                ),
              ),

              // ── Question body ────────────────────
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(20, 10, 20, 0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(color: AppTheme.accentOrange.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(10)),
                        child: Text(question.topic, style: TextStyle(color: AppTheme.accentOrange, fontWeight: FontWeight.w700, fontSize: 12, letterSpacing: 0.5)),
                      ),
                      const SizedBox(height: 16),
                      Text(question.question, style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary, height: 1.4)),
                      const SizedBox(height: 24),
                      ...List.generate(question.options.length, (i) => _buildOptionCard(i, question.options[i], selectedOption == i, isDark)),
                    ],
                  ),
                ),
              ),

              // ── Bottom controls ──────────────────
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 16),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        TextButton.icon(
                          onPressed: _currentIndex > 0 ? _previousQuestion : null,
                          icon: const Icon(Icons.arrow_back_rounded, size: 18),
                          label: const Text('Back'),
                          style: TextButton.styleFrom(foregroundColor: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary),
                        ),
                        Container(
                          decoration: BoxDecoration(
                            gradient: selectedOption != null ? AppTheme.buttonGradient : null,
                            color: selectedOption == null ? Colors.grey.withValues(alpha: 0.3) : null,
                            borderRadius: BorderRadius.circular(18),
                          ),
                          child: ElevatedButton(
                            onPressed: selectedOption != null ? _nextQuestion : null,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.transparent,
                              shadowColor: Colors.transparent,
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                            ),
                            child: Text(_currentIndex < _quizQuestions.length - 1 ? 'Next' : 'Finish', style: const TextStyle(fontWeight: FontWeight.w600)),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    TextButton.icon(
                      onPressed: () => _showHelp(question),
                      icon: const Icon(Icons.psychology_rounded, color: AppTheme.accentOrange, size: 18),
                      label: const Text('Ask Bantaba AI for Help', style: TextStyle(color: AppTheme.accentOrange, fontWeight: FontWeight.w600, fontSize: 13)),
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

  Widget _buildOptionCard(int index, String text, bool isSelected, bool isDark) {
    final cardBg = isDark ? AppTheme.surfaceCard : Colors.white;

    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: InkWell(
        onTap: () => _handleOptionSelected(index),
        borderRadius: BorderRadius.circular(18),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
          decoration: BoxDecoration(
            color: isSelected ? AppTheme.accentOrange.withValues(alpha: 0.12) : cardBg,
            border: Border.all(color: isSelected ? AppTheme.accentOrange : (isDark ? AppTheme.primaryLight.withValues(alpha: 0.2) : Colors.grey.withValues(alpha: 0.12)), width: 1.5),
            borderRadius: BorderRadius.circular(18),
            boxShadow: isDark ? null : [BoxShadow(color: Colors.black.withValues(alpha: 0.03), blurRadius: 8, offset: const Offset(0, 2))],
          ),
          child: Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isSelected ? AppTheme.accentOrange : Colors.transparent,
                  border: Border.all(color: isSelected ? AppTheme.accentOrange : (isDark ? AppTheme.textMuted : Colors.grey.withValues(alpha: 0.4)), width: 1.5),
                ),
                child: Center(child: Text(String.fromCharCode(65 + index), style: TextStyle(color: isSelected ? Colors.white : (isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary), fontWeight: FontWeight.w700, fontSize: 14))),
              ),
              const SizedBox(width: 14),
              Expanded(child: Text(text, style: TextStyle(color: isSelected ? AppTheme.accentOrange : (isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary), fontSize: 15, fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400))),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSummaryScreen(bool isDark, Color bgColor) {
    int score = 0;
    List<QuizQuestion> wrongQuestions = [];
    for (int i = 0; i < _quizQuestions.length; i++) {
      if (_userAnswers[i] == _quizQuestions[i].correctIndex) { score++; } else { wrongQuestions.add(_quizQuestions[i]); }
    }

    return Scaffold(
      body: Container(
        width: double.infinity,
        color: bgColor,
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                Expanded(
                  child: SingleChildScrollView(
                    child: Column(
                      children: [
                        const SizedBox(height: 20),
                        Container(
                          width: 90,
                          height: 90,
                          decoration: BoxDecoration(
                            color: AppTheme.accentOrange.withValues(alpha: 0.12),
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.emoji_events_rounded, size: 48, color: AppTheme.accentOrange),
                        ),
                        const SizedBox(height: 20),
                        Text('Quiz Complete!', style: TextStyle(fontSize: 28, fontWeight: FontWeight.w800, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
                        const SizedBox(height: 10),
                        Text('Your Score: $score / ${_quizQuestions.length}', style: const TextStyle(fontSize: 20, color: AppTheme.accentOrange, fontWeight: FontWeight.w600)),
                        const SizedBox(height: 30),
                        if (wrongQuestions.isNotEmpty) ...[
                          Align(
                            alignment: Alignment.centerLeft,
                            child: Text('Areas to Review:', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
                          ),
                          const SizedBox(height: 12),
                          ...wrongQuestions.map((q) => Container(
                            margin: const EdgeInsets.only(bottom: 10),
                            padding: const EdgeInsets.all(14),
                            decoration: BoxDecoration(
                              color: isDark ? AppTheme.surfaceCard : Colors.white,
                              borderRadius: BorderRadius.circular(18),
                              boxShadow: isDark ? null : [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 8, offset: const Offset(0, 2))],
                            ),
                            child: Row(
                              children: [
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(q.question, style: TextStyle(fontSize: 14, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary, fontWeight: FontWeight.w500)),
                                      const SizedBox(height: 4),
                                      Text(q.topic, style: const TextStyle(fontSize: 12, color: AppTheme.accentOrange, fontWeight: FontWeight.w600)),
                                    ],
                                  ),
                                ),
                                IconButton(icon: const Icon(Icons.psychology_rounded, color: AppTheme.accentOrange), onPressed: () => _showHelp(q)),
                              ],
                            ),
                          )),
                        ] else
                          Padding(
                            padding: const EdgeInsets.symmetric(vertical: 40),
                            child: Text('Perfect Score! You nailed it!', style: TextStyle(color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary, fontSize: 18, fontWeight: FontWeight.w600)),
                          ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: Container(
                    decoration: BoxDecoration(gradient: AppTheme.buttonGradient, borderRadius: BorderRadius.circular(18)),
                    child: ElevatedButton(
                      onPressed: () => Navigator.of(context).pop(),
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.transparent, shadowColor: Colors.transparent, minimumSize: const Size(double.infinity, 52), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18))),
                      child: const Text('Back to Home', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Colors.white)),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
