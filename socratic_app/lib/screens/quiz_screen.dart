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
  final Map<int, int> _userAnswers = {}; // index: selectedOptionIndex
  bool _isFinished = false;

  @override
  void initState() {
    super.initState();
    _loadQuiz();
  }

  Future<void> _loadQuiz() async {
    await _quizService.loadQuizData();
    if (mounted) {
      setState(() {
        _quizQuestions = _quizService.getRandomQuestions(5);
      });
    }
  }

  void _handleOptionSelected(int optionIndex) {
    setState(() {
      _userAnswers[_currentIndex] = optionIndex;
    });
  }

  void _nextQuestion() {
    if (_currentIndex < _quizQuestions.length - 1) {
      setState(() {
        _currentIndex++;
      });
    } else {
      setState(() {
        _isFinished = true;
      });
    }
  }

  void _previousQuestion() {
    if (_currentIndex > 0) {
      setState(() {
        _currentIndex--;
      });
    }
  }

  void _showHelp(QuizQuestion question) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ChatScreen(
          initialTopic: question.topic,
          initialMessage: "I'm working on a quiz question: '${question.question}'. Can you help me understand the concept behind this without giving the answer?",
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    if (_quizQuestions.isEmpty) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Practice Quiz'),
          backgroundColor: Colors.transparent,
        ),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    if (_isFinished) {
      return _buildSummaryScreen();
    }

    final question = _quizQuestions[_currentIndex];
    final selectedOption = _userAnswers[_currentIndex];

    return Scaffold(
      appBar: AppBar(
        title: Text('ML Practice Quiz (${_currentIndex + 1}/${_quizQuestions.length})'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: isDark ? Colors.white : AppTheme.lightTextPrimary,
      ),
      extendBodyBehindAppBar: true,
      body: Container(
        decoration: BoxDecoration(
          gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                LinearProgressIndicator(
                  value: (_currentIndex + 1) / _quizQuestions.length,
                  backgroundColor: isDark ? Colors.white12 : AppTheme.accentOrange.withValues(alpha: 0.1),
                  valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
                ),
                const SizedBox(height: 15),
                Expanded(
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Text(
                          question.topic,
                          style: const TextStyle(
                            color: AppTheme.accentOrange,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.2,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          question.question,
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w500,
                            color: isDark ? Colors.white : AppTheme.lightTextPrimary,
                          ),
                        ),
                        const SizedBox(height: 30),
                        ...List.generate(
                          question.options.length,
                          (index) => _buildOptionCard(index, question.options[index], selectedOption == index, isDark),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    TextButton.icon(
                      onPressed: _currentIndex > 0 ? _previousQuestion : null,
                      icon: const Icon(Icons.arrow_back),
                      label: const Text('Back'),
                      style: TextButton.styleFrom(
                        foregroundColor: isDark ? Colors.white70 : AppTheme.lightTextSecondary,
                      ),
                    ),
                    ElevatedButton(
                      onPressed: selectedOption != null ? _nextQuestion : null,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.accentOrange,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                      child: Text(_currentIndex < _quizQuestions.length - 1 ? 'Next' : 'Finish'),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Center(
                  child: TextButton.icon(
                    onPressed: () => _showHelp(question),
                    icon: const Icon(Icons.psychology, color: AppTheme.accentOrange),
                    label: const Text('Ask Socratic Tutor for Help', style: TextStyle(color: AppTheme.accentOrange)),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildOptionCard(int index, String text, bool isSelected, bool isDark) {
    final textColor = isDark ? Colors.white : AppTheme.lightTextPrimary;
    final secondaryTextColor = isDark ? Colors.white70 : AppTheme.lightTextSecondary;

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: () => _handleOptionSelected(index),
        borderRadius: BorderRadius.circular(15),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          decoration: BoxDecoration(
            color: isSelected 
                ? AppTheme.accentOrange.withValues(alpha: 0.15) 
                : (isDark ? Colors.white.withValues(alpha: 0.05) : Colors.white),
            border: Border.all(
              color: isSelected ? AppTheme.accentOrange : (isDark ? Colors.white10 : Colors.black12),
              width: 1.5,
            ),
            borderRadius: BorderRadius.circular(15),
            boxShadow: !isDark && !isSelected ? [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.03),
                blurRadius: 10,
                offset: const Offset(0, 4),
              )
            ] : null,
          ),
          child: Row(
            children: [
              Container(
                width: 30,
                height: 30,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isSelected ? AppTheme.accentOrange : Colors.transparent,
                  border: Border.all(
                    color: isSelected ? AppTheme.accentOrange : (isDark ? Colors.white30 : Colors.black26),
                  ),
                ),
                child: Center(
                  child: Text(
                    String.fromCharCode(65 + index),
                    style: TextStyle(
                      color: isSelected ? Colors.white : secondaryTextColor,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 15),
              Expanded(
                child: Text(
                  text,
                  style: TextStyle(
                    color: isSelected ? (isDark ? Colors.white : AppTheme.accentOrange) : textColor,
                    fontSize: 16,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSummaryScreen() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    int score = 0;
    List<QuizQuestion> wrongQuestions = [];

    for (int i = 0; i < _quizQuestions.length; i++) {
      if (_userAnswers[i] == _quizQuestions[i].correctIndex) {
        score++;
      } else {
        wrongQuestions.add(_quizQuestions[i]);
      }
    }

    return Scaffold(
      body: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              children: [
                Expanded(
                  child: SingleChildScrollView(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.emoji_events, size: 80, color: AppTheme.accentOrange),
                        const SizedBox(height: 20),
                        Text(
                          'Quiz Complete!',
                          style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: isDark ? Colors.white : AppTheme.lightTextPrimary,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          'Your Score: $score / ${_quizQuestions.length}',
                          style: const TextStyle(fontSize: 20, color: AppTheme.accentOrange),
                        ),
                        const SizedBox(height: 30),
                        if (wrongQuestions.isNotEmpty) ...[
                          Align(
                            alignment: Alignment.centerLeft,
                            child: Text(
                              'Areas to Review:',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: isDark ? Colors.white70 : AppTheme.lightTextSecondary,
                              ),
                            ),
                          ),
                          const SizedBox(height: 10),
                          ListView.builder(
                            shrinkWrap: true,
                            physics: const NeverScrollableScrollPhysics(),
                            itemCount: wrongQuestions.length,
                            itemBuilder: (context, index) {
                              final q = wrongQuestions[index];
                              return Card(
                                color: isDark ? Colors.white.withValues(alpha: 0.05) : Colors.white,
                                elevation: isDark ? 0 : 2,
                                margin: const EdgeInsets.only(bottom: 10),
                                child: ListTile(
                                  title: Text(
                                    q.question,
                                    style: TextStyle(
                                      color: isDark ? Colors.white : AppTheme.lightTextPrimary,
                                      fontSize: 14,
                                    ),
                                  ),
                                  subtitle: Text(
                                    'Topic: ${q.topic}',
                                    style: const TextStyle(color: AppTheme.accentOrange),
                                  ),
                                  trailing: IconButton(
                                    icon: const Icon(Icons.psychology, color: AppTheme.accentOrange),
                                    onPressed: () => _showHelp(q),
                                    tooltip: 'Get help from Tutor',
                                  ),
                                ),
                              );
                            },
                          ),
                        ] else
                          Padding(
                            padding: const EdgeInsets.symmetric(vertical: 40),
                            child: Center(
                              child: Text(
                                'Perfect Score! You nailed it!',
                                style: TextStyle(
                                  color: isDark ? Colors.white70 : AppTheme.lightTextSecondary,
                                  fontSize: 18,
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () => Navigator.of(context).pop(),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.accentOrange,
                    foregroundColor: Colors.white,
                    minimumSize: const Size(double.infinity, 55),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
                  ),
                  child: const Text('Back to Home', style: TextStyle(fontSize: 18)),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
