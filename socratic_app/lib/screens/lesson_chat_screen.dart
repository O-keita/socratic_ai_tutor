import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../models/message.dart';
import '../models/course.dart';
import '../services/tutor_bridge.dart';
import '../services/hybrid_tutor_service.dart';
import '../services/theme_service.dart';
import '../widgets/chat_bubble.dart';
import '../widgets/chat_input.dart';

class LessonChatScreen extends StatefulWidget {
  final Lesson lesson;
  final Course course;

  const LessonChatScreen({
    super.key,
    required this.lesson,
    required this.course,
  });

  @override
  State<LessonChatScreen> createState() => _LessonChatScreenState();
}

class _LessonChatScreenState extends State<LessonChatScreen> with TickerProviderStateMixin {
  final List<Message> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final TutorBridge _tutorBridge = TutorBridge();
  bool _isLoading = false;
  late AnimationController _typingController;

  @override
  void initState() {
    super.initState();
    _typingController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    )..repeat(reverse: true);
    
    _initSession();
  }

  Future<void> _initSession() async {
    await _tutorBridge.initialize();
    
    // Start session with lesson context
    final context = '''
You are helping a student learn about "${widget.lesson.title}" from the course "${widget.course.title}".

Lesson content summary:
${widget.lesson.keyPoints ?? 'This lesson covers ${widget.lesson.title}'}

${widget.lesson.reflectionQuestions != null ? 'Key reflection questions:\n${widget.lesson.reflectionQuestions!.join('\n')}' : ''}

Use the Socratic method to guide the student. Ask questions to help them discover understanding rather than giving direct answers.
''';
    
    await _tutorBridge.startSession(topic: context);
    
    setState(() {
      _messages.add(Message(
        text: 'Hi! I\'m here to help you understand "${widget.lesson.title}".\n\nWhat questions do you have about this lesson?',
        isUser: false,
        timestamp: DateTime.now(),
      ));
    });
  }

  @override
  void dispose() {
    _typingController.stop();
    _typingController.dispose();
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    if (_scrollController.hasClients) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    }
  }

  void _handleSend() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _isLoading) return;

    setState(() {
      _messages.add(Message(text: text, isUser: true, timestamp: DateTime.now()));
      _isLoading = true;
    });
    _controller.clear();
    
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());

    final response = await _tutorBridge.sendMessage(text);

    setState(() {
      _messages.add(response);
      _isLoading = false;
    });
    
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
  }

  void _askAboutTopic(String question) {
    _controller.text = question;
    _handleSend();
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
          child: Column(
            children: [
              _buildAppBar(),
              if (_messages.length <= 1) _buildSuggestions(),
              Expanded(
                child: ListView.builder(
                  controller: _scrollController,
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  itemCount: _messages.length + (_isLoading ? 1 : 0),
                  itemBuilder: (context, index) {
                    if (index == _messages.length && _isLoading) {
                      return _buildTypingIndicator();
                    }
                    return ChatBubble(message: _messages[index]);
                  },
                ),
              ),
              ChatInput(
                controller: _controller,
                onSend: _handleSend,
                isLoading: _isLoading,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAppBar() {
    final colorScheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final hybridService = HybridTutorService();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_new, size: 20),
            onPressed: () => Navigator.pop(context),
            color: colorScheme.onSurface,
          ),
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              gradient: AppTheme.primaryGradient,
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(
              Icons.psychology,
              color: Colors.white,
              size: 22,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      'AI Tutor',
                      style: TextStyle(
                        color: colorScheme.onSurface,
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(width: 8),
                    StreamBuilder<EngineStatus>(
                      stream: hybridService.statusStream,
                      initialData: hybridService.currentStatus,
                      builder: (context, snapshot) {
                        final status = snapshot.data ?? EngineStatus.offline;
                        return Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: status == EngineStatus.online 
                                ? Colors.green 
                                : (status == EngineStatus.connecting ? Colors.orange : Colors.grey),
                            boxShadow: [
                              if (status != EngineStatus.offline)
                                BoxShadow(
                                  color: (status == EngineStatus.online ? Colors.green : Colors.orange).withValues(alpha: 0.5),
                                  blurRadius: 4,
                                  spreadRadius: 1,
                                ),
                            ],
                          ),
                        );
                      },
                    ),
                  ],
                ),
                Text(
                  widget.lesson.title,
                  style: TextStyle(
                    color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                    fontSize: 12,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          IconButton(
            icon: Icon(isDark ? Icons.light_mode : Icons.dark_mode),
            onPressed: () => context.read<ThemeService>().toggleTheme(),
            color: colorScheme.onSurface,
          ),
        ],
      ),
    );
  }

  Widget _buildSuggestions() {
    final colorScheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    final questions = widget.lesson.reflectionQuestions ?? [
      'Can you explain this concept?',
      'Why is this important?',
      'How does this relate to other topics?',
    ];
    
    return Container(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Try asking:',
            style: TextStyle(
              color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: questions.take(3).map((q) {
              return GestureDetector(
                onTap: () => _askAboutTopic(q),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: colorScheme.surface,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: isDark ? AppTheme.primaryLight.withValues(alpha: 0.5) : Colors.grey.withValues(alpha: 0.2),
                    ),
                  ),
                  child: Text(
                    q.length > 40 ? '${q.substring(0, 40)}...' : q,
                    style: TextStyle(
                      color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                      fontSize: 12,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildTypingIndicator() {
    final colorScheme = Theme.of(context).colorScheme;

    return Padding(
      padding: const EdgeInsets.only(left: 8, top: 8, bottom: 8),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: colorScheme.surface,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (index) {
                return AnimatedBuilder(
                  animation: _typingController,
                  builder: (context, child) {
                    final isDarkInner = Theme.of(context).brightness == Brightness.dark;
                    return Container(
                      margin: EdgeInsets.only(left: index == 0 ? 0 : 4),
                      child: Transform.translate(
                        offset: Offset(
                          0,
                          -4 * (_typingController.value - 0.5).abs() * 
                            (index == 1 ? 1 : 0.6),
                        ),
                        child: Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: (isDarkInner ? AppTheme.textMuted : AppTheme.lightTextMuted).withValues(
                              alpha: 0.5 + (_typingController.value * 0.5),
                            ),
                            shape: BoxShape.circle,
                          ),
                        ),
                      ),
                    );
                  },
                );
              }),
            ),
          ),
        ],
      ),
    );
  }
}
