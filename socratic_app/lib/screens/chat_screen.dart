import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../models/message.dart';
import '../services/theme_service.dart';
import '../services/tutor_bridge.dart';
import '../services/hybrid_tutor_service.dart';
import '../widgets/chat_bubble.dart';
import '../widgets/chat_input.dart';

class ChatScreen extends StatefulWidget {
  final String? initialTopic;
  final String? sessionId;
  final String? initialMessage;

  const ChatScreen({
    super.key,
    this.initialTopic,
    this.sessionId,
    this.initialMessage,
  });

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> with TickerProviderStateMixin {
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
    // CRITICAL FIX: initialize() is now called inside startSession(),
    // so the engine is guaranteed to be ready before any message is sent.
    await _tutorBridge.startSession(
      id: widget.sessionId,
      topic: widget.initialTopic,
    );

    if (!mounted) return;
    setState(() {
      if (_tutorBridge.currentSession != null &&
          _tutorBridge.currentSession!.messages.isNotEmpty) {
        _messages.addAll(_tutorBridge.currentSession!.messages);
      } else {
        _messages.add(Message(
          text:
          'Hi! I\'m your Socratic tutor.\n\nAsk me anything and I\'ll guide you to discover the answer through questions.',
          isUser: false,
          timestamp: DateTime.now(),
        ));
      }
    });

    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());

    // CRITICAL FIX: Check initialMessage AFTER setState so _messages is
    // populated. Use the actual welcome-message count (1) as the threshold,
    // not the total message count which may vary when loading history.
    if (widget.initialMessage != null) {
      final hasOnlyWelcome =
          _messages.length == 1 && !_messages.first.isUser;
      if (hasOnlyWelcome) {
        _autoSendMsg(widget.initialMessage!);
      }
    }
  }

  void _autoSendMsg(String text) async {
    await Future.delayed(const Duration(milliseconds: 500));
    if (!mounted) return;
    _controller.text = text;
    _handleSend();
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

    if (!mounted) return;
    setState(() {
      _messages.add(Message(text: text, isUser: true, timestamp: DateTime.now()));
      _isLoading = true;
    });
    _controller.clear();

    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());

    final response = await _tutorBridge.sendMessage(text);

    if (!mounted) return;
    setState(() {
      _messages.add(response);
      _isLoading = false;
    });

    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
  }

  /// CRITICAL FIX: Hint now goes through the real LLM via TutorBridge
  /// instead of returning a hardcoded static string.
  void _requestHint() async {
    if (_isLoading) return;

    if (!mounted) return;
    setState(() => _isLoading = true);

    // Build a meta-prompt that asks the tutor to guide without revealing.
    const hintPrompt =
        '[The student is stuck. Give a Socratic hint — ask one guiding question to help them think through the last topic, without revealing the answer.]';

    final hint = await _tutorBridge.sendMessage(hintPrompt);

    if (!mounted) return;
    setState(() {
      _messages.add(hint);
      _isLoading = false;
    });

    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToBottom());
  }

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
              _buildAppBar(),
              Expanded(
                child: ListView.builder(
                  controller: _scrollController,
                  padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  itemCount: _messages.length + (_isLoading ? 1 : 0),
                  itemBuilder: (context, index) {
                    if (index == _messages.length && _isLoading) {
                      return _buildTypingIndicator();
                    }
                    return ChatBubble(
                      message: _messages[index],
                      onHintRequested:
                      !_messages[index].isUser ? _requestHint : null,
                    );
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
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final isDark = context.watch<ThemeService>().isDarkMode;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_new, size: 20),
            onPressed: () => Navigator.pop(context),
            color: colorScheme.onSurface,
          ),
          const SizedBox(width: 8),
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
          Flexible(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Flexible(
                      child: Text(
                        'Socratic Tutor',
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: colorScheme.onSurface,
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    StreamBuilder<EngineStatus>(
                      stream: HybridTutorService().statusStream,
                      initialData: HybridTutorService().currentStatus,
                      builder: (context, snapshot) {
                        final status =
                            snapshot.data ?? EngineStatus.offline;
                        return Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: status == EngineStatus.online
                                ? Colors.green
                                : (status == EngineStatus.connecting
                                ? Colors.orange
                                : Colors.grey),
                          ),
                        );
                      },
                    ),
                  ],
                ),
                if (widget.initialTopic != null)
                  Text(
                    widget.initialTopic!,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: isDark
                          ? AppTheme.textSecondary
                          : AppTheme.lightTextSecondary,
                      fontSize: 10,
                    ),
                  ),
              ],
            ),
          ),
          const Spacer(),
          IconButton(
            icon: Icon(isDark ? Icons.light_mode : Icons.dark_mode),
            onPressed: () => context.read<ThemeService>().toggleTheme(),
            color: colorScheme.onSurface,
          ),
          TextButton.icon(
            onPressed: _isLoading ? null : _requestHint,
            icon: const Icon(Icons.lightbulb_outline, size: 18),
            label: const Text('Hint'),
            style: TextButton.styleFrom(
              foregroundColor: colorScheme.primary,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTypingIndicator() {
    final theme = Theme.of(context);

    return Padding(
      padding: const EdgeInsets.only(left: 8, top: 8, bottom: 8),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: theme.colorScheme.surface,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (index) {
                return AnimatedBuilder(
                  animation: _typingController,
                  builder: (context, child) {
                    return Container(
                      margin: EdgeInsets.only(left: index == 0 ? 0 : 4),
                      child: Transform.translate(
                        offset: Offset(
                          0,
                          -4 *
                              (_typingController.value - 0.5).abs() *
                              (index == 1 ? 1 : 0.6),
                        ),
                        child: Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: theme.colorScheme.primary.withValues(
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