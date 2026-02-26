import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class ChatInput extends StatefulWidget {
  final TextEditingController controller;
  final VoidCallback onSend;
  final bool isLoading;

  const ChatInput({
    super.key,
    required this.controller,
    required this.onSend,
    this.isLoading = false,
  });

  @override
  State<ChatInput> createState() => _ChatInputState();
}

class _ChatInputState extends State<ChatInput> {
  bool _hasText = false;

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onTextChanged);
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onTextChanged);
    super.dispose();
  }

  void _onTextChanged() {
    final hasText = widget.controller.text.trim().isNotEmpty;
    if (hasText != _hasText) setState(() => _hasText = hasText);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final isDark = theme.brightness == Brightness.dark;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: isDark ? AppTheme.surfaceDark : AppTheme.lightSurface,
        border: Border(
          top: BorderSide(
            color: colorScheme.onSurface.withValues(alpha: 0.06),
          ),
        ),
      ),
      child: SafeArea(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            // Text input
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: isDark ? AppTheme.surfaceCard : AppTheme.lightSurfaceCard,
                  borderRadius: BorderRadius.circular(22),
                  border: Border.all(
                    color: _hasText
                        ? colorScheme.primary.withValues(alpha: 0.25)
                        : colorScheme.onSurface.withValues(alpha: 0.06),
                  ),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Expanded(
                      child: TextField(
                        controller: widget.controller,
                        style: TextStyle(
                          color: colorScheme.onSurface,
                          fontSize: 15,
                          height: 1.4,
                        ),
                        maxLines: 5,
                        minLines: 1,
                        textInputAction: TextInputAction.send,
                        onSubmitted: (_) {
                          if (!widget.isLoading && _hasText) widget.onSend();
                        },
                        decoration: InputDecoration(
                          hintText: 'Ask your question...',
                          hintStyle: TextStyle(
                            color:
                                isDark ? AppTheme.textMuted : AppTheme.lightTextMuted,
                            fontSize: 15,
                          ),
                          border: InputBorder.none,
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 10,
                          ),
                        ),
                      ),
                    ),

                    // Mic button
                    Padding(
                      padding: const EdgeInsets.only(right: 4, bottom: 2),
                      child: IconButton(
                        icon: Icon(
                          Icons.mic_outlined,
                          color: isDark
                              ? AppTheme.textMuted
                              : AppTheme.lightTextMuted,
                          size: 20,
                        ),
                        onPressed: () {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: const Text('Voice input coming soon!'),
                              behavior: SnackBarBehavior.floating,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(width: 8),

            // Send button
            AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                gradient: _hasText && !widget.isLoading
                    ? AppTheme.buttonGradient
                    : null,
                color: _hasText && !widget.isLoading
                    ? null
                    : isDark
                        ? AppTheme.surfaceCard
                        : AppTheme.lightSurfaceCard,
                borderRadius: BorderRadius.circular(14),
                boxShadow: _hasText && !widget.isLoading
                    ? [
                        BoxShadow(
                          color: AppTheme.accentOrange.withValues(alpha: 0.35),
                          blurRadius: 12,
                          offset: const Offset(0, 4),
                        ),
                      ]
                    : null,
              ),
              child: Material(
                color: Colors.transparent,
                child: InkWell(
                  onTap: widget.isLoading || !_hasText ? null : widget.onSend,
                  borderRadius: BorderRadius.circular(14),
                  child: Center(
                    child: widget.isLoading
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              valueColor: AlwaysStoppedAnimation<Color>(
                                  AppTheme.accentOrange),
                            ),
                          )
                        : Icon(
                            Icons.send_rounded,
                            color: _hasText ? Colors.white : AppTheme.textMuted,
                            size: 20,
                          ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
