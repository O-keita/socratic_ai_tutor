import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import '../theme/app_theme.dart';
import '../models/message.dart';

class ChatBubble extends StatelessWidget {
  final Message message;
  final VoidCallback? onHintRequested;

  const ChatBubble({
    super.key,
    required this.message,
    this.onHintRequested,
  });

  String _cleanMessage(String text) {
    if (text.contains('<think>')) {
      return text
          .replaceAll('<think>', '> *Thinking:* ')
          .replaceAll('</think>', '\n\n');
    }
    return text;
  }

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final isDark = theme.brightness == Brightness.dark;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        mainAxisAlignment:
            isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (!isUser) ...[
            // Tutor avatar
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                gradient: AppTheme.primaryGradient,
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.psychology, color: Colors.white, size: 18),
            ),
            const SizedBox(width: 8),
          ],

          // Bubble
          Flexible(
            child: Container(
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width * 0.72,
              ),
              child: Column(
                crossAxisAlignment:
                    isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 10),
                    decoration: BoxDecoration(
                      gradient: isUser ? AppTheme.buttonGradient : null,
                      color: isUser ? null : colorScheme.surface,
                      borderRadius: BorderRadius.only(
                        topLeft: const Radius.circular(18),
                        topRight: const Radius.circular(18),
                        bottomLeft: Radius.circular(isUser ? 18 : 4),
                        bottomRight: Radius.circular(isUser ? 4 : 18),
                      ),
                      border: isUser
                          ? null
                          : Border.all(
                              color:
                                  colorScheme.primary.withValues(alpha: 0.15),
                            ),
                      boxShadow: isUser
                          ? [
                              BoxShadow(
                                color: colorScheme.primary
                                    .withValues(alpha: 0.18),
                                blurRadius: 10,
                                offset: const Offset(0, 3),
                              ),
                            ]
                          : isDark
                              ? null
                              : [
                                  BoxShadow(
                                    color: Colors.black.withValues(alpha: 0.04),
                                    blurRadius: 8,
                                    offset: const Offset(0, 3),
                                  ),
                                ],
                    ),
                    child: MarkdownBody(
                      data: _cleanMessage(message.text),
                      styleSheet: MarkdownStyleSheet(
                        p: TextStyle(
                          color:
                              isUser ? Colors.white : colorScheme.onSurface,
                          fontSize: 15,
                          height: 1.5,
                        ),
                        strong: TextStyle(
                          color:
                              isUser ? Colors.white : colorScheme.onSurface,
                          fontWeight: FontWeight.w600,
                          fontSize: 15,
                        ),
                        code: TextStyle(
                          color: isUser
                              ? Colors.white
                              : isDark
                                  ? AppTheme.textPrimary
                                  : AppTheme.lightTextPrimary,
                          backgroundColor: isUser
                              ? Colors.white24
                              : isDark
                                  ? Colors.white10
                                  : Colors.black.withValues(alpha: 0.05),
                          fontFamily: 'monospace',
                          fontSize: 13,
                        ),
                        codeblockDecoration: BoxDecoration(
                          color: isUser
                              ? Colors.white10
                              : isDark
                                  ? Colors.black26
                                  : Colors.black.withValues(alpha: 0.04),
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                    ),
                  ),

                  // Hint button (AI messages only)
                  if (!isUser && onHintRequested != null) ...[
                    const SizedBox(height: 6),
                    GestureDetector(
                      onTap: onHintRequested,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 5),
                        decoration: BoxDecoration(
                          color: colorScheme.primary.withValues(alpha: 0.08),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.lightbulb_outline,
                              size: 13,
                              color:
                                  colorScheme.primary.withValues(alpha: 0.7),
                            ),
                            const SizedBox(width: 5),
                            Text(
                              'Need a hint?',
                              style: TextStyle(
                                color: colorScheme.primary
                                    .withValues(alpha: 0.7),
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],

                  // Timestamp
                  const SizedBox(height: 3),
                  Text(
                    _formatTime(message.timestamp),
                    style: TextStyle(
                      color: isDark
                          ? AppTheme.textMuted
                          : AppTheme.lightTextMuted,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
          ),

          if (isUser) ...[
            const SizedBox(width: 8),
            // User avatar
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: colorScheme.surface,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: colorScheme.onSurface.withValues(alpha: 0.08),
                ),
              ),
              child: Icon(
                Icons.person,
                color: colorScheme.onSurface.withValues(alpha: 0.5),
                size: 18,
              ),
            ),
          ],
        ],
      ),
    );
  }

  String _formatTime(DateTime timestamp) {
    final hour = timestamp.hour.toString().padLeft(2, '0');
    final minute = timestamp.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }
}
