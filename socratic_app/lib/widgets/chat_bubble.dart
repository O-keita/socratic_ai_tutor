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
    // Optionally hide thinking blocks or format them as blockquotes
    if (text.contains('<think>')) {
      return text.replaceAll('<think>', '> *Thinking:* ').replaceAll('</think>', '\n\n');
    }
    return text;
  }

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[
            // Tutor Avatar
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                gradient: AppTheme.primaryGradient,
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(
                Icons.psychology,
                color: Colors.white,
                size: 20,
              ),
            ),
            const SizedBox(width: 12),
          ],
          
          // Message Bubble
          Flexible(
            child: Container(
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width * 0.75,
              ),
              child: Column(
                crossAxisAlignment: isUser 
                  ? CrossAxisAlignment.end 
                  : CrossAxisAlignment.start,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 12,
                    ),
                    decoration: BoxDecoration(
                      gradient: isUser 
                        ? AppTheme.primaryGradient 
                        : null,
                      color: isUser 
                        ? null 
                        : colorScheme.surface,
                      borderRadius: BorderRadius.only(
                        topLeft: const Radius.circular(20),
                        topRight: const Radius.circular(20),
                        bottomLeft: Radius.circular(isUser ? 20 : 4),
                        bottomRight: Radius.circular(isUser ? 4 : 20),
                      ),
                      border: isUser 
                        ? null 
                        : Border.all(
                            color: colorScheme.primary.withOpacity(0.2),
                          ),
                      boxShadow: isUser
                        ? [
                            BoxShadow(
                              color: colorScheme.primary.withOpacity(0.2),
                              blurRadius: 12,
                              offset: const Offset(0, 4),
                            ),
                          ]
                        : [
                            if (theme.brightness == Brightness.light)
                              BoxShadow(
                                color: Colors.black.withOpacity(0.05),
                                blurRadius: 10,
                                offset: const Offset(0, 4),
                              ),
                          ],
                    ),
                    child: MarkdownBody(
                      data: _cleanMessage(message.text),
                      styleSheet: MarkdownStyleSheet(
                        p: TextStyle(
                          color: isUser ? Colors.white : colorScheme.onSurface,
                          fontSize: 18,
                          height: 1.4,
                        ),
                        code: TextStyle(
                          color: isUser ? Colors.white : (theme.brightness == Brightness.dark ? AppTheme.textPrimary : AppTheme.lightTextPrimary),
                          backgroundColor: isUser ? Colors.white24 : (theme.brightness == Brightness.dark ? Colors.white10 : Colors.black.withOpacity(0.05)),
                          fontFamily: 'monospace',
                        ),
                        codeblockDecoration: BoxDecoration(
                          color: isUser ? Colors.white10 : (theme.brightness == Brightness.dark ? Colors.black26 : Colors.black.withOpacity(0.05)),
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                    ),
                  ),
                  
                  // Hint button for tutor messages
                  if (!isUser && onHintRequested != null) ...[
                    const SizedBox(height: 8),
                    GestureDetector(
                      onTap: onHintRequested,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: colorScheme.primary.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.lightbulb_outline,
                              size: 14,
                              color: colorScheme.primary.withOpacity(0.8),
                            ),
                            const SizedBox(width: 6),
                            Text(
                              'Need a hint?',
                              style: TextStyle(
                                color: colorScheme.primary.withOpacity(0.8),
                                fontSize: 14,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                  
                  // Timestamp
                  const SizedBox(height: 4),
                  Text(
                    _formatTime(message.timestamp),
                    style: TextStyle(
                      color: theme.brightness == Brightness.dark 
                        ? AppTheme.textMuted 
                        : AppTheme.lightTextMuted,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          if (isUser) ...[
            const SizedBox(width: 12),
            // User Avatar
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: colorScheme.surface,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: colorScheme.onSurface.withOpacity(0.1),
                ),
              ),
              child: Icon(
                Icons.person,
                color: colorScheme.onSurface.withOpacity(0.6),
                size: 20,
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
