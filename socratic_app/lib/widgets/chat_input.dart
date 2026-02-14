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
    if (hasText != _hasText) {
      setState(() {
        _hasText = hasText;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final isDark = theme.brightness == Brightness.dark;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark ? AppTheme.surfaceDark : AppTheme.lightSurface,
        border: Border(
          top: BorderSide(
            color: colorScheme.onSurface.withOpacity(0.05),
          ),
        ),
      ),
      child: SafeArea(
        child: Row(
          children: [
            // Attachment Button
            Container(
              decoration: BoxDecoration(
                color: isDark ? AppTheme.surfaceCard : AppTheme.lightSurfaceCard,
                borderRadius: BorderRadius.circular(12),
              ),
              child: IconButton(
                icon: Icon(
                  Icons.add, 
                  color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary
                ),
                onPressed: () => _showAttachmentOptions(context),
              ),
            ),
            
            const SizedBox(width: 12),
            
            // Text Input
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: isDark ? AppTheme.surfaceCard : AppTheme.lightSurfaceCard,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(
                    color: colorScheme.primary.withOpacity(0.1),
                  ),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: widget.controller,
                        style: TextStyle(
                          color: colorScheme.onSurface,
                          fontSize: 18,
                        ),
                        maxLines: 4,
                        minLines: 1,
                        textInputAction: TextInputAction.send,
                        onSubmitted: (_) {
                          if (!widget.isLoading && _hasText) {
                            widget.onSend();
                          }
                        },
                        decoration: InputDecoration(
                          hintText: 'Share your thoughts...',
                          hintStyle: TextStyle(
                            color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted,
                            fontSize: 18,
                          ),
                          border: InputBorder.none,
                          contentPadding: const EdgeInsets.symmetric(
                            horizontal: 20,
                            vertical: 12,
                          ),
                        ),
                      ),
                    ),
                    
                    // Voice Button
                    IconButton(
                      icon: Icon(
                        Icons.mic_outlined,
                        color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                      ),
                      onPressed: () {
                        // TODO: Implement voice input
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
                  ],
                ),
              ),
            ),
            
            const SizedBox(width: 12),
            
            // Send Button
            AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              decoration: BoxDecoration(
                gradient: _hasText && !widget.isLoading
                  ? AppTheme.primaryGradient
                  : null,
                color: _hasText && !widget.isLoading
                  ? null
                  : isDark ? AppTheme.surfaceCard : AppTheme.lightSurfaceCard,
                borderRadius: BorderRadius.circular(16),
                boxShadow: _hasText && !widget.isLoading
                  ? [
                      BoxShadow(
                        color: AppTheme.accentOrange.withOpacity(0.4),
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
                  borderRadius: BorderRadius.circular(16),
                  child: Container(
                    width: 48,
                    height: 48,
                    alignment: Alignment.center,
                    child: widget.isLoading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(
                              AppTheme.accentOrange,
                            ),
                          ),
                        )
                      : Icon(
                          Icons.send_rounded,
                          color: _hasText
                            ? Colors.white
                            : AppTheme.textMuted,
                          size: 22,
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

  void _showAttachmentOptions(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppTheme.surfaceCard,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Container(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: AppTheme.textMuted,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildAttachmentOption(
                  context,
                  Icons.image_outlined,
                  'Image',
                  const Color(0xFF3B82F6),
                ),
                _buildAttachmentOption(
                  context,
                  Icons.code,
                  'Code',
                  const Color(0xFF10B981),
                ),
                _buildAttachmentOption(
                  context,
                  Icons.description_outlined,
                  'File',
                  const Color(0xFFF59E0B),
                ),
                _buildAttachmentOption(
                  context,
                  Icons.link,
                  'Link',
                  AppTheme.accentOrange,
                ),
              ],
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildAttachmentOption(
    BuildContext context,
    IconData icon,
    String label,
    Color color,
  ) {
    return GestureDetector(
      onTap: () {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('$label attachment coming soon!'),
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(10),
            ),
          ),
        );
      },
      child: Column(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Icon(icon, color: color, size: 26),
          ),
          const SizedBox(height: 8),
          Text(
            label,
            style: const TextStyle(
              color: AppTheme.textSecondary,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}
