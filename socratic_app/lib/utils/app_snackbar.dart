import 'package:flutter/material.dart';

/// Centralized, consistently styled toast notifications.
///
/// Usage:
///   AppSnackBar.error(context, 'Login failed');
///   AppSnackBar.success(context, 'Lesson completed!');
///   AppSnackBar.warning(context, 'No internet connection');
///   AppSnackBar.info(context, 'Voice input coming soon');
class AppSnackBar {
  // ── Internal dispatcher ──────────────────────────────────────────────────

  static void _show(
    BuildContext context,
    String message,
    _SnackType type, {
    Duration? duration,
    String? actionLabel,
    VoidCallback? onAction,
  }) {
    Color bg;
    IconData icon;

    switch (type) {
      case _SnackType.error:
        bg = const Color(0xFFEF4444);
        icon = Icons.error_outline_rounded;
        duration ??= const Duration(seconds: 4);
      case _SnackType.warning:
        bg = const Color(0xFFF59E0B);
        icon = Icons.warning_amber_rounded;
        duration ??= const Duration(seconds: 4);
      case _SnackType.success:
        bg = const Color(0xFF10B981);
        icon = Icons.check_circle_outline_rounded;
        duration ??= const Duration(seconds: 2);
      case _SnackType.info:
        bg = const Color(0xFF6366F1);
        icon = Icons.info_outline_rounded;
        duration ??= const Duration(seconds: 3);
    }

    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          content: Row(
            children: [
              Icon(icon, color: Colors.white, size: 20),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  message,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13.5,
                    fontWeight: FontWeight.w500,
                    height: 1.35,
                  ),
                ),
              ),
            ],
          ),
          backgroundColor: bg,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          duration: duration,
          action: actionLabel != null && onAction != null
              ? SnackBarAction(
                  label: actionLabel,
                  textColor: Colors.white.withValues(alpha: 0.88),
                  onPressed: onAction,
                )
              : null,
        ),
      );
  }

  // ── Public API ───────────────────────────────────────────────────────────

  /// Red toast — for failures, exceptions, and blocking errors.
  static void error(
    BuildContext context,
    String message, {
    String? actionLabel,
    VoidCallback? onAction,
  }) =>
      _show(context, message, _SnackType.error,
          actionLabel: actionLabel, onAction: onAction);

  /// Amber toast — for non-fatal warnings (network, limits, etc.).
  static void warning(
    BuildContext context,
    String message, {
    String? actionLabel,
    VoidCallback? onAction,
  }) =>
      _show(context, message, _SnackType.warning,
          actionLabel: actionLabel, onAction: onAction);

  /// Green toast — for successful operations. Auto-dismisses in 2 s.
  static void success(BuildContext context, String message) =>
      _show(context, message, _SnackType.success);

  /// Indigo toast — for neutral informational messages.
  static void info(BuildContext context, String message) =>
      _show(context, message, _SnackType.info);
}

enum _SnackType { error, warning, success, info }
