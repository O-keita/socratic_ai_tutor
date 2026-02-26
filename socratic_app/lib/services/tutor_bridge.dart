import 'package:flutter/foundation.dart';
import '../models/message.dart';
import '../models/session.dart';
import 'hybrid_tutor_service.dart';
import 'session_service.dart';
import 'tutor_engine.dart';

/// Maximum number of previous messages sent to the engine as context.
/// Must match the backend's history limit (backend/ml/inference_engine.py).
const int _kHistoryLimit = 8;

class TutorBridge {
  final TutorEngine _tutorEngine = HybridTutorService();
  Session? _currentSession;

  // Track whether the engine has been initialized to avoid redundant calls.
  bool _engineInitialized = false;

  TutorBridge();

  /// Explicitly initialize the underlying engine.
  /// Safe to call multiple times — no-ops after first success.
  Future<void> initialize() async {
    if (_engineInitialized) return;
    try {
      await _tutorEngine.initialize();
      _engineInitialized = true;
      debugPrint('TutorBridge: ✅ Engine initialized');
    } catch (e, stack) {
      debugPrint('TutorBridge: ❌ Engine initialization failed: $e');
      debugPrint(stack.toString());
      rethrow;
    }
  }

  /// Start or restore a session, and ensure the engine is initialized first.
  Future<void> startSession({String? id, String? topic}) async {
    // CRITICAL FIX: always initialize before any session work.
    await initialize();

    if (id != null) {
      _currentSession = await SessionService.getSession(id);
    }

    if (_currentSession == null) {
      _currentSession = SessionService.createNewSession(topic: topic);
      await SessionService.saveSession(_currentSession!);
    }
  }

  Session? get currentSession => _currentSession;

  Future<Message> sendMessage(String text) async {
    if (_currentSession == null) {
      await startSession();
    }

    try {
      // 1. Add user message
      final userMessage = Message(
        text: text,
        isUser: true,
        timestamp: DateTime.now(),
      );

      final updatedMessages =
      List<Message>.from(_currentSession!.messages)..add(userMessage);
      _currentSession = _currentSession!.copyWith(
        messages: updatedMessages,
        lastActive: DateTime.now(),
      );

      // 2. Build history window — last _kHistoryLimit messages excluding the
      //    message we just added (the engine appends it itself via the prompt).
      final allHistory = updatedMessages.sublist(0, updatedMessages.length - 1);
      final historyWindow = allHistory.length > _kHistoryLimit
          ? allHistory.sublist(allHistory.length - _kHistoryLimit)
          : allHistory;

      // 3. Generate Socratic response
      final llmResult = await _tutorEngine.generateSocraticResponse(
        text,
        history: historyWindow,
      );

      final assistantMessage = Message(
        text: llmResult.text,
        isUser: false,
        timestamp: DateTime.now(),
        metadata: llmResult.metadata,
      );

      // 4. Persist
      final finalMessages =
      List<Message>.from(_currentSession!.messages)..add(assistantMessage);
      _currentSession = _currentSession!.copyWith(
        messages: finalMessages,
        lastActive: DateTime.now(),
      );

      await SessionService.saveSession(_currentSession!);
      return assistantMessage;

    } catch (e, stack) {
      // CRITICAL FIX: log the full stack trace so you can see the real error.
      debugPrint('TutorBridge: ❌ Error generating response: $e');
      debugPrint(stack.toString());

      // Return a message that surfaces the real error in debug mode,
      // and a clean fallback in release mode.
      final errorText = kDebugMode
          ? '⚠️ Debug error:\n$e'
          : 'I apologize, but I am having trouble processing your question right now. What were we discussing?';

      return Message(
        text: errorText,
        isUser: false,
        timestamp: DateTime.now(),
      );
    }
  }
}