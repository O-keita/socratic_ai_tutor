import '../models/message.dart';
import '../models/session.dart';
import 'hybrid_tutor_service.dart';
import 'session_service.dart';
import '../models/message.dart' as model;
import 'tutor_engine.dart';

class TutorBridge {
  final TutorEngine _tutorEngine = HybridTutorService();
  Session? _currentSession;

  TutorBridge();

  Future<void> initialize() async {
    await _tutorEngine.initialize();
  }

  Future<void> startSession({String? id, String? topic}) async {
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
      // 1. Add user message to session
      final userMessage = Message(
        text: text,
        isUser: true,
        timestamp: DateTime.now(),
      );
      
      final updatedMessages = List<Message>.from(_currentSession!.messages)..add(userMessage);
      _currentSession = _currentSession!.copyWith(
        messages: updatedMessages,
        lastActive: DateTime.now(),
      );

      // 2. Generate Socratic response with context
      // Note: We use the list as we just built it, cast to model.Message to avoid collision
      final responseText = await _tutorEngine.generateSocraticResponse(
        text, 
        history: updatedMessages.cast<model.Message>().sublist(0, updatedMessages.length - 1)
      );
      
      final assistantMessage = Message(
        text: responseText,
        isUser: false,
        timestamp: DateTime.now(),
      );

      // 3. Add assistant message and save session
      final finalMessages = List<Message>.from(_currentSession!.messages)..add(assistantMessage);
      _currentSession = _currentSession!.copyWith(
        messages: finalMessages,
        lastActive: DateTime.now(),
      );
      
      await SessionService.saveSession(_currentSession!);

      return assistantMessage;
    } catch (e) {
      print('Error calling local LLM: $e');
    }

    return Message(
      text: 'I apologize, but I am having trouble processing your question right now. What were we discussing?',
      isUser: false,
      timestamp: DateTime.now(),
    );
  }
}
