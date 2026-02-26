import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/message.dart';
import '../models/session.dart';
import '../utils/app_config.dart';

/// Service for communicating with the Socratic Tutor backend API.
/// 
/// This service provides methods for:
/// - Sending messages and receiving AI responses
/// - Managing learning sessions
/// - Fetching performance metrics and analytics
/// - Configuring tutor settings
class ApiService {
  final String baseUrl;
  final http.Client _client;
  
  // Configurable settings for the API
  String? _sessionId;
  String _difficulty = 'intermediate';
  String? _currentTopic;

  ApiService({
    String? baseUrl,
    http.Client? client,
  }) : baseUrl = baseUrl ?? AppConfig.backendUrl,
       _client = client ?? http.Client();

  /// Headers for API requests
  Map<String, String> get _headers => {
    'Content-Type': 'application/json',
    if (_sessionId != null) 'X-Session-Id': _sessionId!,
  };

  /// Start a new learning session
  Future<Session?> startSession({String? topic}) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/session/start'),
        headers: _headers,
        body: jsonEncode({
          'topic': topic,
          'difficulty': _difficulty,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _sessionId = data['session_id'];
        _currentTopic = topic;
        return Session(
          id: data['session_id'],
          topic: topic ?? 'General',
          messages: [],
          startTime: DateTime.now(),
          lastActive: DateTime.now(),
        );
      }
    } catch (e) {
      print('Error starting session: $e');
    }
    return null;
  }

  /// Send a message to the Socratic tutor and receive a response
  Future<Message> sendMessage(String text) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/chat'),
        headers: _headers,
        body: jsonEncode({
          'message': text,
          'session_id': _sessionId,
          'difficulty': _difficulty,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return Message(
          text: data['response'] ?? data['message'] ?? '',
          isUser: false,
          timestamp: DateTime.now(),
        );
      }
    } catch (e) {
      print('Error sending message: $e');
    }

    // Return a fallback Socratic response for development/testing
    return Message(
      text: _generateFallbackResponse(text),
      isUser: false,
      timestamp: DateTime.now(),
    );
  }

  /// Request a hint for the current discussion
  Future<Message> requestHint() async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/hint'),
        headers: _headers,
        body: jsonEncode({
          'session_id': _sessionId,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return Message(
          text: '💡 ${data['hint'] ?? 'Consider approaching this from a different angle.'}',
          isUser: false,
          timestamp: DateTime.now(),
        );
      }
    } catch (e) {
      print('Error requesting hint: $e');
    }

    return Message(
      text: '💡 Hint: Try breaking down the problem into smaller parts. What do you already know for certain?',
      isUser: false,
      timestamp: DateTime.now(),
    );
  }

  /// End the current session and get a summary
  Future<Map<String, dynamic>?> endSession() async {
    if (_sessionId == null) return null;

    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/session/end'),
        headers: _headers,
        body: jsonEncode({
          'session_id': _sessionId,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _sessionId = null;
        return data;
      }
    } catch (e) {
      print('Error ending session: $e');
    }
    return null;
  }

  /// Get performance metrics for the user
  Future<Map<String, dynamic>?> getMetrics() async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl/metrics'),
        headers: _headers,
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
    } catch (e) {
      print('Error fetching metrics: $e');
    }
    return null;
  }

  /// Get session history
  Future<List<Session>> getSessionHistory() async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl/sessions'),
        headers: _headers,
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as List;
        return data.map((json) => Session.fromJson(json)).toList();
      }
    } catch (e) {
      print('Error fetching session history: $e');
    }
    return [];
  }

  /// Update difficulty setting
  void setDifficulty(String difficulty) {
    _difficulty = difficulty;
  }

  /// Get current difficulty
  String get difficulty => _difficulty;

  /// Get current session ID
  String? get sessionId => _sessionId;

  /// Get current topic
  String? get currentTopic => _currentTopic;

  /// Check if backend is available
  Future<bool> healthCheck() async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl/health'),
        headers: _headers,
      ).timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  /// Generate a fallback Socratic response when backend is unavailable
  String _generateFallbackResponse(String input) {
    final responses = [
      'That\'s an interesting thought. Can you elaborate on what led you to that conclusion?',
      'I see where you\'re coming from. What evidence supports this perspective?',
      'Let\'s explore that further. What assumptions are you making here?',
      'Good observation! How might someone with a different viewpoint respond?',
      'That raises an important question. What would be the implications if that were true?',
      'Interesting! Can you think of any counterexamples to this idea?',
      'You\'re on the right track. What would you need to know to be more certain?',
      'Let\'s dig deeper. What\'s the underlying principle here?',
    ];
    
    return responses[input.length % responses.length];
  }

  /// Dispose of resources
  void dispose() {
    _client.close();
  }
}
