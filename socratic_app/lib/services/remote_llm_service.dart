import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/message.dart';
import '../utils/app_config.dart';
import 'auth_service.dart';
import 'tutor_engine.dart';

/// Service for remote LLM inference using the FastAPI backend.
/// Connects to the Qwen3-0.6B model hosted remotely.
///
/// TODO: Replace the simulated streaming in [generateResponse] with a real
/// Server-Sent Events (SSE) implementation once the backend supports it.
class RemoteLlmService implements TutorEngine {
  final String baseUrl;
  bool _isGenerating = false;

  RemoteLlmService({String? url}) : baseUrl = url ?? AppConfig.backendUrl;

  @override
  bool get isReady => true;

  @override
  bool get isGenerating => _isGenerating;

  @override
  Future<bool> initialize() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  @override
  Stream<String> generateResponse(
    String prompt, {
    List<Message>? history,
    int maxTokens = 150,
  }) async* {
    // Note: real SSE streaming is a future improvement.
    // For now we fetch the full response and yield it atomically.
    final result = await generateSocraticResponse(
      prompt,
      history: history,
      maxTokens: maxTokens,
    );
    yield result.text;
  }

  @override
  Future<LLMResponse> generateSocraticResponse(
    String prompt, {
    List<Message>? history,
    int maxTokens = 150,
  }) async {
    _isGenerating = true;
    try {
      final token = await AuthService().getToken();
      final headers = <String, String>{
        'Content-Type': 'application/json',
        if (token != null) 'Authorization': 'Bearer $token',
      };

      final response = await http
          .post(
            Uri.parse('$baseUrl/chat'),
            headers: headers,
            body: jsonEncode({
              'message': prompt,
              'max_tokens': maxTokens,
              'history': history
                  ?.map((m) => {
                        'role': m.isUser ? 'user' : 'assistant',
                        'content': m.text,
                      })
                  .toList(),
            }),
          )
          .timeout(const Duration(seconds: 60));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        String tutorResponse = (data['response'] as String? ?? '').trim();
        // Strip any <think> blocks that slipped through the backend
        tutorResponse =
            tutorResponse.replaceAll(RegExp(r'<think>[\s\S]*?<\/think>'), '').trim();

        final metadata = {
          'socratic_index': data['socratic_index'],
          'scaffolding_level': data['scaffolding_level'],
          'sentiment': data['sentiment'],
        };
        return LLMResponse(tutorResponse, metadata: metadata);
      } else if (response.statusCode == 401) {
        // Token expired or invalid — caller should handle logout
        await AuthService().logout();
        throw Exception('Session expired. Please log in again.');
      } else if (response.statusCode == 503) {
        throw Exception('The AI is taking too long. Please try again.');
      } else {
        throw Exception('Server error (${response.statusCode}). Please try again.');
      }
    } on Exception {
      rethrow;
    } finally {
      _isGenerating = false;
    }
  }
}
