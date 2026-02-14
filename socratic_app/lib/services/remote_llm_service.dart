import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/message.dart';
import 'tutor_engine.dart';

/// Service for remote LLM inference using the FastAPI backend.
/// Connects to the Qwen3-0.6B model hosted remotely.
class RemoteLlmService implements TutorEngine {
  final String baseUrl;
  bool _isGenerating = false;

  RemoteLlmService({this.baseUrl = 'http://10.0.2.2:8000'});

  @override
  bool get isReady => true; // Remote server is assumed ready if reachable

  @override
  bool get isGenerating => _isGenerating;

  @override
  Future<bool> initialize() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/')).timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  @override
  Stream<String> generateResponse(String prompt, {List<Message>? history, int maxTokens = 150}) async* {
    _isGenerating = true;
    
    // In a real implementation, this would use WebSockets or Server-Sent Events (SSE)
    // For now, we'll simulate streaming by calling the non-streaming endpoint
    // and yielding the result in chunks, or just yielding the whole thing if it's simpler.
    try {
      final result = await generateSocraticResponse(prompt, history: history, maxTokens: maxTokens);
      final tokens = result.split(' ');
      for (var token in tokens) {
        yield '$token ';
        await Future.delayed(const Duration(milliseconds: 50));
      }
    } catch (e) {
      yield 'Error connecting to remote tutor.';
    } finally {
      _isGenerating = false;
    }
  }

  @override
  Future<String> generateSocraticResponse(String prompt, {List<Message>? history, int maxTokens = 150}) async {
    _isGenerating = true;
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/chat'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'message': prompt,
          'max_tokens': maxTokens,
          'history': history?.map((m) => m.toJson()).toList(),
        }),
      ).timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['response'] ?? 'No response from server.';
      } else {
        return 'Server error: ${response.statusCode}';
      }
    } catch (e) {
      return 'Connection error: $e';
    } finally {
      _isGenerating = false;
    }
  }
}
