import '../models/message.dart';

/// Response object from an LLM engine including metadata.
class LLMResponse {
  final String text;
  final Map<String, dynamic>? metadata;

  LLMResponse(this.text, {this.metadata});
}

/// Abstract interface for any tutoring engine (Local or Remote).
/// This allows the UI and Bridge to switch between Offline and Online modes
/// without changing their logic.
abstract class TutorEngine {
  /// Whether the engine is initialized and ready for inference
  bool get isReady;
  
  /// Whether the engine is currently generating a response
  bool get isGenerating;

  /// Initialize the engine (load model, connect to socket, etc.)
  Future<bool> initialize();

  /// Generate a stream of tokens for a Socratic response
  Stream<String> generateResponse(String prompt, {List<Message>? history, int maxTokens = 150});

  /// Generate a full string response (non-streaming)
  Future<LLMResponse> generateSocraticResponse(String prompt, {List<Message>? history, int maxTokens = 150});
}
