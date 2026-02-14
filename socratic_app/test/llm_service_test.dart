
import 'package:flutter_test/flutter_test.dart';
import 'package:socratic_tutor/services/llm_service.dart';
import 'dart:io';

void main() {
  // This test requires the model file to be present in assets/models/
  // and might fail in a pure CI environment without the llama_cpp shared libraries.
  // However, we can test the initialization logic if the file exists.

  test('SocraticLlmService initialization test', () async {
    // Note: This test might be skipped or fail if full hardware acceleration 
    // or llama.cpp libraries are missing in the test environment.
    // We wrap it to see if we can at least trigger the file check logic.
    
    // Check if the asset exists where we expect it
    final modelAsset = File('assets/models/socratic-q4_k_m.gguf');
    expect(await modelAsset.exists(), isTrue, reason: 'Model asset must exist for testing');
  });
}
