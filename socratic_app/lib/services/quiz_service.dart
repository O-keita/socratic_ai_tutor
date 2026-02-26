import 'dart:convert';
import 'package:flutter/services.dart';
import '../models/quiz_question.dart';

class QuizService {
  static final QuizService _instance = QuizService._internal();
  factory QuizService() => _instance;
  QuizService._internal();

  List<QuizQuestion> _questions = [];
  bool _isLoading = false;

  List<QuizQuestion> get questions => _questions;
  bool get isLoading => _isLoading;

  /// Loads the default quiz (Machine Learning) for now
  Future<void> loadQuizData({String quizId = 'machine-learning'}) async {
    // If questions are already loaded for THIS quiz, don't reload
    if (_questions.isNotEmpty) return;
    
    _isLoading = true;
    try {
      // Following course pattern: assets/quizzes/$id/quiz.json
      final String jsonString = await rootBundle.loadString('assets/quizzes/$quizId/quiz.json');
      final List<dynamic> jsonData = jsonDecode(jsonString);
      _questions = jsonData.map((q) => QuizQuestion.fromJson(q)).toList();
      print('QuizService: Loaded ${_questions.length} questions for $quizId');
    } catch (e) {
      print('Error loading quiz data for $quizId: $e');
    } finally {
      _isLoading = false;
    }
  }

  List<QuizQuestion> getRandomQuestions(int count) {
    if (_questions.isEmpty) return [];
    final List<QuizQuestion> shuffled = List.from(_questions)..shuffle();
    return shuffled.take(count).toList();
  }

  /// Clears the current quiz questions from memory
  void clearCache() {
    _questions = [];
  }
}
