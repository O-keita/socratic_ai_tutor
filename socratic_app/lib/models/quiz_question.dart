class QuizQuestion {
  final String id;
  final String topic;
  final String question;
  final List<String> options;
  final int correctIndex;
  final String explanation;

  QuizQuestion({
    required this.id,
    required this.topic,
    required this.question,
    required this.options,
    required this.correctIndex,
    required this.explanation,
  });

  factory QuizQuestion.fromJson(Map<String, dynamic> json) {
    return QuizQuestion(
      id: json['id'],
      topic: json['topic'],
      question: json['question'],
      options: List<String>.from(json['options']),
      correctIndex: json['correctIndex'],
      explanation: json['explanation'],
    );
  }
}
