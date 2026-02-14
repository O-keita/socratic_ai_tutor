import 'message.dart';

class Session {
  final String id;
  final String topic;
  final List<Message> messages;
  final DateTime startTime;
  final DateTime lastActive;

  Session({
    required this.id,
    required this.topic,
    required this.messages,
    required this.startTime,
    required this.lastActive,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'topic': topic,
    'messages': messages.map((m) => m.toJson()).toList(),
    'startTime': startTime.toIso8601String(),
    'lastActive': lastActive.toIso8601String(),
  };

  factory Session.fromJson(Map<String, dynamic> json) => Session(
    id: json['id'],
    topic: json['topic'],
    messages: (json['messages'] as List).map((m) => Message.fromJson(m)).toList(),
    startTime: DateTime.parse(json['startTime']),
    lastActive: json['lastActive'] != null 
        ? DateTime.parse(json['lastActive']) 
        : DateTime.parse(json['startTime']),
  );

  Session copyWith({
    String? id,
    String? topic,
    List<Message>? messages,
    DateTime? startTime,
    DateTime? lastActive,
  }) {
    return Session(
      id: id ?? this.id,
      topic: topic ?? this.topic,
      messages: messages ?? this.messages,
      startTime: startTime ?? this.startTime,
      lastActive: lastActive ?? this.lastActive,
    );
  }
}
