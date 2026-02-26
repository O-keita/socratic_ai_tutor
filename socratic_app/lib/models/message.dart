class Message {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final Map<String, dynamic>? metadata;

  Message({
    required this.text,
    required this.isUser,
    required this.timestamp,
    this.metadata,
  });

  Map<String, dynamic> toJson() => {
    'text': text,
    'isUser': isUser,
    'timestamp': timestamp.toIso8601String(),
    if (metadata != null) 'metadata': metadata,
  };

  factory Message.fromJson(Map<String, dynamic> json) => Message(
    text: json['text'],
    isUser: json['isUser'],
    timestamp: DateTime.parse(json['timestamp']),
    metadata: json['metadata'] as Map<String, dynamic>?,
  );
}
