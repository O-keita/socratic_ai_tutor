import 'package:flutter/material.dart';
import '../models/message.dart';
import '../services/tutor_bridge.dart';
import '../widgets/question_card.dart';
import '../widgets/response_input.dart';

class TutorScreen extends StatefulWidget {
  const TutorScreen({super.key});

  @override
  State<TutorScreen> createState() => _TutorScreenState();
}

class _TutorScreenState extends State<TutorScreen> {
  final List<Message> _messages = [
    Message(
      text: 'Hello! I am your Socratic Tutor. What topic are we exploring today?',
      isUser: false,
      timestamp: DateTime.now(),
    ),
  ];
  final TextEditingController _controller = TextEditingController();
  final TutorBridge _tutorBridge = TutorBridge();
  bool _isLoading = false;

  void _handleSend() async {
    final text = _controller.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _messages.add(Message(text: text, isUser: true, timestamp: DateTime.now()));
      _isLoading = true;
    });
    _controller.clear();

    final response = await _tutorBridge.sendMessage(text);

    setState(() {
      _messages.add(response);
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Socratic AI Tutor'),
        elevation: 0,
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.black87,
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(vertical: 16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                
                // Use QuestionCard for tutor messages to make them stand out
                if (!message.isUser) {
                  return QuestionCard(
                    question: message.text,
                    onHintRequested: () {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Hint: Consider the implications of your last statement.')),
                      );
                    },
                  );
                }

                return Align(
                  alignment: Alignment.centerRight,
                  child: Container(
                    margin: const EdgeInsets.fromLTRB(64, 4, 16, 4),
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.teal.withOpacity(0.1),
                      borderRadius: const BorderRadius.only(
                        topLeft: Radius.circular(16),
                        topRight: Radius.circular(16),
                        bottomLeft: Radius.circular(16),
                      ),
                    ),
                    child: Text(
                      message.text,
                      style: const TextStyle(fontSize: 16),
                    ),
                  ),
                );
              },
            ),
          ),
          ResponseInput(
            controller: _controller,
            onSend: _handleSend,
            isLoading: _isLoading,
          ),
        ],
      ),
    );
  }
}
