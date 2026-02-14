import 'package:flutter/material.dart';

class QuestionCard extends StatelessWidget {
  final String question;
  final VoidCallback? onHintRequested;

  const QuestionCard({
    super.key,
    required this.question,
    this.onHintRequested,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.psychology, color: Colors.teal),
                SizedBox(width: 8),
                Text(
                  'Socratic Challenge',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.teal,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              question,
              style: const TextStyle(fontSize: 22, height: 1.4),
            ),
            if (onHintRequested != null) ...[
              const SizedBox(height: 12),
              Align(
                alignment: Alignment.centerRight,
                child: TextButton.icon(
                  onPressed: onHintRequested,
                  icon: const Icon(Icons.lightbulb_outline, size: 16),
                  label: const Text('Need a hint?'),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
