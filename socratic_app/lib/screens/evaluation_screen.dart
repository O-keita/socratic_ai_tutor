import 'package:flutter/material.dart';

class EvaluationScreen extends StatelessWidget {
  const EvaluationScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Performance Evaluation'),
      ),
      body: const Center(
        child: Text('Coming soon: Session analytics and progress metrics.'),
      ),
    );
  }
}
