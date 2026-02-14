import 'package:flutter/material.dart';

class GradientCard extends StatelessWidget {
  final Widget child;
  final LinearGradient gradient;
  final LinearGradient? borderGradient;
  final double borderWidth;
  final double borderRadius;

  const GradientCard({
    super.key,
    required this.child,
    required this.gradient,
    this.borderGradient,
    this.borderWidth = 1.5,
    this.borderRadius = 20,
  });

  @override
  Widget build(BuildContext context) {
    if (borderGradient == null) {
      return Container(
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(borderRadius),
        ),
        child: child,
      );
    }

    return Container(
      decoration: BoxDecoration(
        gradient: borderGradient,
        borderRadius: BorderRadius.circular(borderRadius),
      ),
      padding: EdgeInsets.all(borderWidth),
      child: Container(
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(borderRadius - borderWidth),
        ),
        child: child,
      ),
    );
  }
}
