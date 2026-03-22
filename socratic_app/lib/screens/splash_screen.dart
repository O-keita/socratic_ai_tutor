import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/auth_service.dart';
import '../theme/app_theme.dart';
import 'auth_screen.dart';
import 'eula_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 1500));
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(_controller);
    _controller.forward();
    _checkAuthAndNavigate();
  }

  Future<void> _checkAuthAndNavigate() async {
    await Future.delayed(const Duration(milliseconds: 800));
    if (!mounted) return;
    final authService = Provider.of<AuthService>(context, listen: false);
    int attempts = 0;
    while (!authService.isInitialized && attempts < 20) {
      await Future.delayed(const Duration(milliseconds: 100));
      attempts++;
    }
    if (!mounted) return;
    final navigator = Navigator.of(context);
    if (authService.isAuthenticated) {
      final userId = authService.currentUser!.id;
      final accepted = await EulaScreen.isAccepted(userId);
      if (!mounted) return;
      if (accepted) {
        navigator.pushReplacementNamed('/home');
      } else {
        navigator.pushReplacement(
          MaterialPageRoute(
            builder: (_) => EulaScreen(blocking: true, userId: userId),
          ),
        );
      }
    } else {
      navigator.pushReplacement(MaterialPageRoute(builder: (_) => const AuthScreen()));
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFFEE4C4), Color(0xFFFEF6EE), Color(0xFFFFFFFF)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: FadeTransition(
          opacity: _fadeAnimation,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Hero(
                tag: 'logo',
                child: Container(
                  width: 130,
                  height: 130,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: AppTheme.primaryGradient,
                    boxShadow: [
                      BoxShadow(
                        color: AppTheme.accentOrange.withValues(alpha: 0.4),
                        blurRadius: 40,
                        spreadRadius: 4,
                        offset: const Offset(0, 10),
                      ),
                    ],
                  ),
                  child: ClipOval(
                    child: Image.asset(
                      'assets/images/logo.png',
                      fit: BoxFit.contain,
                      errorBuilder: (context, error, stackTrace) => const Icon(
                        Icons.psychology,
                        size: 64,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 28),
              const Text(
                'Bantaba AI',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF1E293B),
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'Bridging the Digital Reasoning Divide',
                style: TextStyle(
                  color: Color(0xFF64748B),
                  fontSize: 14,
                  fontStyle: FontStyle.italic,
                ),
              ),
              const SizedBox(height: 48),
              const SizedBox(
                width: 28,
                height: 28,
                child: CircularProgressIndicator(
                  strokeWidth: 3,
                  valueColor: AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
