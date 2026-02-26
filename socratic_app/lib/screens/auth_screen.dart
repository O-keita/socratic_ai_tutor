import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/auth_service.dart';
import '../theme/app_theme.dart';
import '../utils/app_snackbar.dart';
import 'onboarding_screen.dart';

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  final _formKey = GlobalKey<FormState>();
  final _usernameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  bool _isLogin = true;
  bool _obscurePassword = true;

  @override
  void dispose() {
    _usernameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _submit() async {
    if (!_formKey.currentState!.validate()) return;

    final authService = Provider.of<AuthService>(context, listen: false);
    String? errorMessage;

    try {
      if (_isLogin) {
        final success = await authService.login(
          _emailController.text,
          _passwordController.text,
        );
        if (success && mounted) {
          Navigator.of(context).pushReplacementNamed('/home');
        } else {
          errorMessage = 'Login failed. Please check your credentials.';
        }
      } else {
        final success = await authService.register(
          _usernameController.text,
          _emailController.text,
          _passwordController.text,
        );
        if (success && mounted) {
          Navigator.of(context).pushReplacement(
            MaterialPageRoute(builder: (_) => OnboardingScreen()),
          );
        } else {
          errorMessage = 'Registration failed.';
        }
      }
    } catch (e) {
      errorMessage = e.toString().replaceFirst('Exception: ', '');
    }

    if (errorMessage != null && mounted) {
      AppSnackBar.error(context, errorMessage);
    }
  }

  @override
  Widget build(BuildContext context) {
    final authService = context.watch<AuthService>();
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final size = MediaQuery.of(context).size;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: isDark
              ? AppTheme.backgroundGradient
              : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Column(
              children: [
                SizedBox(height: size.height * 0.07),

                // ── Branding ──────────────────────────────────────────────
                _buildHeader(isDark),

                const SizedBox(height: 36),

                // ── Form card ─────────────────────────────────────────────
                _buildFormCard(authService, isDark),

                const SizedBox(height: 20),

                // ── Toggle ────────────────────────────────────────────────
                _buildToggle(),

                const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(bool isDark) {
    return Column(
      children: [
        // Logo with glow
        Hero(
          tag: 'logo',
          child: Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              gradient: AppTheme.primaryGradient,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: AppTheme.accentOrange.withValues(alpha: 0.4),
                  blurRadius: 28,
                  spreadRadius: 2,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: const Icon(Icons.psychology, size: 40, color: Colors.white),
          ),
        ),

        const SizedBox(height: 16),

        Text(
          'Socratic AI',
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.w700,
              ),
        ),

        const SizedBox(height: 6),

        Text(
          'Learn through thoughtful questions',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
      ],
    );
  }

  Widget _buildFormCard(AuthService authService, bool isDark) {
    return Container(
      decoration: BoxDecoration(
        color: isDark ? AppTheme.surfaceCard : Colors.white,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: isDark
              ? AppTheme.primaryLight.withValues(alpha: 0.2)
              : AppTheme.tagBackground,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: isDark ? 0.25 : 0.08),
            blurRadius: 40,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      padding: const EdgeInsets.all(28),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              _isLogin ? 'Welcome Back' : 'Create Account',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: AppTheme.accentOrange,
                  ),
              textAlign: TextAlign.center,
            ),

            const SizedBox(height: 24),

            // Username field (register only)
            if (!_isLogin) ...[
              _buildField(
                controller: _usernameController,
                label: 'Username',
                icon: Icons.person_outline,
                isDark: isDark,
                validator: (v) => v!.isEmpty ? 'Enter a username' : null,
              ),
              const SizedBox(height: 14),
            ],

            // Email field
            _buildField(
              controller: _emailController,
              label: _isLogin ? 'Email or Username' : 'Email',
              icon: Icons.email_outlined,
              isDark: isDark,
              keyboardType: TextInputType.emailAddress,
              validator: (v) => v!.isEmpty ? 'Enter your email' : null,
            ),

            const SizedBox(height: 14),

            // Password field
            _buildField(
              controller: _passwordController,
              label: 'Password',
              icon: Icons.lock_outline,
              isDark: isDark,
              obscureText: _obscurePassword,
              suffixIcon: IconButton(
                icon: Icon(
                  _obscurePassword
                      ? Icons.visibility_outlined
                      : Icons.visibility_off_outlined,
                  size: 20,
                  color: AppTheme.textMuted,
                ),
                onPressed: () =>
                    setState(() => _obscurePassword = !_obscurePassword),
              ),
              validator: (v) =>
                  v!.length < 6 ? 'At least 6 characters' : null,
            ),

            const SizedBox(height: 28),

            // Submit button
            Container(
              decoration: BoxDecoration(
                gradient: AppTheme.buttonGradient,
                borderRadius: BorderRadius.circular(14),
                boxShadow: [
                  BoxShadow(
                    color: AppTheme.accentOrange.withValues(alpha: 0.3),
                    blurRadius: 18,
                    offset: const Offset(0, 7),
                  ),
                ],
              ),
              child: ElevatedButton(
                onPressed: authService.isLoading ? null : _submit,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.transparent,
                  shadowColor: Colors.transparent,
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
                child: authService.isLoading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2.5,
                          color: Colors.white,
                        ),
                      )
                    : Text(
                        _isLogin ? 'Sign In' : 'Create Account',
                        style: const TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                          color: Colors.white,
                        ),
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    required bool isDark,
    bool obscureText = false,
    Widget? suffixIcon,
    TextInputType? keyboardType,
    String? Function(String?)? validator,
  }) {
    return TextFormField(
      controller: controller,
      obscureText: obscureText,
      keyboardType: keyboardType,
      validator: validator,
      style: TextStyle(
        color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary,
        fontSize: 14,
      ),
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon, size: 20, color: AppTheme.textMuted),
        suffixIcon: suffixIcon,
        labelStyle: const TextStyle(color: AppTheme.textMuted, fontSize: 14),
        filled: true,
        fillColor: isDark
            ? AppTheme.primaryDark.withValues(alpha: 0.6)
            : AppTheme.lightSurfaceCard,
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(
            color: isDark
                ? AppTheme.primaryLight.withValues(alpha: 0.3)
                : Colors.grey.withValues(alpha: 0.2),
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide:
              const BorderSide(color: AppTheme.accentOrange, width: 1.5),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppTheme.error, width: 1.5),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppTheme.error, width: 1.5),
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
    );
  }

  Widget _buildToggle() {
    return TextButton(
      onPressed: () => setState(() {
        _isLogin = !_isLogin;
        _obscurePassword = true;
      }),
      child: RichText(
        text: TextSpan(
          style: TextStyle(
            fontSize: 13,
            color: AppTheme.textMuted,
          ),
          children: [
            TextSpan(
                text: _isLogin
                    ? "Don't have an account? "
                    : 'Already have an account? '),
            TextSpan(
              text: _isLogin ? 'Sign Up' : 'Sign In',
              style: const TextStyle(
                color: AppTheme.accentOrange,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
