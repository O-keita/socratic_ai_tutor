import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // ── Primary / Dark palette ────────────────────────────────────────────────
  static const Color primaryDark   = Color(0xFF0F172A); // Slate 900
  static const Color primaryMedium = Color(0xFF1E293B); // Slate 800
  static const Color primaryLight  = Color(0xFF334155); // Slate 700
  static const Color darkBlue      = Color(0xFF0F172A);

  // ── Light palette ─────────────────────────────────────────────────────────
  static const Color lightBackground    = Color(0xFFF8FAFC); // Slate 50
  static const Color lightSurface       = Color(0xFFFFFFFF);
  static const Color lightSurfaceCard   = Color(0xFFF1F5F9); // Slate 100
  static const Color lightTextPrimary   = Color(0xFF0F172A);
  static const Color lightTextSecondary = Color(0xFF475569); // Slate 600
  static const Color lightTextMuted     = Color(0xFF94A3B8); // Slate 400

  // ── Accent orange ─────────────────────────────────────────────────────────
  static const Color accentOrange      = Color(0xFFF97316); // Orange 500
  static const Color accentOrangeLight = Color(0xFFFB923C); // Orange 400
  static const Color accentOrangeDark  = Color(0xFFEA580C); // Orange 600

  // ── Gradient stops ────────────────────────────────────────────────────────
  static const Color gradientStart  = Color(0xFFF97316);
  static const Color gradientMiddle = Color(0xFFFB923C);
  static const Color gradientEnd    = Color(0xFFFDBA74);

  // ── Surface tokens ────────────────────────────────────────────────────────
  static const Color surfaceDark      = Color(0xFF0F172A);
  static const Color surfaceCard      = Color(0xFF1E293B);
  static const Color surfaceCardHover = Color(0xFF334155);

  // ── Tag colors ────────────────────────────────────────────────────────────
  static const Color tagBackground = Color(0xFFE2E8F0);
  static const Color tagText       = Color(0xFF1E293B);

  // ── Text tokens ───────────────────────────────────────────────────────────
  static const Color textPrimary   = Color(0xFFF8FAFC);
  static const Color textSecondary = Color(0xFFCBD5E1); // Slate 300
  static const Color textMuted     = Color(0xFF64748B); // Slate 500

  // ── Status ────────────────────────────────────────────────────────────────
  static const Color success = Color(0xFF10B981);
  static const Color warning = Color(0xFFF59E0B);
  static const Color error   = Color(0xFFEF4444);

  // ── Gradients ─────────────────────────────────────────────────────────────
  static const LinearGradient primaryGradient = LinearGradient(
    colors: [gradientStart, gradientEnd],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient cardGradient = LinearGradient(
    colors: [Color(0xFF1E293B), Color(0xFF0F172A)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient backgroundGradient = LinearGradient(
    colors: [Color(0xFF1E293B), Color(0xFF0F172A)],
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
  );

  static const LinearGradient buttonGradient = LinearGradient(
    colors: [accentOrange, accentOrangeDark],
    begin: Alignment.centerLeft,
    end: Alignment.centerRight,
  );

  static const LinearGradient borderGradient = LinearGradient(
    colors: [Color(0xFFF97316), Color(0xFFEA580C), Color(0xFF9A3412)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient lightBackgroundGradient = LinearGradient(
    colors: [Color(0xFFF8FAFC), Color(0xFFE2E8F0)],
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
  );

  static const LinearGradient lightCardGradient = LinearGradient(
    colors: [Color(0xFFFFFFFF), Color(0xFFF1F5F9)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  // ── Typography helper — Plus Jakarta Sans ─────────────────────────────────
  // A modern humanist sans-serif that reads beautifully at all sizes.
  static TextStyle _pjs({
    required double size,
    FontWeight weight = FontWeight.w400,
    Color? color,
    double height = 1.45,
    double? letterSpacing,
  }) =>
      GoogleFonts.plusJakartaSans(
        fontSize: size,
        fontWeight: weight,
        color: color,
        height: height,
        letterSpacing: letterSpacing,
      );

  // ── Text theme factory ────────────────────────────────────────────────────
  static TextTheme _buildTextTheme({required bool dark}) {
    final primary   = dark ? textPrimary   : lightTextPrimary;
    final secondary = dark ? textSecondary : lightTextSecondary;
    final muted     = dark ? textMuted     : lightTextMuted;

    return TextTheme(
      displayLarge:   _pjs(size: 36, weight: FontWeight.w700, color: primary),
      displayMedium:  _pjs(size: 32, weight: FontWeight.w700, color: primary),
      displaySmall:   _pjs(size: 28, weight: FontWeight.w700, color: primary),
      headlineLarge:  _pjs(size: 24, weight: FontWeight.w700, color: primary),
      headlineMedium: _pjs(size: 20, weight: FontWeight.w600, color: primary),
      headlineSmall:  _pjs(size: 18, weight: FontWeight.w600, color: primary),
      titleLarge:     _pjs(size: 18, weight: FontWeight.w600, color: primary),
      titleMedium:    _pjs(size: 16, weight: FontWeight.w500, color: primary),
      titleSmall:     _pjs(size: 14, weight: FontWeight.w500, color: primary),
      bodyLarge:      _pjs(size: 15, color: secondary),
      bodyMedium:     _pjs(size: 14, color: secondary),
      bodySmall:      _pjs(size: 12, color: muted),
      labelLarge:     _pjs(size: 14, weight: FontWeight.w600, color: primary),
      labelMedium:    _pjs(size: 12, weight: FontWeight.w500, color: muted),
      labelSmall:     _pjs(size: 11, weight: FontWeight.w500, color: muted),
    );
  }

  // ── Dark Theme ────────────────────────────────────────────────────────────
  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: primaryDark,
      colorScheme: const ColorScheme.dark(
        primary: accentOrange,
        secondary: accentOrangeLight,
        surface: surfaceCard,
        error: error,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: textPrimary,
        onError: Colors.white,
      ),
      textTheme: _buildTextTheme(dark: true),
      appBarTheme: AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: _pjs(size: 22, weight: FontWeight.w700, color: textPrimary),
        iconTheme: const IconThemeData(color: textPrimary),
      ),
      cardTheme: CardThemeData(
        color: surfaceCard,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceCard,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(color: primaryLight.withValues(alpha: 0.3)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: accentOrange, width: 1.5),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: error, width: 1.5),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: error, width: 1.5),
        ),
        hintStyle: _pjs(size: 14, color: textMuted),
        labelStyle: _pjs(size: 14, color: textMuted),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: accentOrange,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          textStyle: _pjs(size: 15, weight: FontWeight.w600),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(foregroundColor: accentOrange),
      ),
      // Legacy (kept for any remaining BottomNavigationBar references)
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: surfaceDark,
        selectedItemColor: accentOrange,
        unselectedItemColor: textMuted,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
      ),
      // Material 3 NavigationBar — pill indicator, no chrome
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: surfaceDark,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        indicatorColor: accentOrange.withValues(alpha: 0.15),
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          final sel = states.contains(WidgetState.selected);
          return _pjs(
            size: 11,
            weight: sel ? FontWeight.w600 : FontWeight.w400,
            color: sel ? accentOrange : textMuted,
          );
        }),
        iconTheme: WidgetStateProperty.resolveWith((states) {
          final sel = states.contains(WidgetState.selected);
          return IconThemeData(color: sel ? accentOrange : textMuted);
        }),
      ),
    );
  }

  // ── Light Theme ───────────────────────────────────────────────────────────
  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: lightBackground,
      colorScheme: const ColorScheme.light(
        primary: accentOrange,
        secondary: accentOrangeDark,
        surface: lightSurface,
        error: error,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: lightTextPrimary,
        onError: Colors.white,
      ),
      textTheme: _buildTextTheme(dark: false),
      appBarTheme: AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: _pjs(size: 22, weight: FontWeight.w700, color: lightTextPrimary),
        iconTheme: const IconThemeData(color: lightTextPrimary),
      ),
      cardTheme: CardThemeData(
        color: lightSurfaceCard,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: lightSurfaceCard,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(color: accentOrange.withValues(alpha: 0.2)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: accentOrange, width: 1.5),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: error, width: 1.5),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: error, width: 1.5),
        ),
        hintStyle: _pjs(size: 14, color: lightTextMuted),
        labelStyle: _pjs(size: 14, color: lightTextMuted),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: accentOrange,
          foregroundColor: Colors.white,
          elevation: 2,
          shadowColor: accentOrange.withValues(alpha: 0.3),
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          textStyle: _pjs(size: 15, weight: FontWeight.w600),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(foregroundColor: accentOrange),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: lightSurface,
        selectedItemColor: accentOrange,
        unselectedItemColor: lightTextMuted,
        type: BottomNavigationBarType.fixed,
        elevation: 8,
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: lightSurface,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        indicatorColor: accentOrange.withValues(alpha: 0.12),
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          final sel = states.contains(WidgetState.selected);
          return _pjs(
            size: 11,
            weight: sel ? FontWeight.w600 : FontWeight.w400,
            color: sel ? accentOrange : lightTextMuted,
          );
        }),
        iconTheme: WidgetStateProperty.resolveWith((states) {
          final sel = states.contains(WidgetState.selected);
          return IconThemeData(color: sel ? accentOrange : lightTextMuted);
        }),
      ),
    );
  }
}
