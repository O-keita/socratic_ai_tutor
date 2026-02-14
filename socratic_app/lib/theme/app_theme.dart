import 'package:flutter/material.dart';

class AppTheme {
  // Primary Colors - Modern Dark Blue Palette
  static const Color primaryDark = Color(0xFF0F172A); // Slate 900
  static const Color primaryMedium = Color(0xFF1E293B); // Slate 800
  static const Color primaryLight = Color(0xFF334155); // Slate 700
  static const Color darkBlue = Color(0xFF0F172A); // Added for UI compatibility
  
  // Light Theme Colors
  static const Color lightBackground = Color(0xFFF8FAFC); // Slate 50
  static const Color lightSurface = Color(0xFFFFFFFF);
  static const Color lightSurfaceCard = Color(0xFFF1F5F9); // Slate 100
  static const Color lightTextPrimary = Color(0xFF0F172A);
  static const Color lightTextSecondary = Color(0xFF475569); // Slate 600
  static const Color lightTextMuted = Color(0xFF94A3B8); // Slate 400
  
  // Accent Colors - Modern Orange
  static const Color accentOrange = Color(0xFFF97316); // Orange 500
  static const Color accentOrangeLight = Color(0xFFFB923C); // Orange 400
  static const Color accentOrangeDark = Color(0xFFEA580C); // Orange 600
  
  // Gradient Colors
  static const Color gradientStart = Color(0xFFF97316);
  static const Color gradientMiddle = Color(0xFFFB923C);
  static const Color gradientEnd = Color(0xFFFDBA74); // Orange 300
  
  // Surface Colors
  static const Color surfaceDark = Color(0xFF0F172A);
  static const Color surfaceCard = Color(0xFF1E293B);
  static const Color surfaceCardHover = Color(0xFF334155);
  
  // Tag colors for non-orange labels
  static const Color tagBackground = Color(0xFFE2E8F0);
  static const Color tagText = Color(0xFF1E293B);
  
  // Text Colors
  static const Color textPrimary = Color(0xFFF8FAFC);
  static const Color textSecondary = Color(0xFFCBD5E1); // Slate 300
  static const Color textMuted = Color(0xFF64748B); // Slate 500
  
  // Status Colors
  static const Color success = Color(0xFF10B981);
  static const Color warning = Color(0xFFF59E0B);
  static const Color error = Color(0xFFEF4444);
  
  // Gradients
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

  // Border Gradient for Cards
  static const LinearGradient borderGradient = LinearGradient(
    colors: [Color(0xFFF97316), Color(0xFFEA580C), Color(0xFF9A3412)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  // Theme Data
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
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          color: textPrimary,
          fontSize: 24,
          fontWeight: FontWeight.bold,
        ),
        iconTheme: IconThemeData(color: textPrimary),
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
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: primaryLight.withOpacity(0.3)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: accentOrange, width: 2),
        ),
        hintStyle: const TextStyle(color: textMuted),
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: accentOrange,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(30),
          ),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: accentOrange,
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: surfaceDark,
        selectedItemColor: accentOrange,
        unselectedItemColor: textMuted,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(
          fontSize: 36,
          fontWeight: FontWeight.bold,
          color: textPrimary,
        ),
        displayMedium: TextStyle(
          fontSize: 32,
          fontWeight: FontWeight.bold,
          color: textPrimary,
        ),
        headlineLarge: TextStyle(
          fontSize: 28,
          fontWeight: FontWeight.bold,
          color: textPrimary,
        ),
        headlineMedium: TextStyle(
          fontSize: 24,
          fontWeight: FontWeight.w600,
          color: textPrimary,
        ),
        titleLarge: TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.w600,
          color: textPrimary,
        ),
        titleMedium: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w500,
          color: textPrimary,
        ),
        bodyLarge: TextStyle(
          fontSize: 18,
          color: textSecondary,
        ),
        bodyMedium: TextStyle(
          fontSize: 16,
          color: textSecondary,
        ),
        bodySmall: TextStyle(
          fontSize: 14,
          color: textMuted,
        ),
      ),
    );
  }

  // Light Theme Data
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
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          color: lightTextPrimary,
          fontSize: 24,
          fontWeight: FontWeight.bold,
        ),
        iconTheme: IconThemeData(color: lightTextPrimary),
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
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: accentOrange.withOpacity(0.2)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: accentOrange, width: 2),
        ),
        hintStyle: const TextStyle(color: lightTextMuted),
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: accentOrange,
          foregroundColor: Colors.white,
          elevation: 2,
          shadowColor: accentOrange.withOpacity(0.3),
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(30),
          ),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: accentOrange,
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: lightSurface,
        selectedItemColor: accentOrange,
        unselectedItemColor: lightTextMuted,
        type: BottomNavigationBarType.fixed,
        elevation: 8,
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(
          fontSize: 36,
          fontWeight: FontWeight.bold,
          color: lightTextPrimary,
        ),
        displayMedium: TextStyle(
          fontSize: 32,
          fontWeight: FontWeight.bold,
          color: lightTextPrimary,
        ),
        headlineLarge: TextStyle(
          fontSize: 28,
          fontWeight: FontWeight.bold,
          color: lightTextPrimary,
        ),
        headlineMedium: TextStyle(
          fontSize: 24,
          fontWeight: FontWeight.w600,
          color: lightTextPrimary,
        ),
        titleLarge: TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.w600,
          color: lightTextPrimary,
        ),
        titleMedium: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w500,
          color: lightTextPrimary,
        ),
        bodyLarge: TextStyle(
          fontSize: 18,
          color: lightTextSecondary,
        ),
        bodyMedium: TextStyle(
          fontSize: 16,
          color: lightTextSecondary,
        ),
        bodySmall: TextStyle(
          fontSize: 14,
          color: lightTextMuted,
        ),
      ),
    );
  }

  // Light mode gradients
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
}
