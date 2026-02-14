import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ThemeService extends ChangeNotifier {
  static const String _themeKey = 'theme_mode';
  ThemeMode _themeMode = ThemeMode.light;

  ThemeMode get themeMode => _themeMode;
  bool get isDarkMode => _themeMode == ThemeMode.dark;

  ThemeService() {
    _loadTheme();
  }

  Future<void> _loadTheme() async {
    try {
      // Retry logic for emulator startup
      SharedPreferences? prefs;
      for (int i = 0; i < 5; i++) {
        try {
          prefs = await SharedPreferences.getInstance();
          break;
        } catch (e) {
          if (i == 4) return; // Give up and use default
          await Future.delayed(Duration(seconds: 1));
        }
      }
      
      if (prefs != null) {
        // Default to false (light mode) instead of true
        final isDark = prefs.getBool(_themeKey) ?? false;
        _themeMode = isDark ? ThemeMode.dark : ThemeMode.light;
        notifyListeners();
      }
    } catch (e) {
      debugPrint('ThemeService: Error loading theme: $e');
    }
  }

  Future<void> toggleTheme() async {
    _themeMode = _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    notifyListeners();
    
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_themeKey, _themeMode == ThemeMode.dark);
    } catch (e) {
      debugPrint('ThemeService: Error saving theme: $e');
    }
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    _themeMode = mode;
    notifyListeners();
    
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_themeKey, mode == ThemeMode.dark);
    } catch (e) {
      debugPrint('ThemeService: Error saving theme: $e');
    }
  }
}
