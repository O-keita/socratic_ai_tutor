import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'theme/app_theme.dart';
import 'screens/onboarding_screen.dart';
import 'screens/home_screen.dart';
import 'screens/chat_screen.dart';
import 'services/hybrid_tutor_service.dart';
import 'services/theme_service.dart';
import 'services/model_download_service.dart';
import 'services/auth_service.dart';
import 'screens/auth_screen.dart';
import 'screens/splash_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize Hybrid Tutor service in the background
  Future.delayed(const Duration(seconds: 1), () {
    final hybridService = HybridTutorService();
    hybridService.initialize();
  });
  
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => ThemeService()),
        ChangeNotifierProvider(create: (_) => ModelDownloadService()),
        ChangeNotifierProvider(create: (_) => AuthService()),
      ],
      child: const SocraticTutorApp(),
    ),
  );
}

class SocraticTutorApp extends StatelessWidget {
  const SocraticTutorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ThemeService>(
      builder: (context, themeService, child) {
        // Update system UI based on theme
        final isDark = themeService.isDarkMode;
        SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle(
          statusBarColor: Colors.transparent,
          statusBarIconBrightness: isDark ? Brightness.light : Brightness.dark,
          systemNavigationBarColor: isDark ? AppTheme.surfaceDark : AppTheme.lightSurface,
          systemNavigationBarIconBrightness: isDark ? Brightness.light : Brightness.dark,
        ));

        return MaterialApp(
          title: 'Socratic AI Tutor',
          debugShowCheckedModeBanner: false,
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: themeService.themeMode,
          home: const SplashScreen(),
          routes: {
            '/home': (context) => const HomeScreen(),
            '/chat': (context) => ChatScreen(
              initialTopic: ModalRoute.of(context)?.settings.arguments as String?,
            ),
          },
        );
      },
    );
  }
}
