import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'theme/app_theme.dart';
import 'screens/home_screen.dart';
import 'screens/chat_screen.dart';
import 'services/theme_service.dart';
import 'services/model_download_service.dart';
import 'services/llm_service.dart';
import 'services/auth_service.dart';
import 'services/model_version_service.dart';
import 'utils/app_config.dart';
import 'screens/splash_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Refresh model download status from filesystem so the UI starts with the
  // correct state (in-memory DownloadStatus resets to notStarted on restart).
  // Must complete BEFORE runApp so widgets see the real status immediately.
  await ModelDownloadService()
      .refreshStatus(SocraticLlmService.modelFileName);

  // Seed model version for existing users who downloaded the model
  // before version tracking was added.
  final modelExists = await ModelDownloadService()
      .isModelDownloaded(AppConfig.modelFileName);
  final versionService = ModelVersionService();
  final installedVersion = await versionService.getInstalledVersion();
  if (modelExists && installedVersion == null) {
    await versionService.setInstalledVersion(AppConfig.bundledModelVersion);
  }

  // Don't eagerly initialize HybridTutorService here — loading the local
  // GGUF model (~350-770 MB) on startup can OOM-kill budget devices and cause
  // a crash loop.  The engine initializes lazily on the first chat request
  // via _getBestEngine() in HybridTutorService.
  
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => ThemeService()),
        ChangeNotifierProvider(create: (_) => ModelDownloadService()),
        ChangeNotifierProvider(create: (_) => AuthService()),
        ChangeNotifierProvider(create: (_) => ModelVersionService()),
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
