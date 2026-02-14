import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/model_download_service.dart';
import '../theme/app_theme.dart';
import '../services/theme_service.dart';
import 'home_screen.dart';

class ModelSetupScreen extends StatefulWidget {
  const ModelSetupScreen({super.key});

  @override
  State<ModelSetupScreen> createState() => _ModelSetupScreenState();
}

class _ModelSetupScreenState extends State<ModelSetupScreen> {
  static const String modelFileName = 'socratic-q4_k_m.gguf';
  static const String downloadUrl = 'https://huggingface.co/Omar-keita/DSML-Socatic-qwen3-0.6B/resolve/main/socratic-q4_k_m.gguf';

  bool _isLikelyEmulator = false;

  @override
  void initState() {
    super.initState();
    // Use addPostFrameCallback to delay context-dependent checks until after first build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        _checkEmulator();
        _checkStatus();
      }
    });
  }

  void _checkEmulator() {
    // Check if we are on Android
    if (Theme.of(context).platform == TargetPlatform.android) {
        setState(() {
          _isLikelyEmulator = true; 
        });
    }
  }

  Future<void> _checkStatus() async {
    final downloader = ModelDownloadService();
    final exists = await downloader.isModelDownloaded(modelFileName);
    if (exists && mounted) {
      _navigateToHome();
    }
  }

  void _navigateToHome() {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (context) => const HomeScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;
    final downloader = context.watch<ModelDownloadService>();

    // Auto-navigate when download finishes
    if (downloader.status == DownloadStatus.completed) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _navigateToHome();
      });
    }

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 40),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.download_for_offline,
                  size: 80,
                  color: AppTheme.accentOrange,
                ),
                const SizedBox(height: 32),
                Text(
                  'Local Intelligence',
                  style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: isDark ? Colors.white : AppTheme.darkBlue,
                      ),
                ),
                const SizedBox(height: 16),
                if (_isLikelyEmulator) ...[
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.amber.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.amber.withOpacity(0.3)),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.warning_amber_rounded, color: Colors.amber, size: 20),
                        SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            'X86 Emulator detected. Local engine requires ARM64. Please use Skip for now.',
                            style: TextStyle(fontSize: 12, color: Colors.amber, fontWeight: FontWeight.w500),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                ],
                Text(
                  'To learn offline, we need to set up the Qwen3-0.6B engine on your device. This is a one-time setup.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary,
                  ),
                ),
                const SizedBox(height: 48),
                if (downloader.status == DownloadStatus.connecting) ...[
                  const CircularProgressIndicator(color: AppTheme.accentOrange),
                  const SizedBox(height: 16),
                  const Text(
                    'Connecting to server...',
                    style: TextStyle(color: AppTheme.accentOrange, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 32),
                  TextButton(
                    onPressed: () => downloader.cancelDownload(),
                    child: const Text('Cancel', style: TextStyle(color: Colors.redAccent)),
                  ),
                ] else if (downloader.status == DownloadStatus.downloading) ...[
                  LinearProgressIndicator(
                    value: downloader.progress,
                    backgroundColor: Colors.grey.withOpacity(0.2),
                    valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    '${(downloader.progress * 100).toStringAsFixed(1)}%',
                    style: const TextStyle(fontWeight: FontWeight.bold, color: AppTheme.accentOrange),
                  ),
                  const SizedBox(height: 32),
                  TextButton(
                    onPressed: () => downloader.cancelDownload(),
                    child: const Text('Cancel Download', style: TextStyle(color: Colors.redAccent)),
                  ),
                ] else if (downloader.status == DownloadStatus.error) ...[
                  Text(
                    'Error: ${downloader.errorMessage}',
                    style: const TextStyle(color: Colors.redAccent),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 24),
                  _buildDownloadButton(downloader),
                ] else ...[
                  _buildDownloadButton(downloader),
                  const SizedBox(height: 16),
                  TextButton(
                    onPressed: _navigateToHome,
                    child: Text(
                      'Skip for now (Online only)',
                      style: TextStyle(
                        color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted,
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildDownloadButton(ModelDownloadService downloader) {
    return SizedBox(
      width: double.infinity,
      child: Container(
        decoration: BoxDecoration(
          gradient: AppTheme.buttonGradient,
          borderRadius: BorderRadius.circular(16),
        ),
        child: ElevatedButton(
          onPressed: () => downloader.downloadModel(downloadUrl, modelFileName),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.transparent,
            shadowColor: Colors.transparent,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          ),
          child: const Text(
            'Download Engine (350MB)',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
        ),
      ),
    );
  }
}
