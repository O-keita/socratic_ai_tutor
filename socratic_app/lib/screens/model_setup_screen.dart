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
  // Must match SocraticLlmService.modelFileName
  static const String modelFileName = 'socratic-model.gguf';
  // Direct download from HuggingFace (resolve/main/ gives the raw file)
  static const String downloadUrl =
      'https://huggingface.co/Omar-keita/DSML-Socatic-qwen3-0.6B/resolve/main/Socratic-Qwen3-0.6-Merged-Quality_Data-752M-Q4_K_M%20(1).gguf';

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
    // Show tip for all users - the app works on all devices
    // Local engine is optional, online mode works everywhere
    setState(() {
      _isLikelyEmulator = false; // Don't exclude any devices
    });
  }

  Future<void> _checkStatus() async {
    final downloader = ModelDownloadService();
    final exists = await downloader.isModelDownloaded(modelFileName);
    if (exists && mounted) {
      _navigateToHome();
      return;
    }
    // Update hasPartialDownload so button label shows "Resume" if applicable
    await downloader.checkPartialDownload(modelFileName);
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
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: Navigator.of(context).canPop()
            ? IconButton(
                icon: Icon(Icons.arrow_back, color: isDark ? Colors.white : AppTheme.darkBlue),
                onPressed: () => Navigator.of(context).pop(),
              )
            : null,
      ),
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
                      color: Colors.amber.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.amber.withValues(alpha: 0.3)),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.tips_and_updates, color: Colors.amber, size: 20),
                        SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            'All devices can use the tutor online. The local engine is optimized for mobile hardware.',
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
                if (downloader.status == DownloadStatus.notStarted) ...[
                  _buildDownloadButton(downloader),
                  if (downloader.hasPartialDownload) ...[
                    const SizedBox(height: 10),
                    _buildStartFreshButton(downloader),
                  ],
                  const SizedBox(height: 16),
                  _buildSkipButton(),
                ] else if (downloader.status == DownloadStatus.connecting) ...[
                  const CircularProgressIndicator(color: AppTheme.accentOrange),
                  const SizedBox(height: 16),
                  Text(
                    downloader.hasPartialDownload
                        ? 'Reconnecting to resume…'
                        : 'Connecting to server…',
                    style: const TextStyle(
                        color: AppTheme.accentOrange, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 32),
                  TextButton(
                    onPressed: () => downloader.cancelDownload(),
                    child: const Text('Cancel',
                        style: TextStyle(color: Colors.redAccent)),
                  ),
                ] else if (downloader.status == DownloadStatus.downloading) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: downloader.progress,
                      backgroundColor: Colors.grey.withValues(alpha: 0.2),
                      valueColor: const AlwaysStoppedAnimation<Color>(
                          AppTheme.accentOrange),
                      minHeight: 6,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    '${(downloader.progress * 100).toStringAsFixed(1)}%',
                    style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: AppTheme.accentOrange),
                  ),
                  const SizedBox(height: 32),
                  TextButton(
                    onPressed: () => downloader.cancelDownload(),
                    child: const Text('Cancel Download',
                        style: TextStyle(color: Colors.redAccent)),
                  ),
                ] else if (downloader.status == DownloadStatus.error) ...[
                  _buildErrorCard(
                      downloader.errorMessage ?? 'An unexpected error occurred.'),
                  const SizedBox(height: 24),
                  _buildDownloadButton(downloader),
                  if (downloader.hasPartialDownload) ...[
                    const SizedBox(height: 10),
                    _buildStartFreshButton(downloader),
                  ],
                  const SizedBox(height: 16),
                  _buildSkipButton(),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildErrorCard(String message) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFEF4444).withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFEF4444).withValues(alpha: 0.35)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.error_outline_rounded,
              color: Color(0xFFEF4444), size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Download failed',
                  style: TextStyle(
                    color: Color(0xFFEF4444),
                    fontWeight: FontWeight.w600,
                    fontSize: 13.5,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  message,
                  style: const TextStyle(
                    color: Color(0xFFEF4444),
                    fontSize: 12.5,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSkipButton() {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton(
        onPressed: _navigateToHome,
        style: OutlinedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16),
          side: BorderSide(color: AppTheme.accentOrange.withValues(alpha: 0.5)),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
        child: const Text(
          'Skip & Use Online Mode',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: AppTheme.accentOrange,
          ),
        ),
      ),
    );
  }

  Widget _buildStartFreshButton(ModelDownloadService downloader) {
    return SizedBox(
      width: double.infinity,
      child: TextButton.icon(
        onPressed: () async {
          await downloader.clearPartialDownload(modelFileName);
        },
        icon: const Icon(Icons.delete_outline_rounded,
            size: 16, color: Colors.redAccent),
        label: const Text(
          'Start fresh (discard partial download)',
          style: TextStyle(fontSize: 13, color: Colors.redAccent),
        ),
      ),
    );
  }

  Widget _buildDownloadButton(ModelDownloadService downloader) {
    final isResume = downloader.hasPartialDownload;
    return SizedBox(
      width: double.infinity,
      child: Container(
        decoration: BoxDecoration(
          gradient: AppTheme.buttonGradient,
          borderRadius: BorderRadius.circular(16),
        ),
        child: ElevatedButton.icon(
          onPressed: () => downloader.downloadModel(downloadUrl, modelFileName),
          icon: Icon(
            isResume ? Icons.play_arrow_rounded : Icons.download_rounded,
            color: Colors.white,
            size: 20,
          ),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.transparent,
            shadowColor: Colors.transparent,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          ),
          label: Text(
            isResume ? 'Resume Download' : 'Download Engine (~350 MB)',
            style: const TextStyle(
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
