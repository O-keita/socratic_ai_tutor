import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/model_download_service.dart';
import '../services/model_version_service.dart';
import '../theme/app_theme.dart';
import '../services/theme_service.dart';
import '../utils/app_config.dart';
import 'home_screen.dart';

class ModelSetupScreen extends StatefulWidget {
  const ModelSetupScreen({super.key});

  @override
  State<ModelSetupScreen> createState() => _ModelSetupScreenState();
}

class _ModelSetupScreenState extends State<ModelSetupScreen> {
  static String get modelFileName => AppConfig.modelFileName;
  static String get downloadUrl => AppConfig.modelDownloadUrl;

  bool _isLikelyEmulator = false;
  bool _showUpdateUI = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) { _checkEmulator(); _checkStatus(); }
    });
  }

  void _checkEmulator() => setState(() { _isLikelyEmulator = false; });

  Future<void> _checkStatus() async {
    final downloader = ModelDownloadService();
    final exists = await downloader.isModelDownloaded(modelFileName);
    if (exists && mounted) {
      // If an update is available, show the update UI instead of going home
      final versionService = ModelVersionService();
      if (versionService.updateAvailable) {
        setState(() => _showUpdateUI = true);
        return;
      }
      _navigateToHome();
      return;
    }
    await downloader.checkPartialDownload(modelFileName);
  }

  void _navigateToHome() {
    Navigator.of(context).pushReplacement(MaterialPageRoute(builder: (_) => const HomeScreen()));
  }

  @override
  Widget build(BuildContext context) {
    final isDark = context.watch<ThemeService>().isDarkMode;
    final downloader = context.watch<ModelDownloadService>();
    final bgColor = isDark ? AppTheme.primaryDark : const Color(0xFFFEF6EE);

    if (downloader.status == DownloadStatus.completed) {
      WidgetsBinding.instance.addPostFrameCallback((_) { if (mounted) _navigateToHome(); });
    }

    return Scaffold(
      body: Container(
        width: double.infinity,
        color: bgColor,
        child: SafeArea(
          child: Column(
            children: [
              // Back button
              if (Navigator.of(context).canPop())
                Align(
                  alignment: Alignment.topLeft,
                  child: Padding(
                    padding: const EdgeInsets.all(8),
                    child: IconButton(
                      icon: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(color: isDark ? AppTheme.surfaceCard : Colors.white, shape: BoxShape.circle),
                        child: Icon(Icons.arrow_back_ios_new, size: 16, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary),
                      ),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ),
                ),

              Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 40),
                  child: _showUpdateUI
                      ? _buildUpdateUI(isDark, downloader)
                      : _buildDownloadUI(isDark, downloader),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Update UI (model exists + update available) ──────────────────────────
  Widget _buildUpdateUI(bool isDark, ModelDownloadService downloader) {
    final versionService = context.watch<ModelVersionService>();
    final displayName = versionService.latestDisplayName ?? AppConfig.modelDisplayName;
    final newVersion = versionService.latestVersion ?? '?';
    final notes = versionService.releaseNotes;
    final updateUrl = versionService.latestDownloadUrl ?? downloadUrl;

    // While downloading, show progress instead of the update card
    if (downloader.status == DownloadStatus.downloading ||
        downloader.status == DownloadStatus.connecting) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              color: AppTheme.accentOrange.withValues(alpha: 0.12),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.system_update, size: 52, color: AppTheme.accentOrange),
          ),
          const SizedBox(height: 28),
          Text('Updating to v$newVersion', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
          const SizedBox(height: 24),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: downloader.progress > 0 ? downloader.progress : null,
              backgroundColor: AppTheme.accentOrange.withValues(alpha: 0.15),
              valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange),
              minHeight: 8,
            ),
          ),
          const SizedBox(height: 12),
          if (downloader.progress > 0)
            Text('${(downloader.progress * 100).toStringAsFixed(1)}%', style: const TextStyle(fontWeight: FontWeight.w700, color: AppTheme.accentOrange, fontSize: 18)),
          const SizedBox(height: 32),
          TextButton(
            onPressed: () => downloader.cancelDownload(),
            child: const Text('Cancel', style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      );
    }

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // Icon
        Container(
          width: 100,
          height: 100,
          decoration: BoxDecoration(
            color: AppTheme.accentOrange.withValues(alpha: 0.12),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.system_update, size: 52, color: AppTheme.accentOrange),
        ),
        const SizedBox(height: 28),

        Text(
          'Model Update Available',
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary),
        ),
        const SizedBox(height: 16),

        // Version comparison
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: isDark ? AppTheme.surfaceCard : Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppTheme.accentOrange.withValues(alpha: 0.3)),
          ),
          child: Column(
            children: [
              Text(displayName, style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary)),
              const SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  FutureBuilder<String?>(
                    future: versionService.getInstalledVersion(),
                    builder: (context, snap) {
                      final current = snap.data ?? '?';
                      return Text('v$current', style: TextStyle(fontSize: 14, color: isDark ? AppTheme.textMuted : AppTheme.lightTextMuted));
                    },
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 10),
                    child: Icon(Icons.arrow_forward_rounded, size: 18, color: AppTheme.accentOrange),
                  ),
                  Text('v$newVersion', style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: AppTheme.accentOrange)),
                ],
              ),
            ],
          ),
        ),

        // Release notes
        if (notes != null && notes.isNotEmpty) ...[
          const SizedBox(height: 12),
          Text(
            notes,
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 13, color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary, height: 1.4),
          ),
        ],

        if (downloader.status == DownloadStatus.error) ...[
          const SizedBox(height: 16),
          _buildErrorCard(_shortenError(downloader.errorMessage)),
        ],

        const SizedBox(height: 28),

        // Update button
        SizedBox(
          width: double.infinity,
          child: Container(
            decoration: BoxDecoration(gradient: AppTheme.buttonGradient, borderRadius: BorderRadius.circular(18)),
            child: ElevatedButton.icon(
              onPressed: () => downloader.downloadModel(updateUrl, modelFileName),
              icon: const Icon(Icons.download_rounded, color: Colors.white, size: 20),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.transparent,
                shadowColor: Colors.transparent,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
              ),
              label: const Text('Update Model', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Colors.white)),
            ),
          ),
        ),
        const SizedBox(height: 12),

        // Keep current button
        SizedBox(
          width: double.infinity,
          child: OutlinedButton(
            onPressed: () => Navigator.of(context).pop(),
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              side: BorderSide(color: AppTheme.accentOrange.withValues(alpha: 0.5)),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
            ),
            child: const Text('Keep Current', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: AppTheme.accentOrange)),
          ),
        ),
      ],
    );
  }

  // ── First-time download UI ──────────────────────────────────────────────
  Widget _buildDownloadUI(bool isDark, ModelDownloadService downloader) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // Icon
        Container(
          width: 100,
          height: 100,
          decoration: BoxDecoration(
            color: AppTheme.accentOrange.withValues(alpha: 0.12),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.download_for_offline_rounded, size: 52, color: AppTheme.accentOrange),
        ),
        const SizedBox(height: 28),

        Text(
          'Local Intelligence',
          style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: isDark ? AppTheme.textPrimary : AppTheme.lightTextPrimary),
        ),
        const SizedBox(height: 14),

        if (_isLikelyEmulator) ...[
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(color: AppTheme.warning.withValues(alpha: 0.1), borderRadius: BorderRadius.circular(14), border: Border.all(color: AppTheme.warning.withValues(alpha: 0.3))),
            child: Row(children: [
              const Icon(Icons.tips_and_updates, color: AppTheme.warning, size: 20),
              const SizedBox(width: 12),
              Expanded(child: Text('All devices can use the tutor online. The local engine is optimized for mobile hardware.', style: TextStyle(fontSize: 12, color: AppTheme.warning, fontWeight: FontWeight.w500))),
            ]),
          ),
          const SizedBox(height: 16),
        ],

        Text(
          'To learn offline, we need to set up the ${AppConfig.modelDisplayName} engine on your device. This is a one-time setup.',
          textAlign: TextAlign.center,
          style: TextStyle(color: isDark ? AppTheme.textSecondary : AppTheme.lightTextSecondary, fontSize: 14, height: 1.5),
        ),
        const SizedBox(height: 40),

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
          Text(downloader.hasPartialDownload ? 'Reconnecting to resume...' : 'Connecting to server...', style: const TextStyle(color: AppTheme.accentOrange, fontWeight: FontWeight.w600)),
          const SizedBox(height: 32),
          TextButton(onPressed: () => downloader.cancelDownload(), child: const Text('Cancel', style: TextStyle(color: Colors.redAccent))),
        ] else if (downloader.status == DownloadStatus.downloading) ...[
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(value: downloader.progress, backgroundColor: AppTheme.accentOrange.withValues(alpha: 0.15), valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.accentOrange), minHeight: 8),
          ),
          const SizedBox(height: 12),
          Text('${(downloader.progress * 100).toStringAsFixed(1)}%', style: const TextStyle(fontWeight: FontWeight.w700, color: AppTheme.accentOrange, fontSize: 18)),
          const SizedBox(height: 32),
          TextButton(onPressed: () => downloader.cancelDownload(), child: const Text('Cancel Download', style: TextStyle(color: Colors.redAccent))),
        ] else if (downloader.status == DownloadStatus.error) ...[
          _buildErrorCard(_shortenError(downloader.errorMessage)),
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
    );
  }

  /// Shorten a raw error/exception string into a user-friendly one-liner.
  String _shortenError(String? raw) {
    if (raw == null || raw.isEmpty) return 'Download failed. Check your connection and try again.';
    // Strip exception class prefixes
    var msg = raw.replaceAll(RegExp(r'^(Exception:\s*)+'), '');
    // Take only the first line
    msg = msg.split('\n').first.trim();
    // Cap length
    if (msg.length > 120) msg = '${msg.substring(0, 117)}...';
    return msg.isEmpty ? 'Download failed. Check your connection and try again.' : msg;
  }

  Widget _buildErrorCard(String message) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.error.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.error.withValues(alpha: 0.35)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline_rounded, color: AppTheme.error, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(color: AppTheme.error, fontSize: 13, fontWeight: FontWeight.w500),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
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
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        ),
        child: const Text('Skip & Use Online Mode', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: AppTheme.accentOrange)),
      ),
    );
  }

  Widget _buildStartFreshButton(ModelDownloadService downloader) {
    return SizedBox(
      width: double.infinity,
      child: TextButton.icon(
        onPressed: () async { await downloader.clearPartialDownload(modelFileName); },
        icon: const Icon(Icons.delete_outline_rounded, size: 16, color: Colors.redAccent),
        label: const Text('Start fresh (discard partial download)', style: TextStyle(fontSize: 13, color: Colors.redAccent)),
      ),
    );
  }

  Widget _buildDownloadButton(ModelDownloadService downloader) {
    final isResume = downloader.hasPartialDownload;
    return SizedBox(
      width: double.infinity,
      child: Container(
        decoration: BoxDecoration(gradient: AppTheme.buttonGradient, borderRadius: BorderRadius.circular(18)),
        child: ElevatedButton.icon(
          onPressed: () => downloader.downloadModel(downloadUrl, modelFileName),
          icon: Icon(isResume ? Icons.play_arrow_rounded : Icons.download_rounded, color: Colors.white, size: 20),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.transparent,
            shadowColor: Colors.transparent,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          ),
          label: Text(isResume ? 'Resume Download' : 'Download Engine (~230 MB)', style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Colors.white)),
        ),
      ),
    );
  }
}
