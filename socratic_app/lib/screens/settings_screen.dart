import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/theme_service.dart';
import '../services/hybrid_tutor_service.dart';
import '../services/llm_service.dart';
import '../services/model_download_service.dart';
import '../services/model_version_service.dart';
import '../services/session_service.dart';
import '../services/progress_service.dart';
import '../theme/app_theme.dart';
import '../utils/app_snackbar.dart';
import 'model_setup_screen.dart';
import 'eula_screen.dart';
import 'about_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final HybridTutorService _hybridService = HybridTutorService();
  bool _privacyMode = false;

  static const String _privacyModeKey = 'privacy_mode_enabled';

  @override
  void initState() {
    super.initState();
    _loadPrivacyMode();
  }

  Future<void> _loadPrivacyMode() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _privacyMode = prefs.getBool(_privacyModeKey) ?? false;
    });
  }

  Future<void> _togglePrivacyMode(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_privacyModeKey, value);
    setState(() {
      _privacyMode = value;
    });
  }

  Future<void> _clearAllData() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear All Data?'),
        content: const Text(
          'This will permanently delete all your conversation history, '
          'session logs, and lesson progress. This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            style: TextButton.styleFrom(foregroundColor: AppTheme.error),
            child: const Text('Delete Everything'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    // Clear sessions
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('chat_sessions');

    // Clear lesson progress
    await ProgressService.resetProgress();

    if (mounted) {
      AppSnackBar.success(context, 'All data cleared successfully.');
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeService = context.watch<ThemeService>();
    final isDark = themeService.isDarkMode;
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      color: isDark ? AppTheme.primaryDark : const Color(0xFFFEF6EE),
      child: Column(
        children: [
          _buildHeader(context, colorScheme, isDark),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(16),
              children: [
                _buildSectionHeader('Intelligence Engine'),
                _buildModeTile(
                  'Auto (Network-Aware)',
                  'Smart switching between Online/Offline',
                  TutorMode.auto,
                  Icons.auto_awesome,
                ),
                _buildModeTile(
                  'Force Online',
                  'Always use Remote Qwen3-0.6B',
                  TutorMode.online,
                  Icons.cloud_outlined,
                ),
                _buildModeTile(
                  'Force Offline',
                  'Always use Local GGUF',
                  TutorMode.offline,
                  Icons.sd_storage_outlined,
                ),
                const SizedBox(height: 24),
                _buildSectionHeader('Appearance'),
                Card(
                  child: SwitchListTile(
                    title: const Text('Dark Mode'),
                    subtitle: Text(isDark ? 'Professional Dark Slate' : 'Clean Light Mode'),
                    value: isDark,
                    onChanged: (value) => themeService.toggleTheme(),
                    secondary: Icon(isDark ? Icons.dark_mode : Icons.light_mode),
                  ),
                ),
                const SizedBox(height: 24),
                _buildSectionHeader('Offline Capabilities'),
                Consumer2<ModelDownloadService, ModelVersionService>(
                  builder: (context, dl, versionService, _) {
                    final isDownloading =
                        dl.status == DownloadStatus.downloading ||
                        dl.status == DownloadStatus.connecting;
                    final isComplete = dl.status == DownloadStatus.completed;
                    final hasUpdate = versionService.updateAvailable;

                    final String subtitle;
                    if (hasUpdate && isComplete) {
                      final name = versionService.latestDisplayName ?? 'Model';
                      final ver = versionService.latestVersion ?? '';
                      subtitle = 'Update available: $name v$ver';
                    } else if (isComplete) {
                      subtitle = '✓ Model ready — tap to manage';
                    } else if (isDownloading) {
                      subtitle = 'Downloading… ${(dl.progress * 100).toInt()}%';
                    } else if (dl.status == DownloadStatus.error) {
                      subtitle = 'Download failed — tap to retry';
                    } else {
                      subtitle = 'Download or update the offline Socratic engine';
                    }

                    return Card(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          ListTile(
                            title: const Text('Manage Local Model'),
                            subtitle: Text(subtitle),
                            leading: Icon(
                              hasUpdate && isComplete
                                  ? Icons.system_update
                                  : isComplete
                                      ? Icons.check_circle_outline
                                      : Icons.download_for_offline,
                              color: hasUpdate && isComplete
                                  ? AppTheme.accentOrange
                                  : isComplete
                                      ? AppTheme.success
                                      : null,
                            ),
                            onTap: () => Navigator.of(context).push(
                              MaterialPageRoute(
                                  builder: (_) => const ModelSetupScreen(fromSettings: true)),
                            ),
                          ),
                          if (isDownloading)
                            Padding(
                              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(2),
                                child: LinearProgressIndicator(
                                  value: dl.progress > 0 ? dl.progress : null,
                                  backgroundColor: AppTheme.accentOrange
                                      .withValues(alpha: 0.15),
                                  valueColor:
                                      const AlwaysStoppedAnimation<Color>(
                                          AppTheme.accentOrange),
                                  minHeight: 3,
                                ),
                              ),
                            ),
                        ],
                      ),
                    );
                  },
                ),
                const SizedBox(height: 24),
                _buildSectionHeader('Privacy'),
                Card(
                  child: SwitchListTile(
                    title: const Text('Privacy Mode'),
                    subtitle: Text(
                      _privacyMode
                          ? 'Conversations are not saved'
                          : 'Conversations are saved locally',
                    ),
                    value: _privacyMode,
                    onChanged: _togglePrivacyMode,
                    secondary: Icon(
                      _privacyMode
                          ? Icons.visibility_off_outlined
                          : Icons.visibility_outlined,
                    ),
                    activeColor: AppTheme.accentOrange,
                  ),
                ),
                const SizedBox(height: 24),
                _buildSectionHeader('Data'),
                _buildSettingsTile(
                  'Clear All Data',
                  'Delete all conversations and progress',
                  Icons.delete_forever_outlined,
                  onTap: _clearAllData,
                ),
                const SizedBox(height: 24),
                _buildSectionHeader('General'),
                _buildSettingsTile(
                  'Reset Model',
                  'Clear crash flag and re-initialize engine',
                  Icons.refresh,
                  onTap: () async {
                    // Clear both the SharedPreferences crash flag AND in-memory
                    // failure state so initialize() can proceed again.
                    final llmService = SocraticLlmService();
                    await llmService.clearCrashFlag();
                    await llmService.clearInferenceCrashFlag();
                    llmService.resetInitializationFailure();

                    final ok = await _hybridService.initialize();
                    if (context.mounted) {
                      if (ok) {
                        AppSnackBar.success(context, 'Engine re-initialized successfully!');
                      } else {
                        AppSnackBar.error(
                          context,
                          'Initialization failed. Try switching to "Force Online" mode.',
                        );
                      }
                    }
                  },
                ),
                const SizedBox(height: 24),
                _buildSectionHeader('Legal & Info'),
                _buildSettingsTile(
                  'EULA & Privacy Policy',
                  'Review terms of use and data practices',
                  Icons.shield_outlined,
                  onTap: () => Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const EulaScreen()),
                  ),
                ),
                _buildSettingsTile(
                  'About Bantaba AI',
                  'Learn about the system and its limitations',
                  Icons.info_outline,
                  onTap: () => Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const AboutScreen()),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader(BuildContext context, ColorScheme colorScheme, bool isDark) {
    final isOnline = _hybridService.currentStatus == EngineStatus.online;
    final statusColor = isOnline ? AppTheme.success : AppTheme.warning;

    return Container(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
      decoration: BoxDecoration(
        gradient: isDark
            ? AppTheme.headerGradientDark
            : AppTheme.headerGradientLight,
        borderRadius: const BorderRadius.only(
          bottomLeft: Radius.circular(32),
          bottomRight: Radius.circular(32),
        ),
      ),
      child: Row(
        children: [
          Text(
            'Settings',
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: statusColor.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: statusColor.withValues(alpha: 0.3)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 6,
                  height: 6,
                  decoration: BoxDecoration(shape: BoxShape.circle, color: statusColor),
                ),
                const SizedBox(width: 6),
                Text(
                  isOnline ? 'Online' : 'Offline',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: statusColor,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(left: 4, bottom: 8),
      child: Text(
        title.toUpperCase(),
        style: const TextStyle(
          color: AppTheme.accentOrange,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          letterSpacing: 1.2,
        ),
      ),
    );
  }

  Widget _buildModeTile(String title, String subtitle, TutorMode mode, IconData icon) {
    final isSelected = _hybridService.mode == mode;
    return Card(
      child: RadioListTile<TutorMode>(
        title: Text(title),
        subtitle: Text(subtitle),
        value: mode,
        groupValue: _hybridService.mode,
        onChanged: (TutorMode? value) {
          if (value != null) {
            setState(() {
              _hybridService.setMode(value);
            });
          }
        },
        secondary: Icon(icon, color: isSelected ? AppTheme.accentOrange : null),
        activeColor: AppTheme.accentOrange,
      ),
    );
  }

  Widget _buildSettingsTile(String title, String subtitle, IconData icon, {VoidCallback? onTap}) {
    return Card(
      child: ListTile(
        title: Text(title),
        subtitle: Text(subtitle),
        leading: Icon(icon),
        onTap: onTap,
      ),
    );
  }
}
