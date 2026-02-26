import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/theme_service.dart';
import '../services/hybrid_tutor_service.dart';
import '../services/model_download_service.dart';
import '../theme/app_theme.dart';
import '../utils/app_snackbar.dart';
import 'model_setup_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final HybridTutorService _hybridService = HybridTutorService();

  @override
  Widget build(BuildContext context) {
    final themeService = context.watch<ThemeService>();
    final isDark = themeService.isDarkMode;
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      decoration: BoxDecoration(
        gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
      ),
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
                Consumer<ModelDownloadService>(
                  builder: (context, dl, _) {
                    final isDownloading =
                        dl.status == DownloadStatus.downloading ||
                        dl.status == DownloadStatus.connecting;
                    final isComplete = dl.status == DownloadStatus.completed;

                    final subtitle = isComplete
                        ? '✓ Model ready — tap to manage'
                        : isDownloading
                            ? 'Downloading… ${(dl.progress * 100).toInt()}%'
                            : dl.status == DownloadStatus.error
                                ? 'Download failed — tap to retry'
                                : 'Download or update the offline Socratic engine';

                    return Card(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          ListTile(
                            title: const Text('Manage Local Model'),
                            subtitle: Text(subtitle),
                            leading: Icon(
                              isComplete
                                  ? Icons.check_circle_outline
                                  : Icons.download_for_offline,
                              color: isComplete ? AppTheme.success : null,
                            ),
                            onTap: () => Navigator.of(context).push(
                              MaterialPageRoute(
                                  builder: (_) => const ModelSetupScreen()),
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
                _buildSectionHeader('General'),
                _buildSettingsTile(
                  'Reset Model',
                  'Re-initialize local engine',
                  Icons.refresh,
                  onTap: () async {
                    final ok = await _hybridService.initialize();
                    if (context.mounted) {
                      if (ok) {
                        AppSnackBar.success(context, 'Model re-initialized successfully!');
                      } else {
                        AppSnackBar.error(context, 'Initialization failed. Check the model file.');
                      }
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader(BuildContext context, ColorScheme colorScheme, bool isDark) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      child: Row(
        children: [
          Text(
            'Settings',
            style: TextStyle(
              color: colorScheme.onSurface,
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: _hybridService.currentStatus == EngineStatus.online
                  ? Colors.green.withValues(alpha: 0.2)
                  : Colors.orange.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  _hybridService.currentStatus == EngineStatus.online
                      ? Icons.cloud_done
                      : Icons.cloud_off,
                  size: 16,
                  color: _hybridService.currentStatus == EngineStatus.online
                      ? Colors.green
                      : Colors.orange,
                ),
                const SizedBox(width: 4),
                Text(
                  _hybridService.currentStatus == EngineStatus.online ? 'Online' : 'Offline',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: _hybridService.currentStatus == EngineStatus.online
                        ? Colors.green
                        : Colors.orange,
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
