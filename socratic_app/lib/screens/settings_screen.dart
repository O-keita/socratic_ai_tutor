import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/theme_service.dart';
import '../services/hybrid_tutor_service.dart';
import '../theme/app_theme.dart';

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

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: isDark ? AppTheme.backgroundGradient : AppTheme.lightBackgroundGradient,
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildAppBar(context, colorScheme),
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
                    _buildSectionHeader('General'),
                    _buildSettingsTile(
                      'Reset Model',
                      'Re-initialize local engine',
                      Icons.refresh,
                      onTap: () async {
                        final ok = await _hybridService.initialize();
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(content: Text(ok ? 'Model re-initialized!' : 'Initialization failed')),
                          );
                        }
                      },
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAppBar(BuildContext context, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_new, size: 20),
            onPressed: () => Navigator.pop(context),
            color: colorScheme.onSurface,
          ),
          Text(
            'Settings',
            style: TextStyle(
              color: colorScheme.onSurface,
              fontSize: 20,
              fontWeight: FontWeight.bold,
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
