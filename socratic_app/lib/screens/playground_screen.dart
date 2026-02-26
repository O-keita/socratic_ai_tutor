import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webview_flutter/webview_flutter.dart';
import '../theme/app_theme.dart';

class PlaygroundScreen extends StatefulWidget {
  /// When embedded inside HomeScreen via Offstage, this callback switches back
  /// to the previous tab instead of popping the route.
  final VoidCallback? onClose;

  const PlaygroundScreen({super.key, this.onClose});

  @override
  State<PlaygroundScreen> createState() => _PlaygroundScreenState();
}

class _PlaygroundScreenState extends State<PlaygroundScreen> {
  late final WebViewController _controller;

  /// True once the HTML page signals that Pyodide finished loading.
  bool _pyodideReady = false;

  /// Set to true if loading fails (shown in overlay).
  bool _loadError = false;

  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..addJavaScriptChannel(
        'Playground',
        onMessageReceived: (msg) {
          if (!mounted) return;
          setState(() {
            _pyodideReady = msg.message != 'error';
            _loadError = msg.message == 'error';
          });
        },
      );
    _loadHtml();
  }

  Future<void> _loadHtml() async {
    try {
      final html = await rootBundle.loadString('assets/playground/index.html');
      // baseUrl 'https://localhost' lets the page make CDN fetch() calls
      // without file:// origin restrictions on Android WebView.
      await _controller.loadHtmlString(html, baseUrl: 'https://localhost');
    } catch (e) {
      debugPrint('PlaygroundScreen: failed to load HTML — $e');
      if (mounted) setState(() => _loadError = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    _controller.setBackgroundColor(
      isDark ? const Color(0xFF1a1a2e) : Colors.white,
    );

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, size: 18),
          onPressed: () {
            if (widget.onClose != null) {
              widget.onClose!();
            } else {
              Navigator.pop(context);
            }
          },
        ),
        title: Row(
          children: [
            const Text('Python Playground'),
            if (!_pyodideReady && !_loadError) ...[
              const SizedBox(width: 10),
              const SizedBox(
                width: 14,
                height: 14,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: AppTheme.accentOrange,
                ),
              ),
            ],
          ],
        ),
      ),
      body: Stack(
        children: [
          // WebView is always present so the page loads immediately
          WebViewWidget(controller: _controller),

          // Native loading overlay — shown until Pyodide signals ready
          if (!_pyodideReady)
            _LoadingOverlay(error: _loadError, onRetry: _loadError ? _retry : null),
        ],
      ),
    );
  }

  Future<void> _retry() async {
    setState(() {
      _pyodideReady = false;
      _loadError = false;
    });
    await _loadHtml();
  }
}

// ── Native loading overlay ─────────────────────────────────────────────────

class _LoadingOverlay extends StatelessWidget {
  final bool error;
  final VoidCallback? onRetry;

  const _LoadingOverlay({required this.error, this.onRetry});

  @override
  Widget build(BuildContext context) {
    return AnimatedOpacity(
      opacity: 1.0,
      duration: const Duration(milliseconds: 300),
      child: Container(
        color: const Color(0xFF1a1a2e),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (!error)
                const SizedBox(
                  width: 44,
                  height: 44,
                  child: CircularProgressIndicator(
                    strokeWidth: 3,
                    color: AppTheme.accentOrange,
                  ),
                )
              else
                const Icon(Icons.wifi_off_rounded, size: 44, color: Colors.redAccent),
              const SizedBox(height: 20),
              Text(
                error ? 'Failed to load Python runtime' : 'Loading Python runtime…',
                style: const TextStyle(
                  color: Color(0xFFe2e8f0),
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                error
                    ? 'Check your internet connection and try again.'
                    : 'First load fetches ~8 MB — subsequent opens are instant',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Color(0xFF64748b),
                  fontSize: 12,
                ),
              ),
              if (error && onRetry != null) ...[
                const SizedBox(height: 24),
                ElevatedButton.icon(
                  onPressed: onRetry,
                  icon: const Icon(Icons.refresh, size: 18),
                  label: const Text('Retry'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.accentOrange,
                    foregroundColor: Colors.white,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
