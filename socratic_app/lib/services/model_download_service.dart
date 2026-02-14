import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

enum DownloadStatus { notStarted, connecting, downloading, completed, error }

class ModelDownloadService extends ChangeNotifier {
  static final ModelDownloadService _instance = ModelDownloadService._internal();
  factory ModelDownloadService() => _instance;
  ModelDownloadService._internal();

  final Dio _dio = Dio();
  DownloadStatus _status = DownloadStatus.notStarted;
  double _progress = 0.0;
  String? _errorMessage;
  CancelToken? _cancelToken;

  DownloadStatus get status => _status;
  double get progress => _progress;
  String? get errorMessage => _errorMessage;

  /// Check if the model already exists locally
  Future<bool> isModelDownloaded(String fileName) async {
    final directory = await getApplicationSupportDirectory();
    final file = File(p.join(directory.path, fileName));
    return await file.exists();
  }

  /// Start downloading the model
  Future<void> downloadModel(String url, String fileName) async {
    if (_status == DownloadStatus.downloading || _status == DownloadStatus.connecting) return;

    _status = DownloadStatus.connecting;
    _progress = 0.0;
    _errorMessage = null;
    _cancelToken = CancelToken();
    notifyListeners();

    try {
      // Step 0: Early exit for x86 architecture (incompatible with the native engine)
      final isX86 = Platform.isAndroid && 
         (Platform.version.toLowerCase().contains('x86') || 
          Platform.operatingSystemVersion.toLowerCase().contains('x86'));

      if (isX86) {
         _status = DownloadStatus.error;
         _errorMessage = 'Hardware Incompatible: Engine requires ARM64 device.';
         notifyListeners();
         return;
      }

      // Use a timeout for the directory fetch to avoid hangs
      final directory = await getApplicationSupportDirectory().timeout(
        const Duration(seconds: 5),
        onTimeout: () => throw Exception('Failed to get storage directory (Timeout)'),
      );
      
      final savePath = p.join(directory.path, fileName);
      final tempPath = '$savePath.tmp';

      debugPrint('DownloadService: Starting download from $url');
      
      // Ensure directory exists
      final saveDir = Directory(p.dirname(savePath));
      if (!await saveDir.exists()) {
        await saveDir.create(recursive: true);
      }

      await _dio.download(
        url,
        tempPath,
        onReceiveProgress: (received, total) {
          if (_status != DownloadStatus.downloading) {
            _status = DownloadStatus.downloading;
          }
          if (total != -1 && total > 0) {
            _progress = received / total;
          } else {
            // If total is unknown, show some activity (0.01 per MB)
            _progress = (received / (1024 * 1024)) % 1.0; 
          }
          notifyListeners();
        },
        cancelToken: _cancelToken,
        options: Options(
          sendTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(minutes: 60),
          followRedirects: true,
          headers: {
            'User-Agent': 'Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
          },
          validateStatus: (status) => status != null && status < 500,
        ),
      );

      // Successfully downloaded, rename temp file
      final tempFile = File(tempPath);
      if (await tempFile.exists()) {
        // Remove existing file if any
        final existingFile = File(savePath);
        if (await existingFile.exists()) {
          await existingFile.delete();
        }
        await tempFile.rename(savePath);
      }

      _status = DownloadStatus.completed;
      notifyListeners();
    } catch (e) {
      if (!CancelToken.isCancel(e as DioException)) {
        _status = DownloadStatus.error;
        _errorMessage = e is DioException ? e.message : e.toString();
        notifyListeners();
      }
    }
  }

  void cancelDownload() {
    _cancelToken?.cancel();
    _status = DownloadStatus.notStarted;
    notifyListeners();
  }
}
