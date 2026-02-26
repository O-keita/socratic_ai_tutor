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

  /// Remembered from the last (or current) download so that when a retry
  /// starts we can show an accurate progress value immediately — before the
  /// first HTTP response comes back.
  int? _totalBytes;

  DownloadStatus get status => _status;
  double get progress => _progress;
  String? get errorMessage => _errorMessage;

  /// True when a partial `.tmp` file exists and the next download call will
  /// resume rather than restart.  Read by the UI to show "Resume" vs "Download".
  bool _hasPartial = false;
  bool get hasPartialDownload => _hasPartial;

  /// Check if the model already exists locally.
  Future<bool> isModelDownloaded(String fileName) async {
    try {
      final directory = await getApplicationSupportDirectory()
          .timeout(const Duration(seconds: 2));
      final file = File(p.join(directory.path, fileName));
      return await file.exists();
    } catch (_) {
      return false;
    }
  }

  /// Probe whether a partial `.tmp` file exists for [fileName].
  /// Call this on app start or after a cancel so the UI button label is correct.
  Future<void> checkPartialDownload(String fileName) async {
    try {
      final dir = await getApplicationSupportDirectory()
          .timeout(const Duration(seconds: 2));
      final tmpFile = File('${p.join(dir.path, fileName)}.tmp');
      _hasPartial = await tmpFile.exists() && await tmpFile.length() > 0;
    } catch (_) {
      _hasPartial = false;
    }
    notifyListeners();
  }

  /// Download (or resume) the model file.
  ///
  /// Uses HTTP `Range` requests so interrupted downloads continue from where
  /// they left off rather than starting over.
  ///
  /// Flow:
  ///   1. Read existing `.tmp` size → [existingBytes]
  ///   2. Send `Range: bytes=<existingBytes>-` if partial file exists
  ///   3. If server responds 206 → open file in **append** mode
  ///      If server responds 200 → server ignored Range, restart from 0
  ///   4. Stream bytes into the file, updating [_progress] per chunk
  ///   5. On completion rename `.tmp` → final file
  Future<void> downloadModel(String url, String fileName) async {
    if (_status == DownloadStatus.downloading ||
        _status == DownloadStatus.connecting) {
      return;
    }

    _status = DownloadStatus.connecting;
    _errorMessage = null;
    _cancelToken = CancelToken();
    notifyListeners();

    try {
      final directory = await getApplicationSupportDirectory().timeout(
        const Duration(seconds: 5),
        onTimeout: () =>
            throw Exception('Failed to get storage directory (timeout)'),
      );

      final savePath = p.join(directory.path, fileName);
      final tempPath = '$savePath.tmp';
      final tempFile = File(tempPath);

      // ── Detect existing partial download ──────────────────────────────────
      final existingBytes =
          await tempFile.exists() ? await tempFile.length() : 0;

      // Show an immediate progress estimate while waiting for the HTTP response
      if (existingBytes > 0 && _totalBytes != null && _totalBytes! > 0) {
        _progress = existingBytes / _totalBytes!;
      } else {
        _progress = 0.0;
      }
      notifyListeners();

      // ── Build request headers ─────────────────────────────────────────────
      final headers = <String, dynamic>{
        'User-Agent':
            'Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Mobile Safari/537.36',
      };
      if (existingBytes > 0) {
        headers['Range'] = 'bytes=$existingBytes-';
        debugPrint('DownloadService: Resuming from byte $existingBytes');
      } else {
        debugPrint('DownloadService: Starting fresh download');
      }

      // ── Stream the response ───────────────────────────────────────────────
      final response = await _dio.get<ResponseBody>(
        url,
        options: Options(
          responseType: ResponseType.stream,
          headers: headers,
          followRedirects: true,
          sendTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(minutes: 60),
          // Accept 2xx and 416 (range not satisfiable — file already complete)
          validateStatus: (s) => s != null && (s < 300 || s == 416),
        ),
        cancelToken: _cancelToken,
      );

      // 416 means the server says "you already have the whole file"
      if (response.statusCode == 416) {
        debugPrint('DownloadService: 416 — file already fully downloaded');
        if (await tempFile.exists()) {
          final existingFinal = File(savePath);
          if (await existingFinal.exists()) await existingFinal.delete();
          await tempFile.rename(savePath);
        }
        _status = DownloadStatus.completed;
        _progress = 1.0;
        _hasPartial = false;
        notifyListeners();
        return;
      }

      // ── Determine total file size from response headers ───────────────────
      final isResuming = response.statusCode == 206 && existingBytes > 0;
      final resumeBytes = isResuming ? existingBytes : 0;

      final contentRangeHeader = response.headers.value('content-range');
      final contentLengthHeader = response.headers.value('content-length');
      final contentLengthValue =
          int.tryParse(contentLengthHeader ?? '');

      int? totalBytes;
      if (contentRangeHeader != null) {
        // Content-Range: bytes 12345678-349999999/350000000
        final match = RegExp(r'/(\d+)$').firstMatch(contentRangeHeader);
        if (match != null) totalBytes = int.tryParse(match.group(1)!);
      } else if (contentLengthValue != null) {
        totalBytes = resumeBytes + contentLengthValue;
      }

      if (totalBytes != null) _totalBytes = totalBytes;

      _status = DownloadStatus.downloading;
      _progress = (totalBytes != null && totalBytes > 0)
          ? resumeBytes / totalBytes
          : 0.0;
      notifyListeners();

      debugPrint(
          'DownloadService: isResuming=$isResuming  resumeBytes=$resumeBytes  '
          'totalBytes=$totalBytes  statusCode=${response.statusCode}');

      // ── Write stream to file ──────────────────────────────────────────────
      final raf = await tempFile.open(
          mode: isResuming ? FileMode.append : FileMode.write);
      var receivedBytes = resumeBytes;

      try {
        await for (final Uint8List chunk in response.data!.stream) {
          if (_cancelToken?.isCancelled ?? false) break;
          await raf.writeFrom(chunk);
          receivedBytes += chunk.length;
          if (totalBytes != null && totalBytes > 0) {
            _progress = receivedBytes / totalBytes;
            notifyListeners();
          }
        }
      } finally {
        await raf.close();
      }

      // ── Handle cancel mid-stream ──────────────────────────────────────────
      if (_cancelToken?.isCancelled ?? false) {
        // Keep .tmp so the next call resumes from here
        _hasPartial = true;
        _status = DownloadStatus.notStarted;
        notifyListeners();
        return;
      }

      // ── Rename temp → final ───────────────────────────────────────────────
      final existingFinal = File(savePath);
      if (await existingFinal.exists()) await existingFinal.delete();
      await tempFile.rename(savePath);

      _status = DownloadStatus.completed;
      _progress = 1.0;
      _hasPartial = false;
      notifyListeners();

      debugPrint('DownloadService: ✅ Download complete → $savePath');
    } catch (e) {
      if (e is DioException && CancelToken.isCancel(e)) {
        // Cancelled — keep .tmp intact for resume
        _hasPartial = true;
        _status = DownloadStatus.notStarted;
        notifyListeners();
        return;
      }
      _status = DownloadStatus.error;
      _errorMessage =
          e is DioException ? (e.message ?? e.toString()) : e.toString();
      debugPrint('DownloadService: ❌ Error: $_errorMessage');
      notifyListeners();
    }
  }

  /// Cancel the in-progress download.  The partial `.tmp` file is preserved
  /// so the next [downloadModel] call resumes automatically.
  void cancelDownload() {
    _cancelToken?.cancel();
    // Status + _hasPartial updated in the catch/cancel branch of downloadModel()
  }

  /// Delete the partial `.tmp` file and reset state (for a forced fresh start).
  Future<void> clearPartialDownload(String fileName) async {
    try {
      final dir = await getApplicationSupportDirectory()
          .timeout(const Duration(seconds: 2));
      final tmpFile = File('${p.join(dir.path, fileName)}.tmp');
      if (await tmpFile.exists()) await tmpFile.delete();
    } catch (_) {}
    _hasPartial = false;
    _totalBytes = null;
    _progress = 0.0;
    if (_status == DownloadStatus.error || _status == DownloadStatus.notStarted) {
      notifyListeners();
    }
  }
}
