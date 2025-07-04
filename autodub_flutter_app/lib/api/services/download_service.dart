import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'api_service.dart';

class DownloadProgress {
  final double progress;
  final String message;
  final int downloaded;
  final int total;

  DownloadProgress({
    required this.progress,
    required this.message,
    required this.downloaded,
    required this.total,
  });

  String get downloadedMB => (downloaded / (1024 * 1024)).toStringAsFixed(1);
  String get totalMB => (total / (1024 * 1024)).toStringAsFixed(1);
  String get progressPercentage => '${(progress * 100).toStringAsFixed(1)}%';
}

class DownloadResult {
  final bool success;
  final String? filePath;
  final String? error;

  DownloadResult.success(this.filePath) : success = true, error = null;
  DownloadResult.error(this.error) : success = false, filePath = null;
}

class DownloadService {
  static String getVideoPreviewUrl(String filename) {
    return ApiService.getStreamUrl(filename);
  }

  static String getDownloadUrl(String filename) {
    return ApiService.getDownloadUrl(filename);
  }

  static Future<DownloadResult> downloadFile({
    required String filename,
    Function(DownloadProgress)? onProgress,
  }) async {
    try {
      // Request storage permission
      if (Platform.isAndroid) {
        final status = await Permission.storage.request();
        if (!status.isGranted) {
          return DownloadResult.error('Storage permission denied');
        }
      }

      // Get download directory
      Directory? downloadDir;
      if (Platform.isAndroid) {
        downloadDir = Directory('/storage/emulated/0/Download');
        if (!downloadDir.existsSync()) {
          downloadDir = await getExternalStorageDirectory();
        }
      } else {
        downloadDir = await getApplicationDocumentsDirectory();
      }

      if (downloadDir == null) {
        return DownloadResult.error('Could not access download directory');
      }

      final filePath = '${downloadDir.path}/$filename';
      final file = File(filePath);

      // Start download
      final url = getDownloadUrl(filename);
      final client = HttpClient();
      
      try {
        final request = await client.getUrl(Uri.parse(url));
        final response = await request.close();

        if (response.statusCode != 200) {
          return DownloadResult.error('Download failed: HTTP ${response.statusCode}');
        }

        final total = response.contentLength;
        int downloaded = 0;

        final sink = file.openWrite();

        await for (final chunk in response) {
          sink.add(chunk);
          downloaded += chunk.length;
          
          final progress = total > 0 ? downloaded / total : 0.0;
          onProgress?.call(DownloadProgress(
            progress: progress,
            message: 'Downloading... ${(progress * 100).toStringAsFixed(1)}%',
            downloaded: downloaded,
            total: total,
          ));
        }

        await sink.close();
        client.close();

        return DownloadResult.success(filePath);
      } catch (e) {
        client.close();
        throw e;
      }
    } catch (e) {
      debugPrint('Download error: $e');
      return DownloadResult.error('Download failed: $e');
    }
  }

  static Future<bool> checkFileExists(String filename) async {
    try {
      final url = getDownloadUrl(filename);
      final client = HttpClient();
      
      try {
        final request = await client.headUrl(Uri.parse(url));
        final response = await request.close();
        client.close();
        
        return response.statusCode == 200;
      } catch (e) {
        client.close();
        return false;
      }
    } catch (e) {
      return false;
    }
  }
}
