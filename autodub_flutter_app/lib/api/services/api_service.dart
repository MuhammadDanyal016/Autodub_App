import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';

import 'package:autodub_flutter_app/config/app_config.dart';

class ApiService {
  // Use configuration from AppConfig
  static String get baseUrl => AppConfig.apiBaseUrl;
  static Duration get _timeout => AppConfig.apiTimeout;
  
  static final http.Client _client = http.Client();

  // Use hardcoded configuration from AppConfig
  static Map<String, String> get supportedLanguages => AppConfig.supportedLanguages;
  static List<String> get supportedVideoFormats => AppConfig.supportedVideoFormats;
  static int get maxFileSizeMb => AppConfig.maxFileSizeMb;

  // Headers for all requests
  static Map<String, String> get _headers => {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'User-Agent': 'AutoDub-Flutter/1.0',
  };

  // Health check endpoint
  static Future<ApiResponse<Map<String, dynamic>>> healthCheck() async {
    try {
      final response = await _client
          .get(
            Uri.parse('$baseUrl/health'),
            headers: _headers,
          )
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return ApiResponse.success(data);
      } else {
        return ApiResponse.error('Health check failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Health check error: $e');
      return ApiResponse.error('Health check failed: $e');
    }
  }

  // Get system information
  static Future<ApiResponse<Map<String, dynamic>>> getSystemInfo() async {
    try {
      final response = await _client
          .get(
            Uri.parse('$baseUrl/'),
            headers: _headers,
          )
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return ApiResponse.success(data);
      } else {
        return ApiResponse.error('Failed to get system info: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('System info error: $e');
      return ApiResponse.error('System info failed: $e');
    }
  }

  // Upload video file
  static Future<ApiResponse<UploadResponse>> uploadVideo({
    required String filePath,
    required String targetLanguage,
    String? sourceLanguage,
    Function(double)? onProgress,
  }) async {
    try {
      final file = File(filePath);
      if (!await file.exists()) {
        return ApiResponse.error('File does not exist: $filePath');
      }

      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/upload'),
      );

      // Add form fields
      request.fields['target_language'] = targetLanguage;
      if (sourceLanguage != null) {
        request.fields['source_language'] = sourceLanguage;
      }

      // Add file
      final multipartFile = await http.MultipartFile.fromPath(
        'file',
        filePath,
        filename: file.path.split('/').last,
      );
      request.files.add(multipartFile);

      // Add headers
      request.headers.addAll({
        'Accept': 'application/json',
        'User-Agent': 'AutoDub-Flutter/1.0',
      });

      debugPrint('Uploading to: $baseUrl/upload');
      debugPrint('Target language: $targetLanguage');
      debugPrint('Source language: $sourceLanguage');
      debugPrint('File: ${file.path.split('/').last}');

      // Send request
      final streamedResponse = await _client.send(request).timeout(_timeout);
      final response = await http.Response.fromStream(streamedResponse);

      debugPrint('Upload response status: ${response.statusCode}');
      debugPrint('Upload response body: ${response.body}');

      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return ApiResponse.success(UploadResponse.fromJson(data));
      } else {
        try {
          final errorData = json.decode(response.body) as Map<String, dynamic>;
          return ApiResponse.error(errorData['detail']?.toString() ?? 'Upload failed');
        } catch (e) {
          return ApiResponse.error('Upload failed with status ${response.statusCode}');
        }
      }
    } catch (e) {
      debugPrint('Upload error: $e');
      return ApiResponse.error('Upload failed: $e');
    }
  }

  // Start video processing
  static Future<ApiResponse<ProcessResponse>> processVideo({
    required String tempPath,
    required String targetLanguage,
    String? sourceLanguage,
  }) async {
    try {
      final body = {
        'temp_path': tempPath,
        'target_language': targetLanguage,
        if (sourceLanguage != null) 'source_language': sourceLanguage,
      };

      debugPrint('Processing request to: $baseUrl/process');
      debugPrint('Processing body: $body');

      final response = await _client
          .post(
            Uri.parse('$baseUrl/process'),
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: body.entries.map((e) => '${e.key}=${Uri.encodeComponent(e.value)}').join('&'),
          )
          .timeout(_timeout);

      debugPrint('Process response status: ${response.statusCode}');
      debugPrint('Process response body: ${response.body}');

      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return ApiResponse.success(ProcessResponse.fromJson(data));
      } else {
        try {
          final errorData = json.decode(response.body) as Map<String, dynamic>;
          return ApiResponse.error(errorData['detail']?.toString() ?? 'Processing failed');
        } catch (e) {
          return ApiResponse.error('Processing failed with status ${response.statusCode}');
        }
      }
    } catch (e) {
      debugPrint('Process error: $e');
      return ApiResponse.error('Processing failed: $e');
    }
  }

  // Get processing status
  static Future<ApiResponse<ProcessingStatus>> getProcessingStatus(String processId) async {
    try {
      final response = await _client
          .get(
            Uri.parse('$baseUrl/status/$processId'),
            headers: _headers,
          )
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return ApiResponse.success(ProcessingStatus.fromJson(data));
      } else if (response.statusCode == 404) {
        return ApiResponse.error('Process not found');
      } else {
        return ApiResponse.error('Failed to get status: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Status error: $e');
      return ApiResponse.error('Status check failed: $e');
    }
  }

  // Get process files
  static Future<ApiResponse<ProcessFiles>> getProcessFiles(String processId) async {
    try {
      final response = await _client
          .get(
            Uri.parse('$baseUrl/files/$processId'),
            headers: _headers,
          )
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return ApiResponse.success(ProcessFiles.fromJson(data));
      } else {
        return ApiResponse.error('Failed to get files: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Files error: $e');
      return ApiResponse.error('Files check failed: $e');
    }
  }

  // Download file URL
  static String getDownloadUrl(String filename) {
    return '$baseUrl/download/$filename';
  }

  // Stream file URL
  static String getStreamUrl(String filename) {
    return '$baseUrl/stream/$filename';
  }

  // Direct download URL (fallback)
  static String getDirectDownloadUrl(String filename) {
    return '$baseUrl/download-direct/$filename';
  }

  // Cleanup process
  static Future<ApiResponse<Map<String, dynamic>>> cleanupProcess(String processId) async {
    try {
      final response = await _client
          .delete(
            Uri.parse('$baseUrl/cleanup/$processId'),
            headers: _headers,
          )
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return ApiResponse.success(data);
      } else {
        return ApiResponse.error('Cleanup failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Cleanup error: $e');
      return ApiResponse.error('Cleanup failed: $e');
    }
  }

  // Test connection with detailed logging
  static Future<bool> testConnection() async {
    try {
      debugPrint('Testing connection to: $baseUrl');
      
      final response = await _client
          .get(
            Uri.parse('$baseUrl/health'),
            headers: _headers,
          )
          .timeout(Duration(seconds: 10));

      debugPrint('Connection test - Status: ${response.statusCode}');
      debugPrint('Connection test - Body: ${response.body}');
      
      return response.statusCode == 200;
    } catch (e) {
      debugPrint('Connection test failed: $e');
      return false;
    }
  }

  // Dispose client
  static void dispose() {
    _client.close();
  }
}

// Generic API Response wrapper
class ApiResponse<T> {
  final bool success;
  final T? data;
  final String? error;

  ApiResponse.success(this.data) : success = true, error = null;
  ApiResponse.error(this.error) : success = false, data = null;
}

// Data Models
class UploadResponse {
  final String uploadId;
  final String filename;
  final String tempPath;
  final String targetLanguage;
  final String? sourceLanguage;
  final double fileSizeMb;
  final String message;

  UploadResponse({
    required this.uploadId,
    required this.filename,
    required this.tempPath,
    required this.targetLanguage,
    this.sourceLanguage,
    required this.fileSizeMb,
    required this.message,
  });

  factory UploadResponse.fromJson(Map<String, dynamic> json) {
    return UploadResponse(
      uploadId: json['upload_id'] ?? '',
      filename: json['filename'] ?? '',
      tempPath: json['temp_path'] ?? '',
      targetLanguage: json['target_language'] ?? '',
      sourceLanguage: json['source_language'],
      fileSizeMb: (json['file_size_mb'] ?? 0).toDouble(),
      message: json['message'] ?? '',
    );
  }
}

class ProcessResponse {
  final String processId;
  final String status;
  final String message;
  final String progressUrl;
  final String? estimatedTime;
  final bool usingSharedProcessor;
  final bool modelsPersistent;

  ProcessResponse({
    required this.processId,
    required this.status,
    required this.message,
    required this.progressUrl,
    this.estimatedTime,
    required this.usingSharedProcessor,
    required this.modelsPersistent,
  });

  factory ProcessResponse.fromJson(Map<String, dynamic> json) {
    return ProcessResponse(
      processId: json['process_id'] ?? '',
      status: json['status'] ?? '',
      message: json['message'] ?? '',
      progressUrl: json['progress_url'] ?? '',
      estimatedTime: json['estimated_time'],
      usingSharedProcessor: json['using_shared_processor'] ?? false,
      modelsPersistent: json['models_persistent'] ?? false,
    );
  }
}

class ProcessingStatus {
  final String status;
  final double progress;
  final String message;
  final double startTime;
  final String targetLanguage;
  final String? sourceLanguage;
  final String inputFile;
  final bool usingSharedProcessor;
  final bool modelsPersistent;
  final Map<String, PipelineStage> pipelineStages;
  final double? elapsedTime;
  final double? estimatedRemaining;
  final double? endTime;
  final Map<String, dynamic>? result;
  final String? outputVideo;
  final double? outputSizeMb;
  final String? error;
  final String? currentStage;

  ProcessingStatus({
    required this.status,
    required this.progress,
    required this.message,
    required this.startTime,
    required this.targetLanguage,
    this.sourceLanguage,
    required this.inputFile,
    required this.usingSharedProcessor,
    required this.modelsPersistent,
    required this.pipelineStages,
    this.elapsedTime,
    this.estimatedRemaining,
    this.endTime,
    this.result,
    this.outputVideo,
    this.outputSizeMb,
    this.error,
    this.currentStage,
  });

  factory ProcessingStatus.fromJson(Map<String, dynamic> json) {
    final pipelineStagesJson = json['pipeline_stages'] as Map<String, dynamic>? ?? {};
    final pipelineStages = <String, PipelineStage>{};
    
    pipelineStagesJson.forEach((key, value) {
      if (value is Map<String, dynamic>) {
        pipelineStages[key] = PipelineStage.fromJson(value);
      }
    });

    return ProcessingStatus(
      status: json['status'] ?? '',
      progress: (json['progress'] ?? 0).toDouble(),
      message: json['message'] ?? '',
      startTime: (json['start_time'] ?? 0).toDouble(),
      targetLanguage: json['target_language'] ?? '',
      sourceLanguage: json['source_language'],
      inputFile: json['input_file'] ?? '',
      usingSharedProcessor: json['using_shared_processor'] ?? false,
      modelsPersistent: json['models_persistent'] ?? false,
      pipelineStages: pipelineStages,
      elapsedTime: json['elapsed_time']?.toDouble(),
      estimatedRemaining: json['estimated_remaining']?.toDouble(),
      endTime: json['end_time']?.toDouble(),
      result: json['result'],
      outputVideo: json['output_video'],
      outputSizeMb: json['output_size_mb']?.toDouble(),
      error: json['error'],
      currentStage: json['current_stage'],
    );
  }

  bool get isCompleted => status == 'completed';
  bool get isFailed => status == 'failed';
  bool get isProcessing => status == 'processing';
  bool get isQueued => status == 'queued';
}

class PipelineStage {
  final String status;
  final double progress;

  PipelineStage({
    required this.status,
    required this.progress,
  });

  factory PipelineStage.fromJson(Map<String, dynamic> json) {
    return PipelineStage(
      status: json['status'] ?? '',
      progress: (json['progress'] ?? 0).toDouble(),
    );
  }
}

class ProcessFiles {
  final String processId;
  final String status;
  final List<ProcessFile> files;
  final int totalFiles;

  ProcessFiles({
    required this.processId,
    required this.status,
    required this.files,
    required this.totalFiles,
  });

  factory ProcessFiles.fromJson(Map<String, dynamic> json) {
    final filesJson = json['files'] as List<dynamic>? ?? [];
    final files = filesJson
        .map((file) => ProcessFile.fromJson(file as Map<String, dynamic>))
        .toList();

    return ProcessFiles(
      processId: json['process_id'] ?? '',
      status: json['status'] ?? '',
      files: files,
      totalFiles: json['total_files'] ?? 0,
    );
  }
}

class ProcessFile {
  final String type;
  final String filename;
  final String fullPath;
  final double sizeMb;
  final String downloadUrl;
  final String downloadDirectUrl;
  final String streamUrl;

  ProcessFile({
    required this.type,
    required this.filename,
    required this.fullPath,
    required this.sizeMb,
    required this.downloadUrl,
    required this.downloadDirectUrl,
    required this.streamUrl,
  });

  factory ProcessFile.fromJson(Map<String, dynamic> json) {
    return ProcessFile(
      type: json['type'] ?? '',
      filename: json['filename'] ?? '',
      fullPath: json['full_path'] ?? '',
      sizeMb: (json['size_mb'] ?? 0).toDouble(),
      downloadUrl: json['download_url'] ?? '',
      downloadDirectUrl: json['download_direct_url'] ?? '',
      streamUrl: json['stream_url'] ?? '',
    );
  }
}
