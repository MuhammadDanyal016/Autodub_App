import 'dart:io';
import 'package:flutter/foundation.dart';
import 'api_service.dart';

enum UploadStage {
  preparing,
  uploading,
  processing,
  complete,
  error,
}

class UploadProgress {
  final UploadStage stage;
  final double progress;
  final String message;
  final int? uploadedBytes;
  final int? totalBytes;

  UploadProgress({
    required this.stage,
    required this.progress,
    required this.message,
    this.uploadedBytes,
    this.totalBytes,
  });

  String get progressPercentage => '${(progress * 100).toStringAsFixed(1)}%';
  
  String get uploadedMB => uploadedBytes != null 
      ? (uploadedBytes! / (1024 * 1024)).toStringAsFixed(1) 
      : '0.0';
      
  String get totalMB => totalBytes != null 
      ? (totalBytes! / (1024 * 1024)).toStringAsFixed(1) 
      : '0.0';
}

class UploadService {
  static const List<String> supportedExtensions = [
    'mp4', 'avi', 'mov', 'mkv', 'wmv', '3gp', 'webm', 'm4v'
  ];

  static Future<String?> validateFile(String filePath) async {
    try {
      final file = File(filePath);
      
      if (!await file.exists()) {
        return 'File does not exist';
      }
      
      final extension = filePath.split('.').last.toLowerCase();
      if (!supportedExtensions.contains(extension)) {
        return 'Unsupported file format. Supported: ${supportedExtensions.join(', ')}';
      }
      
      final fileSizeBytes = await file.length();
      final fileSizeMB = fileSizeBytes / (1024 * 1024);
      
      if (fileSizeMB > 500) {
        return 'File too large. Maximum size is 500MB';
      }
      
      return null; // No error
    } catch (e) {
      return 'Error validating file: $e';
    }
  }

  static Future<double> getFileSizeMB(String filePath) async {
    try {
      final file = File(filePath);
      final bytes = await file.length();
      return bytes / (1024 * 1024);
    } catch (e) {
      return 0.0;
    }
  }

  static Future<ApiResponse<UploadResponse>> uploadVideoWithProgress({
    required String filePath,
    required String targetLanguage,
    String? sourceLanguage,
    Function(UploadProgress)? onProgress,
  }) async {
    try {
      // Stage 1: Preparing
      onProgress?.call(UploadProgress(
        stage: UploadStage.preparing,
        progress: 0.0,
        message: 'Preparing upload...',
      ));

      // Validate file
      final validationError = await validateFile(filePath);
      if (validationError != null) {
        return ApiResponse.error(validationError);
      }

      // Stage 2: Uploading
      onProgress?.call(UploadProgress(
        stage: UploadStage.uploading,
        progress: 0.1,
        message: 'Starting upload...',
      ));

      final result = await ApiService.uploadVideo(
        filePath: filePath,
        targetLanguage: targetLanguage,
        sourceLanguage: sourceLanguage,
        onProgress: (progress) {
          onProgress?.call(UploadProgress(
            stage: UploadStage.uploading,
            progress: 0.1 + (progress * 0.8), // 10% to 90%
            message: 'Uploading... ${(progress * 100).toStringAsFixed(1)}%',
          ));
        },
      );

      if (result.success) {
        onProgress?.call(UploadProgress(
          stage: UploadStage.complete,
          progress: 1.0,
          message: 'Upload completed successfully!',
        ));
      }

      return result;
    } catch (e) {
      debugPrint('Upload service error: $e');
      return ApiResponse.error('Upload failed: $e');
    }
  }
}
