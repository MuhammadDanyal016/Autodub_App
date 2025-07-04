import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';

class FileUtils {
  // Supported video formats
  static const List<String> supportedVideoExtensions = [
    'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', '3gp', 'webm', 'm4v'
  ];
  
  // Maximum file size in bytes (500MB)
  static const int maxFileSizeBytes = 500 * 1024 * 1024;
  
  /// Check if file has a supported video extension
  static bool isValidVideoFile(String fileName) {
    final extension = fileName.toLowerCase().split('.').last;
    return supportedVideoExtensions.contains(extension);
  }
  
  /// Get file size in MB
  static Future<double> getFileSizeInMB(String filePath) async {
    try {
      final file = File(filePath);
      final bytes = await file.length();
      return bytes / (1024 * 1024);
    } catch (e) {
      debugPrint('Error getting file size: $e');
      return 0.0;
    }
  }
  
  /// Check if file size is within limits
  static Future<bool> isFileSizeValid(String filePath) async {
    try {
      final file = File(filePath);
      final bytes = await file.length();
      return bytes <= maxFileSizeBytes;
    } catch (e) {
      debugPrint('Error checking file size: $e');
      return false;
    }
  }
  
  /// Validate file completely
  static Future<FileValidationResult> validateVideoFile(String filePath) async {
    try {
      final file = File(filePath);
      
      // Check if file exists
      if (!await file.exists()) {
        return FileValidationResult(
          isValid: false,
          error: 'File does not exist or cannot be accessed.',
        );
      }
      
      // Check file extension
      final fileName = filePath.split('/').last;
      if (!isValidVideoFile(fileName)) {
        return FileValidationResult(
          isValid: false,
          error: 'Unsupported file format. Supported formats: ${supportedVideoExtensions.join(', ').toUpperCase()}',
        );
      }
      
      // Check file size
      final bytes = await file.length();
      final sizeMB = bytes / (1024 * 1024);
      
      if (bytes > maxFileSizeBytes) {
        return FileValidationResult(
          isValid: false,
          error: 'File too large (${sizeMB.toStringAsFixed(1)}MB). Maximum size is ${(maxFileSizeBytes / (1024 * 1024)).toStringAsFixed(0)}MB.',
        );
      }
      
      // Check if file is readable
      try {
        await file.openRead().take(1024).toList();
      } catch (e) {
        return FileValidationResult(
          isValid: false,
          error: 'File cannot be read. It may be corrupted or in use by another application.',
        );
      }
      
      return FileValidationResult(
        isValid: true,
        fileSizeMB: sizeMB,
        fileName: fileName,
      );
      
    } catch (e) {
      return FileValidationResult(
        isValid: false,
        error: 'Error validating file: $e',
      );
    }
  }
  
  /// Request necessary permissions for file access
  static Future<bool> requestFilePermissions() async {
    try {
      if (Platform.isAndroid) {
        // For Android 13+ (API 33+), request granular media permissions
        Map<Permission, PermissionStatus> permissions = {};
        
        // Always request these basic permissions
        permissions[Permission.videos] = await Permission.videos.request();
        permissions[Permission.audio] = await Permission.audio.request();
        
        // For older Android versions, also request storage permission
        permissions[Permission.storage] = await Permission.storage.request();
        
        // Check if we got at least one of the required permissions
        bool hasVideoPermission = permissions[Permission.videos]?.isGranted ?? false;
        bool hasStoragePermission = permissions[Permission.storage]?.isGranted ?? false;
        
        return hasVideoPermission || hasStoragePermission;
      }
      
      return true; // For iOS and other platforms
    } catch (e) {
      debugPrint('Error requesting permissions: $e');
      
      // Fallback: try requesting storage permission only
      try {
        final storageStatus = await Permission.storage.request();
        return storageStatus.isGranted;
      } catch (fallbackError) {
        debugPrint('Fallback permission request failed: $fallbackError');
        return false;
      }
    }
  }

  /// Check if we have necessary file permissions
  static Future<bool> hasFilePermissions() async {
    try {
      if (Platform.isAndroid) {
        // Check multiple permission types
        bool hasVideos = await Permission.videos.isGranted;
        bool hasStorage = await Permission.storage.isGranted;
        bool hasManageStorage = await Permission.manageExternalStorage.isGranted;
        
        return hasVideos || hasStorage || hasManageStorage;
      }
      
      return true; // For iOS and other platforms
    } catch (e) {
      debugPrint('Error checking permissions: $e');
      return false;
    }
  }
  
  /// Get Android SDK version
  static Future<int> _getAndroidVersion() async {
    try {
      // This is a simplified version - in a real app you might want to use
      // device_info_plus package for more accurate version detection
      return 33; // Assume modern Android for now
    } catch (e) {
      return 30; // Default to Android 11
    }
  }
  
  /// Format file size for display
  static String formatFileSize(double sizeInMB) {
    if (sizeInMB < 1) {
      return '${(sizeInMB * 1024).toStringAsFixed(0)} KB';
    } else if (sizeInMB < 1024) {
      return '${sizeInMB.toStringAsFixed(1)} MB';
    } else {
      return '${(sizeInMB / 1024).toStringAsFixed(1)} GB';
    }
  }
}

class FileValidationResult {
  final bool isValid;
  final String? error;
  final double? fileSizeMB;
  final String? fileName;
  
  FileValidationResult({
    required this.isValid,
    this.error,
    this.fileSizeMB,
    this.fileName,
  });
}
