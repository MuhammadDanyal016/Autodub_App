import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'api_service.dart';

class ProcessingService {
  static Future<ProcessingStatus?> getProcessingStatus(String processId) async {
    try {
      final response = await ApiService.getProcessingStatus(processId);
      if (response.success && response.data != null) {
        return response.data;
      } else {
        debugPrint('Failed to get processing status: ${response.error}');
        return null;
      }
    } catch (e) {
      debugPrint('Processing status error: $e');
      return null;
    }
  }

  static Future<ProcessFiles?> getProcessingFiles(String processId) async {
    try {
      final response = await ApiService.getProcessFiles(processId);
      if (response.success && response.data != null) {
        return response.data;
      } else {
        debugPrint('Failed to get processing files: ${response.error}');
        return null;
      }
    } catch (e) {
      debugPrint('Processing files error: $e');
      return null;
    }
  }

  static Stream<ProcessingStatus> watchProcessingStatus(String processId) async* {
    while (true) {
      final status = await getProcessingStatus(processId);
      if (status != null) {
        yield status;
        
        // Stop watching if completed or failed
        if (status.isCompleted || status.isFailed) {
          break;
        }
      }
      
      // Wait before next check
      await Future.delayed(Duration(seconds: 3));
    }
  }

  static String formatProcessingStage(String stage) {
    switch (stage.toLowerCase()) {
      case 'audio_processing_v2':
        return 'Processing Audio';
      case 'speaker_diarization_clean':
        return 'Identifying Speakers';
      case 'analysis_v2_clean':
        return 'Analyzing Speech';
      case 'speech_recognition_clean':
        return 'Transcribing Audio';
      case 'translation':
        return 'Translating Text';
      case 'tts_v2_with_silence':
        return 'Generating Speech';
      case 'lip_sync_with_background':
        return 'Syncing Video';
      default:
        return stage.replaceAll('_', ' ').split(' ').map((word) => 
            word.isNotEmpty ? word[0].toUpperCase() + word.substring(1) : word
        ).join(' ');
    }
  }

  static String getStatusDisplayText(ProcessingStatus status) {
    if (status.isCompleted) {
      return 'Processing completed successfully!';
    } else if (status.isFailed) {
      return 'Processing failed: ${status.error ?? 'Unknown error'}';
    } else if (status.isProcessing) {
      return status.message.isNotEmpty ? status.message : 'Processing video...';
    } else if (status.isQueued) {
      return 'Waiting in queue...';
    } else {
      return 'Unknown status';
    }
  }

  static Color getStatusColor(ProcessingStatus status) {
    if (status.isCompleted) {
      return const Color(0xFF05CEA8); // Green
    } else if (status.isFailed) {
      return const Color(0xFFE74C3C); // Red
    } else if (status.isProcessing) {
      return const Color(0xFF3498DB); // Blue
    } else if (status.isQueued) {
      return const Color(0xFFF39C12); // Orange
    } else {
      return const Color(0xFF95A5A6); // Gray
    }
  }
}
