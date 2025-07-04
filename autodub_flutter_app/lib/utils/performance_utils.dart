import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';

class PerformanceUtils {
  static bool _isPerformanceLoggingEnabled = kDebugMode;
  static final Map<String, DateTime> _operationStartTimes = {};
  
  /// Enable or disable performance logging
  static void setPerformanceLogging(bool enabled) {
    _isPerformanceLoggingEnabled = enabled;
  }
  
  /// Start timing an operation
  static void startOperation(String operationName) {
    if (!_isPerformanceLoggingEnabled) return;
    _operationStartTimes[operationName] = DateTime.now();
    debugPrint('🚀 Started: $operationName');
  }
  
  /// End timing an operation and log the duration
  static void endOperation(String operationName) {
    if (!_isPerformanceLoggingEnabled) return;
    
    final startTime = _operationStartTimes[operationName];
    if (startTime != null) {
      final duration = DateTime.now().difference(startTime);
      debugPrint('✅ Completed: $operationName in ${duration.inMilliseconds}ms');
      _operationStartTimes.remove(operationName);
      
      // Warn about slow operations
      if (duration.inMilliseconds > 1000) {
        debugPrint('⚠️ Slow operation detected: $operationName took ${duration.inMilliseconds}ms');
      }
    }
  }
  
  /// Check if the main thread is busy
  static bool isMainThreadBusy() {
    return SchedulerBinding.instance.schedulerPhase != SchedulerPhase.idle;
  }
  
  /// Defer execution until the next frame to avoid blocking UI
  static Future<void> deferToNextFrame() async {
    await SchedulerBinding.instance.endOfFrame;
  }
  
  /// Execute a heavy operation in chunks to avoid blocking UI
  static Future<void> executeInChunks<T>(
    List<T> items,
    Future<void> Function(T item) operation, {
    int chunkSize = 10,
    Duration delay = const Duration(milliseconds: 1),
  }) async {
    for (int i = 0; i < items.length; i += chunkSize) {
      final chunk = items.skip(i).take(chunkSize);
      
      for (final item in chunk) {
        await operation(item);
      }
      
      // Give the UI thread a chance to update
      if (i + chunkSize < items.length) {
        await Future.delayed(delay);
      }
    }
  }
  
  /// Log memory usage (debug only)
  static void logMemoryUsage(String context) {
    if (!_isPerformanceLoggingEnabled) return;
    
    // This is a simplified memory check - in production you might want more detailed metrics
    debugPrint('📊 Memory check at $context');
  }
}
