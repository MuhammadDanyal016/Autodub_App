import 'package:flutter/material.dart';
import 'dart:async';
import 'package:percent_indicator/percent_indicator.dart';
import 'package:video_player/video_player.dart';
import 'package:chewie/chewie.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:autodub_flutter_app/api/services/api_service.dart';
import 'package:autodub_flutter_app/api/services/processing_service.dart';
import 'package:autodub_flutter_app/api/services/download_service.dart';

class DownloadPage extends StatefulWidget {
  final String processId;

  const DownloadPage({super.key, required this.processId});

  @override
  _DownloadPageState createState() => _DownloadPageState();
}

class _DownloadPageState extends State<DownloadPage>
    with SingleTickerProviderStateMixin {
  bool isDownloading = false;
  DownloadProgress? downloadProgress;
  ProcessingStatus? processingStatus;
  ProcessFiles? processFiles;
  late AnimationController _controller;
  late Animation<double> _animation;
  VideoPlayerController? _videoPlayerController;
  ChewieController? _chewieController;
  bool _videoError = false;
  String _errorMessage = '';
  StreamSubscription<ProcessingStatus>? _statusSubscription;
  Timer? _statusTimer;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: true);
    _animation = CurvedAnimation(parent: _controller, curve: Curves.easeInOut);
    _startStatusMonitoring();
  }

  @override
  void dispose() {
    _controller.dispose();
    _videoPlayerController?.dispose();
    _chewieController?.dispose();
    _statusSubscription?.cancel();
    _statusTimer?.cancel();
    super.dispose();
  }

  void _startStatusMonitoring() {
    _statusTimer = Timer.periodic(Duration(seconds: 3), (timer) async {
      await _checkProcessingStatus();
    });
    
    // Initial status check
    _checkProcessingStatus();
  }

  Future<void> _checkProcessingStatus() async {
    try {
      final status = await ProcessingService.getProcessingStatus(widget.processId);
      if (status != null && mounted) {
        setState(() {
          processingStatus = status;
        });

        // If processing is complete, get files and initialize video
        if (status.isCompleted) {
          _statusTimer?.cancel();
          await _loadProcessFiles();
          await _initializeVideoPlayer();
        } else if (status.isFailed) {
          _statusTimer?.cancel();
          setState(() {
            _errorMessage = status.error ?? 'Processing failed';
            _videoError = true;
          });
        }
      }
    } catch (e) {
      debugPrint('Status check error: $e');
      if (mounted) {
        setState(() {
          _errorMessage = 'Failed to check processing status: $e';
          _videoError = true;
        });
      }
    }
  }

  Future<void> _loadProcessFiles() async {
    try {
      final files = await ProcessingService.getProcessingFiles(widget.processId);
      if (files != null && mounted) {
        setState(() {
          processFiles = files;
        });
      }
    } catch (e) {
      debugPrint('Load files error: $e');
    }
  }

  Future<void> _initializeVideoPlayer() async {
    if (processFiles == null || processFiles!.files.isEmpty) return;

    try {
      final outputFile = processFiles!.files.firstWhere(
        (file) => file.type == 'output_video',
        orElse: () => processFiles!.files.first,
      );

      final videoUrl = DownloadService.getVideoPreviewUrl(outputFile.filename);
      
      _videoPlayerController = VideoPlayerController.network(videoUrl);
      await _videoPlayerController!.initialize();
      
      _chewieController = ChewieController(
        videoPlayerController: _videoPlayerController!,
        autoPlay: false,
        looping: false,
        aspectRatio: _videoPlayerController!.value.aspectRatio,
        placeholder: Center(
          child: SpinKitWave(color: Color(0xFF05CEA8), size: 50.0)
        ),
      );
      
      if (mounted) {
        setState(() {
          _videoError = false;
        });
      }
    } catch (e) {
      debugPrint('Error initializing video player: $e');
      if (mounted) {
        setState(() {
          _videoError = true;
          _errorMessage = 'Error loading video preview: $e';
        });
      }
    }
  }

  Future<void> _downloadFile() async {
    if (processFiles == null || processFiles!.files.isEmpty) {
      _showErrorDialog('No files available for download');
      return;
    }

    final outputFile = processFiles!.files.firstWhere(
      (file) => file.type == 'output_video',
      orElse: () => processFiles!.files.first,
    );

    setState(() => isDownloading = true);

    try {
      final result = await DownloadService.downloadFile(
        filename: outputFile.filename,
        onProgress: (progress) {
          if (mounted) {
            setState(() {
              downloadProgress = progress;
            });
          }
        },
      );

      if (mounted) {
        setState(() => isDownloading = false);
        
        if (result.success) {
          _showCompletionDialog(
            'Download Complete',
            'Video saved successfully!\nLocation: ${result.filePath}',
          );
        } else {
          _showErrorDialog('Download failed: ${result.error}');
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() => isDownloading = false);
        _showErrorDialog('Download error: $e');
      }
    }
  }

  void _cancelDownload() {
    setState(() {
      isDownloading = false;
      downloadProgress = null;
    });
    _showInfoSnackBar('Download cancelled');
  }

  void _showErrorDialog(String message) {
    if (!mounted) return;
    
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Text('Error',
            style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold)),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text('OK',
                style: TextStyle(
                    color: Color(0xFF05CEA8), fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  void _showCompletionDialog(String title, String message) {
    if (!mounted) return;
    
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Text(title,
            style: TextStyle(
                color: Color(0xFF05CEA8), fontWeight: FontWeight.bold)),
        content: Text(message, textAlign: TextAlign.center),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text('OK',
                style: TextStyle(
                    color: Color(0xFF05CEA8), fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  void _showInfoSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.blue,
        duration: Duration(seconds: 2),
      ),
    );
  }

  Widget _buildProcessingStatus() {
    if (processingStatus == null) {
      return Card(
        elevation: 8,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        color: Color(0xFF2C3A36),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            children: [
              SpinKitWave(color: Color(0xFF05CEA8), size: 50.0),
              SizedBox(height: 16),
              Text(
                'Loading processing status...',
                style: TextStyle(color: Colors.white70),
              ),
            ],
          ),
        ),
      );
    }

    if (processingStatus!.isCompleted) {
      return Card(
        elevation: 8,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        color: Color(0xFF2C3A36),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            children: [
              AnimatedBuilder(
                animation: _animation,
                builder: (context, child) {
                  return Transform.scale(
                    scale: 1.0 + (_animation.value * 0.1),
                    child: Icon(
                      Icons.check_circle,
                      size: 80,
                      color: Color(0xFF05CEA8),
                    ),
                  );
                },
              ),
              SizedBox(height: 24),
              Text(
                'Your video is ready!',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              SizedBox(height: 16),
              Text(
                'Processing completed successfully. You can now preview and download your dubbed video.',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.white70,
                ),
                textAlign: TextAlign.center,
              ),
              if (processingStatus!.elapsedTime != null)
                Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(
                    'Processing time: ${processingStatus!.elapsedTime!.toStringAsFixed(1)}s',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.white60,
                    ),
                  ),
                ),
            ],
          ),
        ),
      );
    }

    if (processingStatus!.isFailed) {
      return Card(
        elevation: 8,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        color: Color(0xFF2C3A36),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            children: [
              Icon(
                Icons.error,
                size: 80,
                color: Colors.red,
              ),
              SizedBox(height: 24),
              Text(
                'Processing Failed',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.red,
                ),
              ),
              SizedBox(height: 16),
              Text(
                processingStatus!.error ?? 'Unknown error occurred',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.white70,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
    }

    // Processing in progress
    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      color: Color(0xFF2C3A36),
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            SpinKitWave(color: Color(0xFF05CEA8), size: 50.0),
            SizedBox(height: 24),
            Text(
              'Processing Video...',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            SizedBox(height: 16),
            LinearProgressIndicator(
              value: processingStatus!.progress / 100,
              backgroundColor: Colors.white.withOpacity(0.2),
              valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF05CEA8)),
            ),
            SizedBox(height: 8),
            Text(
              '${processingStatus!.progress.toStringAsFixed(1)}% - ${processingStatus!.message}',
              style: TextStyle(color: Colors.white70, fontSize: 14),
              textAlign: TextAlign.center,
            ),
            if (processingStatus!.currentStage != null)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Text(
                  'Current: ${ProcessingService.formatProcessingStage(processingStatus!.currentStage!)}',
                  style: TextStyle(color: Colors.white60, fontSize: 12),
                ),
              ),
            if (processingStatus!.estimatedRemaining != null)
              Padding(
                padding: const EdgeInsets.only(top: 4.0),
                child: Text(
                  'Est. remaining: ${processingStatus!.estimatedRemaining!.toStringAsFixed(0)}s',
                  style: TextStyle(color: Colors.white60, fontSize: 12),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildVideoPreview() {
    if (processingStatus == null || !processingStatus!.isCompleted) {
      return SizedBox.shrink();
    }

    if (_chewieController != null && !_videoError) {
      return Card(
        elevation: 8,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        color: Color(0xFF2C3A36),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: AspectRatio(
            aspectRatio: 16 / 9,
            child: Chewie(controller: _chewieController!),
          ),
        ),
      );
    } else if (_videoError) {
      return Card(
        elevation: 8,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        color: Color(0xFF2C3A36),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              Icon(Icons.video_library, size: 60, color: Colors.white54),
              SizedBox(height: 16),
              Text(
                'Video preview not available',
                style: TextStyle(color: Colors.white70, fontSize: 16),
                textAlign: TextAlign.center,
              ),
              if (_errorMessage.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(
                    _errorMessage,
                    style: TextStyle(color: Colors.red, fontSize: 12),
                    textAlign: TextAlign.center,
                  ),
                ),
            ],
          ),
        ),
      );
    } else {
      return Card(
        elevation: 8,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        color: Color(0xFF2C3A36),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: SpinKitWave(color: Color(0xFF05CEA8), size: 50.0),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Processing & Download',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Theme.of(context).primaryColor,
        elevation: 0,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Theme.of(context).primaryColor,
              Theme.of(context).primaryColor.withOpacity(0.8),
              Color(0xFF293431),
            ],
            stops: [0.0, 0.3, 1.0],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Processing Status
                  _buildProcessingStatus(),
                  
                  SizedBox(height: 24),
                  
                  // Video Preview
                  _buildVideoPreview(),
                  
                  SizedBox(height: 24),
                  
                  // Download Button
                  if (processingStatus?.isCompleted ?? false)
                    ElevatedButton(
                      onPressed: isDownloading ? null : _downloadFile,
                      style: ElevatedButton.styleFrom(
                        padding: EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(30),
                        ),
                        backgroundColor: Color(0xFF05CEA8),
                        disabledBackgroundColor:
                            Color(0xFF05CEA8).withOpacity(0.5),
                      ),
                      child: isDownloading
                          ? SizedBox(
                              width: 24,
                              height: 24,
                              child: CircularProgressIndicator(
                                color: Colors.white,
                                strokeWidth: 2,
                              ),
                            )
                          : Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.download, color: Colors.white),
                                SizedBox(width: 8),
                                Text(
                                  'Download Video',
                                  style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white),
                                ),
                              ],
                            ),
                    ),
                  
                  SizedBox(height: 16),
                  
                  // Download Progress
                  if (isDownloading && downloadProgress != null)
                    Column(
                      children: [
                        LinearPercentIndicator(
                          width: MediaQuery.of(context).size.width - 48,
                          lineHeight: 8.0,
                          percent: downloadProgress!.progress,
                          backgroundColor: Colors.white.withOpacity(0.2),
                          progressColor: Color(0xFF05CEA8),
                          linearStrokeCap: LinearStrokeCap.roundAll,
                          padding: EdgeInsets.zero,
                        ),
                        SizedBox(height: 8),
                        Text(
                          downloadProgress!.message,
                          style: TextStyle(color: Colors.white70, fontSize: 14),
                        ),
                        if (downloadProgress!.total > 0)
                          Text(
                            '${downloadProgress!.downloadedMB} / ${downloadProgress!.totalMB}',
                            style: TextStyle(color: Colors.white60, fontSize: 12),
                          ),
                        SizedBox(height: 8),
                        ElevatedButton(
                          onPressed: _cancelDownload,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.red,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                          child: Text('Cancel Download',
                              style: TextStyle(color: Colors.white)),
                        ),
                      ],
                    ),
                  
                  // File Info
                  if (processFiles != null && processFiles!.files.isNotEmpty)
                    Card(
                      elevation: 4,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
                      color: Colors.white.withOpacity(0.1),
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'File Information',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                            ),
                            SizedBox(height: 8),
                            ...processFiles!.files.map((file) => Padding(
                              padding: const EdgeInsets.symmetric(vertical: 4.0),
                              child: Row(
                                children: [
                                  Icon(Icons.video_file, color: Color(0xFF05CEA8), size: 16),
                                  SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      file.filename,
                                      style: TextStyle(color: Colors.white70, fontSize: 14),
                                    ),
                                  ),
                                  Text(
                                    '${file.sizeMb.toStringAsFixed(1)}MB',
                                    style: TextStyle(color: Colors.white60, fontSize: 12),
                                  ),
                                ],
                              ),
                            )).toList(),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
