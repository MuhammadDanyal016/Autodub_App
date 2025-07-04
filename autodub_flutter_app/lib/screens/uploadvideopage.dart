import 'dart:async';
import 'dart:io';
import 'package:autodub_flutter_app/screens/download.dart';
import 'package:autodub_flutter_app/api/services/api_service.dart';
import 'package:autodub_flutter_app/api/services/upload_service.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:page_transition/page_transition.dart';
import 'package:permission_handler/permission_handler.dart';

class UploadVideoPage extends StatefulWidget {
  const UploadVideoPage({super.key});

  @override
  _UploadVideoPageState createState() => _UploadVideoPageState();
}

class _UploadVideoPageState extends State<UploadVideoPage> {
  String? filePath;
  bool isUploading = false;
  UploadProgress? currentProgress;
  Timer? _timeoutTimer;
  bool _connectionTested = false;
  bool _connectionOk = false;
  
  // Language selections with API codes - HARDCODED (no configuration screen)
  String selectedSourceLanguage = 'en';
  String selectedTargetLanguage = 'ur';

  @override
  void initState() {
    super.initState();
    _testConnection();
  }

  @override
  void dispose() {
    _timeoutTimer?.cancel();
    super.dispose();
  }

  Future<void> _testConnection() async {
    try {
      final isConnected = await ApiService.testConnection();
      setState(() {
        _connectionTested = true;
        _connectionOk = isConnected;
      });
      
      if (!isConnected) {
        _showConnectionDialog();
      }
    } catch (e) {
      setState(() {
        _connectionTested = true;
        _connectionOk = false;
      });
      _showConnectionDialog();
    }
  }

  void _showConnectionDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        title: Text('Connection Error', style: TextStyle(color: Colors.red)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('Cannot connect to the AutoDub server.'),
            SizedBox(height: 16),
            Text('Please make sure:', style: TextStyle(fontWeight: FontWeight.bold)),
            SizedBox(height: 8),
            Text('• Your Cloudflare tunnel is running'),
            Text('• The tunnel URL is correctly configured'),
            Text('• Your internet connection is stable'),
            Text('• The backend server is running on Colab'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(ctx).pop();
              _testConnection();
            },
            child: Text('Retry'),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text('Continue Anyway'),
          ),
        ],
      ),
    );
  }

  Future<bool> _requestPermissions() async {
    try {
      if (Platform.isAndroid) {
        final status = await Permission.storage.request();
        if (status.isGranted) return true;
        
        final videoStatus = await Permission.videos.request();
        return videoStatus.isGranted;
      }
      return true;
    } catch (e) {
      debugPrint('Permission request error: $e');
      return true;
    }
  }

  Future<void> _pickFile() async {
    try {
      final hasPermissions = await _requestPermissions();
      if (!hasPermissions) {
        _showErrorDialog('Storage permissions are required to select files.');
        return;
      }

      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ApiService.supportedVideoFormats,
        allowMultiple: false,
      );
      
      if (result != null && result.files.single.path != null) {
        final selectedPath = result.files.single.path!;
        
        // Validate file using upload service
        final validationError = await UploadService.validateFile(selectedPath);
        if (validationError != null) {
          _showErrorDialog(validationError);
          return;
        }
        
        final fileSizeInMB = await UploadService.getFileSizeMB(selectedPath);
        
        setState(() {
          filePath = selectedPath;
        });
        
        _showSuccessSnackBar('File selected: ${fileSizeInMB.toStringAsFixed(1)}MB');
      }
    } catch (e) {
      debugPrint('File picker error: $e');
      _showErrorDialog('Error selecting file: $e');
    }
  }

  Future<void> _uploadAndProcess() async {
    if (filePath == null) {
      _showErrorDialog('No file selected to upload.');
      return;
    }

    // Validate language selection
    if (!ApiService.supportedLanguages.containsKey(selectedSourceLanguage) || 
        !ApiService.supportedLanguages.containsKey(selectedTargetLanguage)) {
      _showErrorDialog('Invalid language selection. Please select valid languages.');
      return;
    }

    if (selectedSourceLanguage == selectedTargetLanguage) {
      _showErrorDialog('Source and target languages cannot be the same.');
      return;
    }

    setState(() {
      isUploading = true;
      currentProgress = null;
    });

    // Set a timeout timer
    _timeoutTimer = Timer(Duration(minutes: 20), () {
      if (isUploading) {
        setState(() {
          isUploading = false;
          currentProgress = UploadProgress(
            stage: UploadStage.error,
            progress: 0.0,
            message: 'Upload timed out. Please try again.',
          );
        });
      }
    });

    try {
      debugPrint('=== Starting Upload and Processing ===');
      debugPrint('API Base URL: ${ApiService.baseUrl}');
      debugPrint('Source Language Code: $selectedSourceLanguage (${ApiService.supportedLanguages[selectedSourceLanguage]})');
      debugPrint('Target Language Code: $selectedTargetLanguage (${ApiService.supportedLanguages[selectedTargetLanguage]})');
      
      final result = await UploadService.uploadVideoWithProgress(
        filePath: filePath!,
        targetLanguage: selectedTargetLanguage,
        sourceLanguage: selectedSourceLanguage,
        onProgress: (progress) {
          if (mounted) {
            setState(() {
              currentProgress = progress;
            });
          }
        },
      );

      _timeoutTimer?.cancel();

      if (result.success && result.data != null) {
        _showSuccessSnackBar('Upload completed successfully!');
        
        await Future.delayed(Duration(milliseconds: 1000));
        
        if (mounted) {
          // Start processing the uploaded video
          try {
            setState(() {
              currentProgress = UploadProgress(
                stage: UploadStage.processing,
                progress: 0.5,
                message: 'Starting video processing...',
              );
            });
            
            // Call the process endpoint with the temp_path
            final processResponse = await ApiService.processVideo(
              tempPath: result.data!.tempPath,
              targetLanguage: selectedTargetLanguage,
              sourceLanguage: selectedSourceLanguage,
            );
            
            if (processResponse.success && processResponse.data != null) {
              setState(() {
                currentProgress = UploadProgress(
                  stage: UploadStage.complete,
                  progress: 1.0,
                  message: 'Processing started successfully!',
                );
              });
              
              // Navigate to download page with the process ID
              Navigator.push(
                context,
                PageTransition(
                  type: PageTransitionType.rightToLeft,
                  child: DownloadPage(
                    processId: processResponse.data!.processId,
                  ),
                ),
              );
            } else {
              throw Exception(processResponse.error ?? 'Failed to start processing');
            }
          } catch (e) {
            setState(() {
              isUploading = false;
              currentProgress = UploadProgress(
                stage: UploadStage.error,
                progress: 0.0,
                message: 'Processing failed: $e',
              );
            });
            _showErrorDialog('Processing failed: $e');
          }
        }
      } else {
        setState(() {
          isUploading = false;
        });
        _showErrorDialog('Upload failed: ${result.error}');
      }
    } catch (e) {
      _timeoutTimer?.cancel();
      
      if (mounted) {
        setState(() {
          isUploading = false;
          currentProgress = UploadProgress(
            stage: UploadStage.error,
            progress: 0.0,
            message: 'Upload failed: $e',
          );
        });
        _showErrorDialog('Upload failed: $e');
      }
    }
  }

  void _cancelUpload() {
    _timeoutTimer?.cancel();
    setState(() {
      isUploading = false;
      currentProgress = null;
    });
    _showInfoSnackBar('Upload cancelled');
  }

  void _showErrorDialog(String message) {
    if (!mounted) return;
    
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Error', style: TextStyle(color: Color(0xFF05CEA8))),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showSuccessSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Color(0xFF05CEA8),
        duration: Duration(seconds: 2),
      ),
    );
  }

  void _showInfoSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Color(0xFF05CEA8),
        duration: Duration(seconds: 2),
      ),
    );
  }

  Color _getProgressColor() {
    if (currentProgress == null) return Color(0xFF05CEA8);
    
    switch (currentProgress!.stage) {
      case UploadStage.preparing:
        return Color(0xFF05CEA8);
      case UploadStage.uploading:
        return Color(0xFF05CEA8);
      case UploadStage.processing:
        return Color(0xFF05CEA8);
      case UploadStage.complete:
        return Color(0xFF05CEA8);
      case UploadStage.error:
        return Color(0xFF05CEA8);
    }
  }

  String _getStageText() {
    if (currentProgress == null) return 'Preparing...';
    
    switch (currentProgress!.stage) {
      case UploadStage.preparing:
        return 'Preparing Upload';
      case UploadStage.uploading:
        return 'Uploading File';
      case UploadStage.processing:
        return 'Processing';
      case UploadStage.complete:
        return 'Complete';
      case UploadStage.error:
        return 'Error';
    }
  }

  Widget _buildLanguageDropdown({
    required String value,
    required void Function(String?) onChanged,
    required String labelText,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white30),
      ),
      child: DropdownButtonFormField<String>(
        value: value,
        items: ApiService.supportedLanguages.entries
            .map((entry) => DropdownMenuItem(
                  value: entry.key,
                  child: Text(
                    entry.value,
                    style: TextStyle(color: Colors.white),
                  ),
                ))
            .toList(),
        onChanged: onChanged,
        decoration: InputDecoration(
          labelText: labelText,
          labelStyle: TextStyle(color: Colors.white70),
          border: InputBorder.none,
          contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        ),
        dropdownColor: Color(0xFF293431),
        style: TextStyle(color: Colors.white),
        icon: Icon(Icons.arrow_drop_down, color: Colors.white70),
      ),
    );
  }

  Widget _buildConnectionStatus() {
    if (!_connectionTested) {
      return Container(
        padding: EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Color(0xFF05CEA8).withOpacity(0.2),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          children: [
            SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF05CEA8)),
            ),
            SizedBox(width: 8),
            Text(
              'Testing connection...',
              style: TextStyle(color: Color(0xFF05CEA8), fontSize: 12),
            ),
          ],
        ),
      );
    }

    return Container(
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Color(0xFF05CEA8).withOpacity(0.2),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(
            _connectionOk ? Icons.check_circle : Icons.error,
            color: Color(0xFF05CEA8),
            size: 16,
          ),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              _connectionOk 
                  ? 'Connected to AutoDub server'
                  : 'Cannot connect to server',
              style: TextStyle(
                color: Color(0xFF05CEA8),
                fontSize: 12,
              ),
            ),
          ),
          if (!_connectionOk)
            TextButton(
              onPressed: _testConnection,
              child: Text('Retry', style: TextStyle(color: Color(0xFF05CEA8), fontSize: 12)),
            ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Upload & Process Video'),
        backgroundColor: Theme.of(context).primaryColor,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Theme.of(context).primaryColor, Color(0xFF293431)],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Connection Status
                _buildConnectionStatus(),
                
                SizedBox(height: 20),
                
                // Upload icon
                Center(
                  child: Container(
                    height: 100,
                    width: 100,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Color(0xFF05CEA8),
                    ),
                    child: Icon(
                      Icons.cloud_upload, 
                      size: 50, 
                      color: Colors.white
                    ),
                  ),
                ),
                SizedBox(height: 24),
                
                // Language Selection Card
                Card(
                  color: Colors.white.withOpacity(0.1),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Language Settings',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 18,
                          ),
                        ),
                        SizedBox(height: 16),
                        
                        // Source Language
                        _buildLanguageDropdown(
                          value: selectedSourceLanguage,
                          onChanged: (value) {
                            if (!isUploading && value != null) {
                              setState(() {
                                selectedSourceLanguage = value;
                              });
                            }
                          },
                          labelText: 'Source Language (Original)',
                        ),
                        
                        SizedBox(height: 16),
                        
                        // Target Language
                        _buildLanguageDropdown(
                          value: selectedTargetLanguage,
                          onChanged: (value) {
                            if (!isUploading && value != null) {
                              setState(() {
                                selectedTargetLanguage = value;
                              });
                            }
                          },
                          labelText: 'Target Language (Dub to)',
                        ),
                        
                        SizedBox(height: 12),
                        
                        // Language info
                        Container(
                          padding: EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Color(0xFF05CEA8).withOpacity(0.2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(
                            children: [
                              Icon(Icons.info_outline, color: Color(0xFF05CEA8), size: 16),
                              SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  'Supported: ${ApiService.supportedLanguages.values.join(', ')}',
                                  style: TextStyle(color: Colors.white70, fontSize: 12),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                
                SizedBox(height: 20),
                
                // File format info
                Card(
                  color: Colors.white.withOpacity(0.1),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        Text(
                          'Supported Video Formats',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                        SizedBox(height: 8),
                        Text(
                          ApiService.supportedVideoFormats.join(', ').toUpperCase(),
                          style: TextStyle(color: Colors.white70, fontSize: 12),
                          textAlign: TextAlign.center,
                        ),
                        SizedBox(height: 4),
                        Text(
                          'Maximum file size: ${ApiService.maxFileSizeMb}MB',
                          style: TextStyle(color: Colors.white60, fontSize: 11),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),
                
                SizedBox(height: 20),
                
                // Select file button
                ElevatedButton.icon(
                  onPressed: isUploading ? null : _pickFile,
                  icon: Icon(Icons.attach_file),
                  label: Text('Select Video File'),
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(vertical: 16),
                    backgroundColor: Color(0xFF05CEA8),
                  ),
                ),
                
                SizedBox(height: 16),
                
                // Selected file info
                if (filePath != null)
                  Container(
                    padding: EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.check_circle, color: Color(0xFF05CEA8), size: 20),
                        SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Selected: ${filePath!.split('/').last}',
                            style: TextStyle(color: Colors.white, fontSize: 14),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                  ),
                
                SizedBox(height: 24),
                
                // Upload and process button
                ElevatedButton.icon(
                  onPressed: (filePath != null && !isUploading && _connectionOk) ? _uploadAndProcess : null,
                  icon: Icon(isUploading ? Icons.hourglass_empty : Icons.play_arrow),
                  label: Text(
                    isUploading ? 'Processing...' : 'Upload & Start Dubbing',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(vertical: 16),
                    backgroundColor: Color(0xFF05CEA8),
                  ),
                ),
                
                SizedBox(height: 24),
                
                // Upload progress
                if (isUploading || currentProgress != null)
                  Card(
                    color: Colors.white.withOpacity(0.1),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        children: [
                          // Stage indicator
                          Row(
                            children: [
                              Container(
                                width: 12,
                                height: 12,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: _getProgressColor(),
                                ),
                              ),
                              SizedBox(width: 8),
                              Text(
                                _getStageText(),
                                style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                          
                          SizedBox(height: 12),
                          
                          // Progress bar
                          LinearProgressIndicator(
                            value: currentProgress?.progress,
                            backgroundColor: Colors.white.withOpacity(0.2),
                            valueColor: AlwaysStoppedAnimation<Color>(_getProgressColor()),
                          ),
                          
                          SizedBox(height: 8),
                          
                          // Progress message
                          if (currentProgress != null)
                            Text(
                              currentProgress!.message,
                              style: TextStyle(color: Colors.white70, fontSize: 12),
                              textAlign: TextAlign.center,
                            ),
                          
                          // Progress percentage
                          if (currentProgress != null)
                            Padding(
                              padding: const EdgeInsets.only(top: 4.0),
                              child: Text(
                                currentProgress!.progressPercentage,
                                style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          
                          SizedBox(height: 16),
                          
                          // Cancel button
                          if (isUploading)
                            ElevatedButton(
                              onPressed: _cancelUpload,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Color(0xFF05CEA8),
                              ),
                              child: Text('Cancel'),
                            ),
                        ],
                      ),
                    ),
                  ),
                
                SizedBox(height: 20),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
