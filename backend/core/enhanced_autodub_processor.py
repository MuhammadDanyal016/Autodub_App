"""
ENHANCED AutoDub Processor v4.0 - High-level processing interface with comprehensive improvements
"""

import logging
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional, List
import json
import uuid
import shutil

from core.enhanced_autodub_pipeline import EnhancedAutoDubPipeline
from utils.gpu_utils import gpu_manager
from config import Config

logger = logging.getLogger(__name__)

class EnhancedAutoDubProcessor:
    """ENHANCED AutoDub Processor v4.0 with comprehensive improvements"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.pipeline = None
        self.processing_history = []
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Initialize the ENHANCED processor with comprehensive setup"""
        try:
            logger.info("🚀 Initializing ENHANCED AutoDub Processor v4.0...")
            logger.info("🔧 Features enabled:")
            logger.info("   ✅ ML-based gender detection (audio + video)")
            logger.info("   ✅ Pretrained emotion recognition")
            logger.info("   ✅ Enhanced speaker diarization with embeddings")
            logger.info("   ✅ Unique voice allocation per speaker")
            logger.info("   ✅ Background music separation")
            logger.info("   ✅ Advanced text cleaning and TTS")
            logger.info("   ✅ Comprehensive error handling")
            
            # Initialize the enhanced pipeline with config
            self.pipeline = EnhancedAutoDubPipeline(self.config)
            
            logger.info("✅ ENHANCED AutoDub Processor initialized successfully!")
            
        except Exception as e:
            logger.error(f"ENHANCED processor initialization failed: {e}")
            raise Exception(f"Failed to initialize ENHANCED processor: {str(e)}")
    
    def process_video(self, video_path: str, target_language: str, 
                     source_language: Optional[str] = None, 
                     output_dir: Optional[str] = None) -> Dict:
        """
        Process video with ENHANCED features and comprehensive validation
        """
        processing_start = time.time()
        
        try:
            logger.info("🎬 Starting ENHANCED video processing...")
            logger.info(f"📁 Input: {video_path}")
            logger.info(f"🌍 Target: {target_language}")
            logger.info(f"🌍 Source: {source_language or 'auto-detect'}")
            logger.info(f"📁 Output: {output_dir or 'default'}")
            
            # Validate inputs
            validation_result = self._validate_inputs(video_path, target_language, source_language)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'validation_failed': True
                }
            
            # Get video info for processing optimization
            video_info = validation_result['video_info']
            logger.info(f"📊 Video info: {video_info['duration']:.1f}s, "
                       f"{video_info['width']}x{video_info['height']}, "
                       f"{video_info['fps']:.1f}fps")
            
            # Setup output directory if specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Using custom output directory: {output_path}")
            
            # Process with enhanced pipeline
            result = self.pipeline.process_video_enhanced(
                video_path, target_language, source_language
            )
            
            if result['success']:
                # Organize output files
                organized_result = self._organize_output(result, video_path, output_dir)
                
                # Add processing history
                self.processing_history.append({
                    'processing_id': result['processing_id'],
                    'timestamp': time.time(),
                    'video_path': video_path,
                    'target_language': target_language,
                    'source_language': source_language,
                    'output_dir': output_dir,
                    'success': True,
                    'processing_time': result['total_processing_time'],
                    'speakers_found': result['speakers_found'],
                    'segments_processed': result['segments_processed']
                })
                
                logger.info("🎉 ENHANCED video processing completed successfully!")
                return organized_result
            else:
                logger.error(f"ENHANCED processing failed: {result.get('error')}")
                return result
                
        except Exception as e:
            processing_time = time.time() - processing_start
            logger.error(f"ENHANCED video processing failed after {processing_time:.2f}s: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'stage': 'processor_level_error'
            }
    
    def _validate_inputs(self, video_path: str, target_language: str, 
                        source_language: Optional[str]) -> Dict:
        """Enhanced input validation with comprehensive checks"""
        try:
            logger.info("🔍 Validating inputs...")
            
            # Check video file
            video_path = Path(video_path)
            if not video_path.exists():
                return {'valid': False, 'error': f"Video file not found: {video_path}"}
            
            if video_path.stat().st_size == 0:
                return {'valid': False, 'error': "Video file is empty"}
            
            # Check file format
            if not video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                return {'valid': False, 'error': f"Unsupported video format: {video_path.suffix}"}
            
            # Check languages
            if target_language not in Config.SUPPORTED_LANGUAGES:
                return {
                    'valid': False, 
                    'error': f"Unsupported target language: {target_language}"
                }
            
            if source_language and source_language not in Config.SUPPORTED_LANGUAGES:
                return {
                    'valid': False, 
                    'error': f"Unsupported source language: {source_language}"
                }
            
            # Get video information
            video_info = self._get_video_info(str(video_path))
            
            # Check video duration
            if video_info['duration'] > Config.VIDEO_MAX_DURATION:
                return {
                    'valid': False, 
                    'error': f"Video too long: {video_info['duration']:.1f}s (max: {Config.VIDEO_MAX_DURATION}s)"
                }
            
            if video_info['duration'] < 1.0:
                return {'valid': False, 'error': "Video too short (minimum 1 second)"}
            
            logger.info("✅ Input validation passed")
            return {
                'valid': True,
                'video_info': video_info
            }
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return {'valid': False, 'error': f"Validation error: {str(e)}"}
    
    def _get_video_info(self, video_path: str) -> Dict:
        """Get comprehensive video information"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'aspect_ratio': width / height if height > 0 else 1.0
            }
            
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
            return {
                'duration': 0,
                'fps': 25,
                'frame_count': 0,
                'width': 1920,
                'height': 1080,
                'resolution': "1920x1080",
                'aspect_ratio': 16/9
            }
    
    def _organize_output(self, result: Dict, original_video_path: str, output_dir: Optional[str] = None) -> Dict:
        """Organize output files with enhanced structure"""
        try:
            logger.info("📁 Organizing output files...")
            
            # Determine output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = Config.OUTPUT_DIR
            
            base_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create organized output structure
            processing_id = result['processing_id']
            timestamp = int(time.time())
            
            # Generate output filenames
            video_name = Path(original_video_path).stem
            output_video_name = f"{video_name}_dubbed_{result['target_language']}_{timestamp}.mp4"
            script_name = f"{video_name}_script_{result['target_language']}_{timestamp}.json"
            analysis_name = f"{video_name}_analysis_{result['target_language']}_{timestamp}.json"
            
            # Move output video to organized location
            if result.get('output_video'):
                organized_video_path = base_output_dir / output_video_name
                shutil.move(result['output_video'], organized_video_path)
                result['output_video'] = str(organized_video_path)
        
            # Create comprehensive script file
            script_data = {
                'processing_info': {
                    'processing_id': processing_id,
                    'timestamp': timestamp,
                    'original_video': str(original_video_path),
                    'target_language': result['target_language'],
                    'detected_language': result.get('detected_language'),
                    'processing_time': result['total_processing_time'],
                    'speakers_found': result['speakers_found'],
                    'segments_processed': result['segments_processed']
                },
                'segments': result.get('segments', []),
                'speaker_analysis': result.get('speaker_analysis', {}),
                'voice_assignments': result.get('voice_assignments', {}),
                'processing_stages': result.get('processing_stages', {}),
                'quality_metrics': result.get('analysis_data', {}).get('quality_metrics', {})
            }
            
            script_path = base_output_dir / script_name
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(script_data, f, indent=2, ensure_ascii=False)
        
            # Create analysis report
            analysis_data = {
                'processing_summary': {
                    'success_rate': result.get('success_rate', 0),
                    'total_speakers': result['speakers_found'],
                    'total_segments': result['segments_processed'],
                    'processing_time': result['total_processing_time'],
                    'stages_completed': len(result.get('processing_stages', {}))
                },
                'speaker_analysis': result.get('speaker_analysis', {}),
                'voice_assignments': result.get('voice_assignments', {}),
                'translation_stats': result.get('translation_stats', {}),
                'memory_usage': result.get('analysis_data', {}).get('memory_usage', []),
                'quality_metrics': result.get('analysis_data', {}).get('quality_metrics', {})
            }
            
            analysis_path = base_output_dir / analysis_name
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
            # Update result with organized paths
            result.update({
                'output_video': str(organized_video_path) if result.get('output_video') else None,
                'script_file': str(script_path),
                'analysis_file': str(analysis_path),
                'output_directory': str(base_output_dir),
                'output_files': {
                    'video': output_video_name,
                    'script': script_name,
                    'analysis': analysis_name
                }
            })
            
            logger.info("✅ Output files organized successfully")
            logger.info(f"📹 Video: {output_video_name}")
            logger.info(f"📄 Script: {script_name}")
            logger.info(f"📊 Analysis: {analysis_name}")
            logger.info(f"📁 Directory: {base_output_dir}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Output organization failed: {e}")
            return result
    
    def get_processing_status(self) -> Dict:
        """Get comprehensive processing status"""
        try:
            memory_info = gpu_manager.monitor_memory()
            
            return {
                'processor_ready': self.pipeline is not None,
                'system_info': memory_info,
                'processing_history_count': len(self.processing_history),
                'last_processing': self.processing_history[-1] if self.processing_history else None,
                'supported_languages': list(Config.SUPPORTED_LANGUAGES.keys()),
                'max_video_duration': Config.VIDEO_MAX_DURATION,
                'features': {
                    'ml_gender_detection': True,
                    'emotion_recognition': True,
                    'enhanced_diarization': True,
                    'unique_voice_allocation': True,
                    'background_music_separation': True,
                    'advanced_text_cleaning': True
                }
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                'processor_ready': False,
                'error': str(e)
            }
    
    def cleanup(self):
        """Cleanup processor resources"""
        try:
            logger.info("🧹 Cleaning up ENHANCED AutoDub Processor...")
            
            if self.pipeline:
                self.pipeline.cleanup()
                self.pipeline = None
            
            # Clear processing history
            self.processing_history.clear()
            
            # Clear GPU memory
            gpu_manager.clear_gpu_cache()
            
            logger.info("✅ ENHANCED AutoDub Processor cleanup completed")
            
        except Exception as e:
            logger.warning(f"ENHANCED AutoDub Processor cleanup warning: {e}")
