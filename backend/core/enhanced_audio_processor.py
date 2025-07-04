"""
ENHANCED Audio Processor Pipeline v2.0 - WITH PERFORMANCE METRICS
Transforms raw, noisy, music-laced audio into clean speech-focused signals
with proper silence preservation for TTS alignment
"""

import os
import time
import numpy as np
import librosa
import soundfile as sf
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import shutil
import warnings
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import json
import pandas as pd
from collections import defaultdict
import psutil
import threading

# Suppress warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class AudioProcessorMetrics:
    """Performance metrics tracker for audio processing pipeline"""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.pipeline_metrics = {
            'total_pipelines': 0,
            'successful_pipelines': 0,
            'failed_pipelines': 0,
            'total_processing_time': 0.0,
            'processing_times': []
        }
        
        self.step_metrics = {
            'audio_extraction': {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'processing_times': [],
                'file_sizes': [],
                'durations': []
            },
            'vocal_separation': {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'processing_times': [],
                'methods_used': defaultdict(int),
                'method_success_rates': defaultdict(list)
            },
            'audio_enhancement': {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'processing_times': [],
                'noise_reduction_effectiveness': [],
                'volume_improvements': []
            },
            'silence_processing': {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'processing_times': [],
                'silence_removed_durations': [],
                'preservation_accuracy': []
            },
            'background_analysis': {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'processing_times': [],
                'segments_detected': [],
                'activity_ratios': []
            }
        }
        
        self.quality_metrics = {
            'input_audio_quality': [],
            'output_audio_quality': [],
            'snr_improvements': [],
            'dynamic_range_improvements': []
        }
        
        self.resource_metrics = {
            'peak_memory_usage': [],
            'average_cpu_usage': [],
            'disk_io_operations': [],
            'temp_files_created': []
        }
        
        self.error_tracking = {
            'extraction_errors': defaultdict(int),
            'separation_errors': defaultdict(int),
            'enhancement_errors': defaultdict(int),
            'general_errors': defaultdict(int)
        }
    
    def start_pipeline_tracking(self, video_path: str) -> Dict:
        """Start tracking a new pipeline execution"""
        return {
            'start_time': time.time(),
            'video_path': video_path,
            'video_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0,
            'initial_memory': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'steps_completed': [],
            'step_times': {}
        }
    
    def record_step_performance(self, step_name: str, success: bool, processing_time: float, 
                              additional_data: Dict = None):
        """Record performance metrics for a processing step"""
        if step_name in self.step_metrics:
            metrics = self.step_metrics[step_name]
            metrics['attempts'] += 1
            metrics['processing_times'].append(processing_time)
            
            if success:
                metrics['successes'] += 1
            else:
                metrics['failures'] += 1
            
            # Store additional step-specific data
            if additional_data:
                for key, value in additional_data.items():
                    if key in metrics and isinstance(metrics[key], list):
                        metrics[key].append(value)
                    elif key == 'method_used' and step_name == 'vocal_separation':
                        metrics['methods_used'][value] += 1
                        metrics['method_success_rates'][value].append(success)
    
    def record_quality_metrics(self, input_quality: float, output_quality: float, 
                             snr_improvement: float = None, dynamic_range_improvement: float = None):
        """Record audio quality metrics"""
        self.quality_metrics['input_audio_quality'].append(input_quality)
        self.quality_metrics['output_audio_quality'].append(output_quality)
        
        if snr_improvement is not None:
            self.quality_metrics['snr_improvements'].append(snr_improvement)
        if dynamic_range_improvement is not None:
            self.quality_metrics['dynamic_range_improvements'].append(dynamic_range_improvement)
    
    def record_resource_usage(self, memory_usage: float, cpu_usage: float, 
                            disk_operations: int = None, temp_files: int = None):
        """Record resource usage metrics"""
        self.resource_metrics['peak_memory_usage'].append(memory_usage)
        self.resource_metrics['average_cpu_usage'].append(cpu_usage)
        
        if disk_operations is not None:
            self.resource_metrics['disk_io_operations'].append(disk_operations)
        if temp_files is not None:
            self.resource_metrics['temp_files_created'].append(temp_files)
    
    def record_error(self, error_type: str, error_message: str):
        """Record error occurrences"""
        if error_type in self.error_tracking:
            self.error_tracking[error_type][error_message] += 1
        else:
            self.error_tracking['general_errors'][f"{error_type}: {error_message}"] += 1
    
    def finish_pipeline_tracking(self, tracking_data: Dict, success: bool, final_memory: float):
        """Finish tracking a pipeline execution"""
        total_time = time.time() - tracking_data['start_time']
        
        self.pipeline_metrics['total_pipelines'] += 1
        self.pipeline_metrics['total_processing_time'] += total_time
        self.pipeline_metrics['processing_times'].append(total_time)
        
        if success:
            self.pipeline_metrics['successful_pipelines'] += 1
        else:
            self.pipeline_metrics['failed_pipelines'] += 1
        
        # Record memory usage
        memory_used = final_memory - tracking_data['initial_memory']
        self.resource_metrics['peak_memory_usage'].append(memory_used)
    
    def calculate_step_metrics(self, step_name: str) -> Dict:
        """Calculate comprehensive metrics for a processing step"""
        if step_name not in self.step_metrics:
            return {}
        
        metrics = self.step_metrics[step_name]
        
        total_attempts = metrics['attempts']
        if total_attempts == 0:
            return {'step_name': step_name, 'no_data': True}
        
        success_rate = (metrics['successes'] / total_attempts * 100)
        failure_rate = (metrics['failures'] / total_attempts * 100)
        
        times = metrics['processing_times']
        avg_time = np.mean(times) if times else 0
        time_std = np.std(times) if times else 0
        min_time = np.min(times) if times else 0
        max_time = np.max(times) if times else 0
        
        result = {
            'step_name': step_name,
            'total_attempts': total_attempts,
            'successful_operations': metrics['successes'],
            'failed_operations': metrics['failures'],
            'success_rate_percent': round(success_rate, 2),
            'failure_rate_percent': round(failure_rate, 2),
            'average_processing_time_ms': round(avg_time * 1000, 2),
            'processing_time_std_ms': round(time_std * 1000, 2),
            'processing_time_range_ms': [round(min_time * 1000, 2), round(max_time * 1000, 2)],
            'throughput_ops_per_second': round(1 / avg_time, 2) if avg_time > 0 else 0
        }
        
        # Add step-specific metrics
        if step_name == 'vocal_separation':
            methods_used = dict(metrics['methods_used'])
            method_success_rates = {}
            for method, successes in metrics['method_success_rates'].items():
                if successes:
                    method_success_rates[method] = round(np.mean(successes) * 100, 2)
            
            result.update({
                'separation_methods_used': methods_used,
                'method_success_rates': method_success_rates,
                'most_successful_method': max(method_success_rates.items(), key=lambda x: x[1])[0] if method_success_rates else None
            })
        
        elif step_name == 'audio_extraction':
            file_sizes = metrics['file_sizes']
            durations = metrics['durations']
            
            if file_sizes:
                result.update({
                    'average_file_size_mb': round(np.mean(file_sizes) / 1024 / 1024, 2),
                    'file_size_range_mb': [round(np.min(file_sizes) / 1024 / 1024, 2), 
                                         round(np.max(file_sizes) / 1024 / 1024, 2)]
                })
            
            if durations:
                result.update({
                    'average_audio_duration_s': round(np.mean(durations), 2),
                    'duration_range_s': [round(np.min(durations), 2), round(np.max(durations), 2)]
                })
        
        elif step_name == 'silence_processing':
            silence_removed = metrics['silence_removed_durations']
            if silence_removed:
                result.update({
                    'average_silence_removed_s': round(np.mean(silence_removed), 2),
                    'silence_removal_efficiency': round(np.mean(silence_removed) / avg_time, 2) if avg_time > 0 else 0
                })
        
        elif step_name == 'background_analysis':
            segments = metrics['segments_detected']
            activity_ratios = metrics['activity_ratios']
            
            if segments:
                result.update({
                    'average_segments_detected': round(np.mean(segments), 1),
                    'segment_detection_range': [int(np.min(segments)), int(np.max(segments))]
                })
            
            if activity_ratios:
                result.update({
                    'average_activity_ratio': round(np.mean(activity_ratios), 3),
                    'activity_ratio_range': [round(np.min(activity_ratios), 3), round(np.max(activity_ratios), 3)]
                })
        
        return result
    
    def calculate_pipeline_metrics(self) -> Dict:
        """Calculate overall pipeline performance metrics"""
        pipeline = self.pipeline_metrics
        
        if pipeline['total_pipelines'] == 0:
            return {'total_pipelines': 0}
        
        success_rate = (pipeline['successful_pipelines'] / pipeline['total_pipelines'] * 100)
        failure_rate = (pipeline['failed_pipelines'] / pipeline['total_pipelines'] * 100)
        
        times = pipeline['processing_times']
        avg_time = np.mean(times) if times else 0
        time_std = np.std(times) if times else 0
        
        return {
            'total_pipelines_processed': pipeline['total_pipelines'],
            'successful_pipelines': pipeline['successful_pipelines'],
            'failed_pipelines': pipeline['failed_pipelines'],
            'pipeline_success_rate_percent': round(success_rate, 2),
            'pipeline_failure_rate_percent': round(failure_rate, 2),
            'average_pipeline_time_s': round(avg_time, 2),
            'pipeline_time_std_s': round(time_std, 2),
            'total_processing_time_hours': round(pipeline['total_processing_time'] / 3600, 2),
            'average_throughput_pipelines_per_hour': round(3600 / avg_time, 2) if avg_time > 0 else 0
        }
    
    def calculate_quality_metrics(self) -> Dict:
        """Calculate audio quality improvement metrics"""
        quality = self.quality_metrics
        
        result = {}
        
        if quality['input_audio_quality'] and quality['output_audio_quality']:
            input_avg = np.mean(quality['input_audio_quality'])
            output_avg = np.mean(quality['output_audio_quality'])
            improvement = output_avg - input_avg
            
            result.update({
                'average_input_quality': round(input_avg, 3),
                'average_output_quality': round(output_avg, 3),
                'quality_improvement': round(improvement, 3),
                'quality_improvement_percent': round((improvement / input_avg * 100) if input_avg > 0 else 0, 2)
            })
        
        if quality['snr_improvements']:
            snr_improvements = quality['snr_improvements']
            result.update({
                'average_snr_improvement_db': round(np.mean(snr_improvements), 2),
                'snr_improvement_range_db': [round(np.min(snr_improvements), 2), round(np.max(snr_improvements), 2)]
            })
        
        if quality['dynamic_range_improvements']:
            dr_improvements = quality['dynamic_range_improvements']
            result.update({
                'average_dynamic_range_improvement_db': round(np.mean(dr_improvements), 2),
                'dynamic_range_improvement_range_db': [round(np.min(dr_improvements), 2), round(np.max(dr_improvements), 2)]
            })
        
        return result
    
    def calculate_resource_metrics(self) -> Dict:
        """Calculate resource usage metrics"""
        resources = self.resource_metrics
        
        result = {}
        
        if resources['peak_memory_usage']:
            memory_usage = resources['peak_memory_usage']
            result.update({
                'average_memory_usage_mb': round(np.mean(memory_usage), 2),
                'peak_memory_usage_mb': round(np.max(memory_usage), 2),
                'memory_usage_std_mb': round(np.std(memory_usage), 2)
            })
        
        if resources['average_cpu_usage']:
            cpu_usage = resources['average_cpu_usage']
            result.update({
                'average_cpu_usage_percent': round(np.mean(cpu_usage), 2),
                'peak_cpu_usage_percent': round(np.max(cpu_usage), 2)
            })
        
        if resources['temp_files_created']:
            temp_files = resources['temp_files_created']
            result.update({
                'average_temp_files_created': round(np.mean(temp_files), 1),
                'total_temp_files_created': sum(temp_files)
            })
        
        return result
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_performance': self.calculate_pipeline_metrics(),
            'step_performance': {},
            'quality_metrics': self.calculate_quality_metrics(),
            'resource_metrics': self.calculate_resource_metrics(),
            'error_summary': self._summarize_errors()
        }
        
        # Calculate metrics for each step
        for step_name in self.step_metrics.keys():
            report['step_performance'][step_name] = self.calculate_step_metrics(step_name)
        
        return report
    
    def _summarize_errors(self) -> Dict:
        """Summarize error occurrences"""
        summary = {}
        total_errors = 0
        
        for error_type, errors in self.error_tracking.items():
            if errors:
                summary[error_type] = dict(errors)
                total_errors += sum(errors.values())
        
        summary['total_errors'] = total_errors
        return summary
    
    def create_performance_table(self) -> pd.DataFrame:
        """Create a performance comparison table for all steps"""
        data = []
        
        for step_name in self.step_metrics.keys():
            metrics = self.calculate_step_metrics(step_name)
            if not metrics.get('no_data', False):
                data.append({
                    'Processing Step': step_name.replace('_', ' ').title(),
                    'Success Rate (%)': metrics.get('success_rate_percent', 0),
                    'Avg Time (ms)': metrics.get('average_processing_time_ms', 0),
                    'Throughput (ops/s)': metrics.get('throughput_ops_per_second', 0),
                    'Total Attempts': metrics.get('total_attempts', 0),
                    'Successful Ops': metrics.get('successful_operations', 0),
                    'Failed Ops': metrics.get('failed_operations', 0)
                })
        
        return pd.DataFrame(data)

class EnhancedAudioProcessorV2:
    """
    Enhanced Audio Processor Pipeline v2.0 - WITH PERFORMANCE METRICS

    Features:
    - Video to audio extraction (16kHz mono)
    - Demucs/Spleeter vocal separation
    - Advanced noise reduction
    - Silence detection and preservation
    - Audio enhancement (filtering, normalization)
    - TTS silence reinsertion for perfect alignment
    - Background music preservation
    - Comprehensive performance tracking
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = self.config.get('sample_rate', 16000)  # Optimal for Whisper/Pyannote
        self.temp_files = []
        
        # Initialize performance metrics tracker
        self.performance_metrics = AudioProcessorMetrics()
        
        # Critical: Store silence information for audio-video alignment
        self.silence_info = {}
        self.background_timestamps = {}
        
        logger.info(f"🚀 Enhanced Audio Processor v2.0 initialized with Performance Metrics")
        logger.info(f"📁 Temp directory: {self.temp_dir}")
        logger.info(f"🎵 Sample rate: {self.sample_rate}Hz")

    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Step 1: Extract audio from video
        Converts input video into raw .wav audio file (16kHz mono)
        """
        step_start_time = time.time()
        try:
            logger.info("🎞️ Step 1: Extracting audio from video...")
            
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".wav")
            
            self.temp_files.append(output_path)
            
            # FFmpeg command optimized for speech processing
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-acodec", "pcm_s16le",
                "-ac", "1",  # Mono channel
                "-ar", str(self.sample_rate),  # 16kHz for optimal model performance
                "-af", "highpass=f=80,lowpass=f=8000",  # Pre-filter for speech range
                "-loglevel", "error",
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                raise Exception("Audio extraction failed - no output generated")
            
            # Validate audio properties
            audio_info = self._validate_audio_properties(output_path)
            
            # Record performance metrics
            processing_time = time.time() - step_start_time
            self.performance_metrics.record_step_performance(
                'audio_extraction', 
                True, 
                processing_time,
                {
                    'file_sizes': audio_info.get('file_size', 0),
                    'durations': audio_info.get('duration', 0)
                }
            )
            
            logger.info(f"✅ Audio extracted: {audio_info['duration']:.2f}s, "
                       f"{audio_info['sample_rate']}Hz, {audio_info['channels']} channel(s)")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            processing_time = time.time() - step_start_time
            error_msg = f"FFmpeg extraction failed: {e.stderr}"
            
            self.performance_metrics.record_step_performance('audio_extraction', False, processing_time)
            self.performance_metrics.record_error('extraction_errors', error_msg)
            
            logger.error(error_msg)
            raise Exception(f"Failed to extract audio: {e}")
        except Exception as e:
            processing_time = time.time() - step_start_time
            error_msg = f"Audio extraction error: {e}"
            
            self.performance_metrics.record_step_performance('audio_extraction', False, processing_time)
            self.performance_metrics.record_error('extraction_errors', error_msg)
            
            logger.error(error_msg)
            raise Exception(f"Audio extraction failed: {str(e)}")

    def separate_vocals_and_background(self, audio_path: str) -> Dict:
        """
        Step 2: Separate vocal and background using Demucs
        Splits audio into vocals.wav (speech-only) and no_vocals.wav (background)
        """
        step_start_time = time.time()
        try:
            logger.info("🎧 Step 2: Separating vocals and background...")
            
            # Try multiple separation methods in order of preference
            methods = [
                ('demucs', self._try_demucs_separation),
                ('spleeter', self._try_spleeter_separation),
                ('librosa_hpss', self._try_librosa_hpss_separation),
                ('spectral', self._try_spectral_separation)
            ]
            
            for method_name, method_func in methods:
                try:
                    result = method_func(audio_path)
                    if result['success']:
                        processing_time = time.time() - step_start_time
                        
                        # Record performance metrics
                        self.performance_metrics.record_step_performance(
                            'vocal_separation', 
                            True, 
                            processing_time,
                            {'method_used': method_name}
                        )
                        
                        logger.info(f"✅ Separation successful using {result['method']}")
                        return result
                except Exception as e:
                    logger.debug(f"Separation method {method_name} failed: {e}")
                    continue
            
            # If all methods fail, return original audio as vocals
            processing_time = time.time() - step_start_time
            self.performance_metrics.record_step_performance(
                'vocal_separation', 
                False, 
                processing_time,
                {'method_used': 'no_separation'}
            )
            
            logger.warning("⚠️ All separation methods failed, using original audio")
            return {
                'success': True,
                'vocals_path': audio_path,
                'background_path': None,
                'method': 'no_separation'
            }
            
        except Exception as e:
            processing_time = time.time() - step_start_time
            error_msg = f"Vocal separation failed: {e}"
            
            self.performance_metrics.record_step_performance('vocal_separation', False, processing_time)
            self.performance_metrics.record_error('separation_errors', error_msg)
            
            logger.error(error_msg)
            return {
                'success': False,
                'error': str(e),
                'vocals_path': audio_path,
                'background_path': None
            }

    def enhance_speech_audio(self, vocals_path: str, audio_id: str = None) -> str:
        """
        Step 3: Apply audio enhancement on vocals.wav
        - Noise reduction
        - Silence trimming with preservation
        - Bandpass filtering
        - Normalization
        """
        step_start_time = time.time()
        try:
            logger.info("🧹 Step 3: Enhancing speech audio...")
            
            # Load vocals audio
            audio, sr = librosa.load(vocals_path, sr=self.sample_rate)
            
            if audio_id is None:
                audio_id = os.path.basename(vocals_path)
            
            logger.info(f"🔊 Processing audio: {len(audio)/sr:.2f}s, ID: {audio_id}")
            
            # Calculate input quality metrics
            input_quality = self._calculate_audio_quality(audio, sr)
            
            # Apply enhancement pipeline
            enhanced_audio = self._apply_enhancement_pipeline(audio, sr, audio_id)
            
            # Calculate output quality metrics
            output_quality = self._calculate_audio_quality(enhanced_audio, sr)
            quality_improvement = output_quality - input_quality
            
            # Save enhanced audio
            output_path = tempfile.mktemp(suffix="_enhanced.wav")
            self.temp_files.append(output_path)
            sf.write(output_path, enhanced_audio, sr)
            
            # Record performance metrics
            processing_time = time.time() - step_start_time
            self.performance_metrics.record_step_performance(
                'audio_enhancement', 
                True, 
                processing_time,
                {
                    'noise_reduction_effectiveness': quality_improvement,
                    'volume_improvements': abs(np.mean(enhanced_audio) - np.mean(audio))
                }
            )
            
            # Record quality metrics
            self.performance_metrics.record_quality_metrics(
                input_quality, output_quality, quality_improvement
            )
            
            logger.info(f"✅ Audio enhanced and saved: {output_path}")
            return output_path
            
        except Exception as e:
            processing_time = time.time() - step_start_time
            error_msg = f"Audio enhancement failed: {e}"
            
            self.performance_metrics.record_step_performance('audio_enhancement', False, processing_time)
            self.performance_metrics.record_error('enhancement_errors', error_msg)
            
            logger.error(error_msg)
            return vocals_path  # Return original if enhancement fails

    def _apply_enhancement_pipeline(self, audio: np.ndarray, sr: int, audio_id: str) -> np.ndarray:
        """Apply comprehensive audio enhancement pipeline"""
        try:
            logger.info("🔧 Applying enhancement pipeline...")
            
            # Step 3a: Advanced noise reduction
            logger.debug("   🔇 Noise reduction...")
            enhanced = self._advanced_noise_reduction(audio, sr)
            
            # Step 3b: Trim silence and store info (CRITICAL for TTS alignment)
            logger.debug("   ✂️ Silence trimming with preservation...")
            enhanced, silence_info = self._trim_silence_with_preservation(enhanced, sr, audio_id)
            
            # Step 3c: Bandpass filtering for speech
            logger.debug("   🎛️ Speech-optimized filtering...")
            enhanced = self._apply_speech_filter(enhanced, sr)
            
            # Step 3d: Volume normalization
            logger.debug("   📊 Volume normalization...")
            enhanced = self._normalize_volume(enhanced)
            
            # Step 3e: Dynamic range compression
            logger.debug("   🎚️ Dynamic range compression...")
            enhanced = self._apply_compression(enhanced)
            
            logger.info("✅ Enhancement pipeline completed")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Enhancement pipeline failed: {e}")
            return audio

    def _trim_silence_with_preservation(self, audio: np.ndarray, sr: int, audio_id: str) -> Tuple[np.ndarray, Dict]:
        """
        CRITICAL: Trim silence while preserving timing information for TTS alignment
        """
        step_start_time = time.time()
        try:
            original_length = len(audio)
            original_duration = original_length / sr
            
            # Use librosa's trim with optimized parameters for speech
            trimmed, index = librosa.effects.trim(
                audio,
                top_db=25,  # More aggressive for speech
                frame_length=2048,
                hop_length=512
            )
            
            # Calculate precise silence information
            start_sample = index[0] if len(index) > 0 else 0
            end_sample = index[1] if len(index) > 1 else original_length
            
            start_silence_samples = start_sample
            end_silence_samples = original_length - end_sample
            start_silence_duration = start_silence_samples / sr
            end_silence_duration = end_silence_samples / sr
            
            # Create comprehensive silence info
            silence_info = {
                'audio_id': audio_id,
                'original_duration': original_duration,
                'trimmed_duration': len(trimmed) / sr,
                'start_silence_duration': start_silence_duration,
                'end_silence_duration': end_silence_duration,
                'start_silence_samples': start_silence_samples,
                'end_silence_samples': end_silence_samples,
                'sample_rate': sr,
                'trim_indices': index.tolist(),
                'silence_removed_total': start_silence_duration + end_silence_duration,
                'speech_start_time': start_silence_duration,
                'speech_end_time': original_duration - end_silence_duration
            }
            
            # Store silence info for TTS alignment
            self.silence_info[audio_id] = silence_info
            
            # Record performance metrics
            processing_time = time.time() - step_start_time
            self.performance_metrics.record_step_performance(
                'silence_processing', 
                True, 
                processing_time,
                {
                    'silence_removed_durations': start_silence_duration + end_silence_duration,
                    'preservation_accuracy': 1.0  # Assume perfect preservation for successful operations
                }
            )
            
            logger.info(f"🔇 Silence preserved for '{audio_id}': "
                       f"start={start_silence_duration:.3f}s, end={end_silence_duration:.3f}s")
            logger.debug(f"   Original: {original_duration:.3f}s -> Trimmed: {len(trimmed)/sr:.3f}s")
            
            return trimmed, silence_info
            
        except Exception as e:
            processing_time = time.time() - step_start_time
            error_msg = f"Silence trimming failed: {e}"
            
            self.performance_metrics.record_step_performance('silence_processing', False, processing_time)
            self.performance_metrics.record_error('enhancement_errors', error_msg)
            
            logger.warning(error_msg)
            # Return original with empty silence info
            silence_info = {
                'audio_id': audio_id,
                'original_duration': len(audio) / sr,
                'trimmed_duration': len(audio) / sr,
                'start_silence_duration': 0.0,
                'end_silence_duration': 0.0,
                'error': str(e)
            }
            return audio, silence_info

    def _calculate_audio_quality(self, audio: np.ndarray, sr: int) -> float:
        """Calculate a simple audio quality metric based on SNR and dynamic range"""
        try:
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio**2))
            
            # Estimate noise floor (bottom 10% of energy)
            frame_length = 2048
            hop_length = 512
            
            # Compute frame-wise RMS
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            frame_rms = np.sqrt(np.mean(frames**2, axis=0))
            
            # Noise floor estimation
            noise_floor = np.percentile(frame_rms, 10)
            
            # Signal-to-noise ratio approximation
            snr = 20 * np.log10(rms / (noise_floor + 1e-10))
            
            # Dynamic range
            dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (noise_floor + 1e-10))
            
            # Combined quality score (normalized)
            quality_score = (snr + dynamic_range) / 100.0
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.debug(f"Quality calculation failed: {e}")
            return 0.5  # Default neutral quality

    def _advanced_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Advanced noise reduction using spectral gating and Wiener filtering"""
        try:
            # Compute STFT
            D = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Estimate noise profile from quiet segments
            noise_profile = self._estimate_noise_profile(magnitude)
            
            # Apply spectral gating
            # Create adaptive threshold based on noise profile
            threshold = noise_profile * 2.0  # 6dB above noise floor
            
            # Soft masking for natural sound
            mask = np.maximum(0.1, np.minimum(1.0, magnitude / threshold))
            
            # Apply Wiener-like filtering
            cleaned_magnitude = magnitude * mask
            
            # Reconstruct audio
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(cleaned_stft, hop_length=512)
            
            return cleaned_audio
            
        except Exception as e:
            logger.debug(f"Advanced noise reduction failed: {e}")
            return audio

    def _estimate_noise_profile(self, magnitude: np.ndarray) -> np.ndarray:
        """Estimate noise profile from quiet segments"""
        try:
            # Find quiet frames (bottom 20% of energy)
            frame_energy = np.mean(magnitude, axis=0)
            quiet_threshold = np.percentile(frame_energy, 20)
            quiet_frames = magnitude[:, frame_energy <= quiet_threshold]
            
            if quiet_frames.shape[1] > 0:
                # Average magnitude of quiet frames
                noise_profile = np.mean(quiet_frames, axis=1, keepdims=True)
            else:
                # Fallback: use minimum values
                noise_profile = np.min(magnitude, axis=1, keepdims=True)
            
            return noise_profile
            
        except Exception as e:
            logger.debug(f"Noise profile estimation failed: {e}")
            return np.ones((magnitude.shape[0], 1)) * 0.01

    def _apply_speech_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply bandpass filter optimized for speech (80Hz - 8kHz)"""
        try:
            nyquist = sr / 2
            
            # High-pass filter: remove low-frequency noise
            low_cutoff = 80 / nyquist
            if low_cutoff < 1.0:
                b, a = butter(4, low_cutoff, btype='high')
                audio = filtfilt(b, a, audio)
            
            # Low-pass filter: remove high-frequency noise
            high_cutoff = min(8000 / nyquist, 0.95)
            if high_cutoff > 0:
                b, a = butter(4, high_cutoff, btype='low')
                audio = filtfilt(b, a, audio)
            
            return audio
            
        except Exception as e:
            logger.debug(f"Speech filtering failed: {e}")
            return audio

    def _normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume with peak and RMS limiting"""
        try:
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 0.15  # Target RMS level
                audio = audio * (target_rms / rms)
            
            # Peak limiting to prevent clipping
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio * (0.95 / peak)
            
            return audio
            
        except Exception as e:
            logger.debug(f"Volume normalization failed: {e}")
            return audio

    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply gentle dynamic range compression"""
        try:
            # Simple soft compression
            threshold = 0.5
            ratio = 4.0
            
            # Find samples above threshold
            above_threshold = np.abs(audio) > threshold
            
            # Apply compression to loud samples
            compressed = audio.copy()
            compressed[above_threshold] = np.sign(audio[above_threshold]) * (
                threshold + (np.abs(audio[above_threshold]) - threshold) / ratio
            )
            
            return compressed
            
        except Exception as e:
            logger.debug(f"Compression failed: {e}")
            return audio

    def reinsert_silence_for_tts(self, tts_audio_path: str, audio_id: str, 
                                output_path: str = None) -> str:
        """
        CRITICAL: Reinsert silence into TTS audio for perfect audio-video alignment
        """
        try:
            logger.info(f"🔇 Reinserting silence for TTS alignment (audio_id: {audio_id})...")
            
            if not os.path.exists(tts_audio_path):
                logger.warning(f"TTS audio file not found: {tts_audio_path}")
                return tts_audio_path
            
            # Get stored silence information
            silence_info = self.silence_info.get(audio_id)
            if not silence_info:
                logger.warning(f"No silence info found for audio_id: {audio_id}")
                logger.debug(f"Available audio_ids: {list(self.silence_info.keys())}")
                return tts_audio_path
            
            # Load TTS audio
            tts_audio, sr = librosa.load(tts_audio_path, sr=self.sample_rate)
            
            # Get silence durations
            start_silence_duration = silence_info.get('start_silence_duration', 0.0)
            end_silence_duration = silence_info.get('end_silence_duration', 0.0)
            
            # Create silence arrays
            start_silence_samples = int(start_silence_duration * sr)
            end_silence_samples = int(end_silence_duration * sr)
            
            start_silence = np.zeros(start_silence_samples, dtype=tts_audio.dtype)
            end_silence = np.zeros(end_silence_samples, dtype=tts_audio.dtype)
            
            # Concatenate: silence + TTS + silence
            aligned_audio = np.concatenate([start_silence, tts_audio, end_silence])
            
            # Generate output path
            if output_path is None:
                base_name = os.path.splitext(tts_audio_path)[0]
                output_path = f"{base_name}_aligned.wav"
            
            # Save aligned audio
            sf.write(output_path, aligned_audio, sr)
            
            logger.info(f"✅ Silence reinserted: {start_silence_duration:.3f}s + "
                       f"{len(tts_audio)/sr:.3f}s + {end_silence_duration:.3f}s = "
                       f"{len(aligned_audio)/sr:.3f}s")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to reinsert silence: {e}")
            return tts_audio_path

    def detect_background_timestamps(self, background_path: str) -> Dict:
        """
        Step 5: Background timestamping using energy-based VAD
        Detect background music/laughter segments for context preservation
        """
        step_start_time = time.time()
        try:
            if not background_path or not os.path.exists(background_path):
                self.performance_metrics.record_step_performance('background_analysis', False, 0.0)
                return {'success': False, 'timestamps': []}
            
            logger.info("🧠 Step 5: Detecting background timestamps...")
            
            # Load background audio
            background, sr = librosa.load(background_path, sr=self.sample_rate)
            
            # Compute energy-based features
            frame_length = 2048
            hop_length = 512
            
            # RMS energy
            rms = librosa.feature.rms(y=background, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=background, sr=sr, hop_length=hop_length)[0]
            
            # Zero crossing rate (texture)
            zcr = librosa.feature.zero_crossing_rate(background, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Detect active segments
            rms_threshold = np.percentile(rms, 30)  # Bottom 30% is silence
            active_frames = rms > rms_threshold
            
            # Convert frame indices to time
            times = librosa.frames_to_time(np.arange(len(active_frames)), sr=sr, hop_length=hop_length)
            
            # Group consecutive active frames into segments
            segments = []
            in_segment = False
            segment_start = 0
            
            for i, (frame_time, active) in enumerate(zip(times, active_frames)):
                if active and not in_segment:
                    segment_start = frame_time
                    in_segment = True
                elif not active and in_segment:
                    segments.append({
                        'start': segment_start,
                        'end': frame_time,
                        'duration': frame_time - segment_start,
                        'type': 'background_active',
                        'avg_energy': np.mean(rms[max(0, i-10):i+1]),
                        'avg_brightness': np.mean(spectral_centroid[max(0, i-10):i+1])
                    })
                    in_segment = False
            
            # Close final segment if needed
            if in_segment:
                segments.append({
                    'start': segment_start,
                    'end': times[-1],
                    'duration': times[-1] - segment_start,
                    'type': 'background_active'
                })
            
            # Store background timestamps
            background_info = {
                'success': True,
                'segments': segments,
                'total_duration': len(background) / sr,
                'active_duration': sum(s['duration'] for s in segments),
                'background_ratio': sum(s['duration'] for s in segments) / (len(background) / sr)
            }
            
            self.background_timestamps[background_path] = background_info
            
            # Record performance metrics
            processing_time = time.time() - step_start_time
            self.performance_metrics.record_step_performance(
                'background_analysis', 
                True, 
                processing_time,
                {
                    'segments_detected': len(segments),
                    'activity_ratios': background_info['background_ratio']
                }
            )
            
            logger.info(f"🎵 Background analysis: {len(segments)} active segments, "
                       f"{background_info['background_ratio']:.1%} activity")
            
            return background_info
            
        except Exception as e:
            processing_time = time.time() - step_start_time
            error_msg = f"Background timestamp detection failed: {e}"
            
            self.performance_metrics.record_step_performance('background_analysis', False, processing_time)
            self.performance_metrics.record_error('enhancement_errors', error_msg)
            
            logger.error(error_msg)
            return {'success': False, 'error': str(e), 'timestamps': []}

    def process_complete_pipeline(self, video_path: str, preserve_background: bool = True) -> Dict:
        """
        Complete audio processing pipeline
        Video → Audio → Separation → Enhancement → Background Analysis
        """
        # Start pipeline tracking
        tracking_data = self.performance_metrics.start_pipeline_tracking(video_path)
        
        try:
            logger.info("🚀 Starting complete audio processing pipeline...")
            logger.info(f"📹 Input: {video_path}")
            logger.info(f"🎼 Background preservation: {preserve_background}")
            
            pipeline_start = time.time()
            audio_id = os.path.basename(video_path)
            
            # Monitor resource usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Step 1: Extract audio from video
            raw_audio_path = self.extract_audio_from_video(video_path)
            tracking_data['steps_completed'].append('audio_extraction')
            
            # Step 2: Separate vocals and background
            separation_result = self.separate_vocals_and_background(raw_audio_path)
            vocals_path = separation_result['vocals_path']
            background_path = separation_result.get('background_path')
            tracking_data['steps_completed'].append('vocal_separation')
            
            # Step 3: Enhance speech audio
            enhanced_vocals_path = self.enhance_speech_audio(vocals_path, audio_id)
            tracking_data['steps_completed'].append('audio_enhancement')
            
            # Step 4: Optional background analysis
            background_info = {}
            if preserve_background and background_path:
                background_info = self.detect_background_timestamps(background_path)
                tracking_data['steps_completed'].append('background_analysis')
            
            # Record final resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()
            
            self.performance_metrics.record_resource_usage(
                final_memory - initial_memory, 
                cpu_percent,
                temp_files=len(self.temp_files)
            )
            
            # Compile results
            pipeline_duration = time.time() - pipeline_start
            
            # Finish pipeline tracking
            self.performance_metrics.finish_pipeline_tracking(tracking_data, True, final_memory)
            
            result = {
                'success': True,
                'pipeline_version': '2.0_with_metrics',
                'processing_time': pipeline_duration,
                'audio_id': audio_id,
                'files': {
                    'raw_audio': raw_audio_path,
                    'enhanced_vocals': enhanced_vocals_path,
                    'background_audio': background_path,
                    'original_video': video_path
                },
                'separation': {
                    'method': separation_result.get('method', 'unknown'),
                    'success': separation_result['success']
                },
                'silence_info': self.silence_info.get(audio_id, {}),
                'background_info': background_info,
                'audio_properties': self._validate_audio_properties(enhanced_vocals_path),
                'performance_metrics': self.performance_metrics.generate_performance_report()
            }
            
            logger.info("✅ Complete pipeline processing finished!")
            logger.info(f"⏱️ Total time: {pipeline_duration:.2f}s")
            logger.info(f"🎯 Enhanced vocals: {enhanced_vocals_path}")
            
            return result
            
        except Exception as e:
            # Record pipeline failure
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.performance_metrics.finish_pipeline_tracking(tracking_data, False, final_memory)
            self.performance_metrics.record_error('general_errors', str(e))
            
            logger.error(f"Complete pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'pipeline_version': '2.0_with_metrics',
                'performance_metrics': self.performance_metrics.generate_performance_report()
            }

    def _try_demucs_separation(self, audio_path: str) -> Dict:
        """Try Demucs separation (Facebook's state-of-the-art)"""
        try:
            logger.debug("🎵 Trying Demucs separation...")
            
            # Check if demucs is available
            try:
                subprocess.run(['python', '-c', 'import demucs'], 
                             capture_output=True, check=True, timeout=10)
            except:
                return {'success': False, 'error': 'Demucs not available'}
            
            # Create output directory
            demucs_output = os.path.join(self.temp_dir, "demucs_output")
            os.makedirs(demucs_output, exist_ok=True)
            
            # Run demucs separation
            cmd = [
                'python', '-m', 'demucs.separate',
                '-o', demucs_output,
                '--two-stems', 'vocals',
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Find output files
                audio_name = os.path.splitext(os.path.basename(audio_path))[0]
                
                possible_paths = [
                    (os.path.join(demucs_output, "htdemucs", audio_name, "vocals.wav"),
                     os.path.join(demucs_output, "htdemucs", audio_name, "no_vocals.wav")),
                    (os.path.join(demucs_output, "mdx_extra", audio_name, "vocals.wav"),
                     os.path.join(demucs_output, "mdx_extra", audio_name, "no_vocals.wav"))
                ]
                
                for vocals_path, background_path in possible_paths:
                    if os.path.exists(vocals_path) and os.path.exists(background_path):
                        self.temp_files.extend([vocals_path, background_path])
                        return {
                            'success': True,
                            'vocals_path': vocals_path,
                            'background_path': background_path,
                            'method': 'demucs'
                        }
            
            return {'success': False, 'error': 'Demucs output files not found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _try_spleeter_separation(self, audio_path: str) -> Dict:
        """Try Spleeter separation (Deezer's model)"""
        try:
            logger.debug("🎵 Trying Spleeter separation...")
            
            try:
                from spleeter.separator import Separator
                
                # Create output directory
                spleeter_output = os.path.join(self.temp_dir, "spleeter_output")
                os.makedirs(spleeter_output, exist_ok=True)
                
                # Initialize separator
                separator = Separator('spleeter:2stems-16kHz')
                
                # Separate audio
                separator.separate_to_file(audio_path, spleeter_output)
                
                # Get output files
                audio_name = os.path.splitext(os.path.basename(audio_path))[0]
                vocals_path = os.path.join(spleeter_output, audio_name, "vocals.wav")
                background_path = os.path.join(spleeter_output, audio_name, "accompaniment.wav")
                
                if os.path.exists(vocals_path) and os.path.exists(background_path):
                    self.temp_files.extend([vocals_path, background_path])
                    return {
                        'success': True,
                        'vocals_path': vocals_path,
                        'background_path': background_path,
                        'method': 'spleeter'
                    }
                
                return {'success': False, 'error': 'Spleeter output files not found'}
                
            except ImportError:
                return {'success': False, 'error': 'Spleeter not installed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _try_librosa_hpss_separation(self, audio_path: str) -> Dict:
        """Try librosa harmonic-percussive separation"""
        try:
            logger.debug("🎵 Trying librosa H/P separation...")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
            
            # Create output paths
            vocals_path = os.path.join(self.temp_dir, "librosa_vocals.wav")
            background_path = os.path.join(self.temp_dir, "librosa_background.wav")
            
            # Save separated audio (percussive as vocals, harmonic as background)
            sf.write(vocals_path, y_percussive, sr)
            sf.write(background_path, y_harmonic, sr)
            
            self.temp_files.extend([vocals_path, background_path])
            
            return {
                'success': True,
                'vocals_path': vocals_path,
                'background_path': background_path,
                'method': 'librosa_hpss'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _try_spectral_separation(self, audio_path: str) -> Dict:
        """Try spectral masking separation (fallback)"""
        try:
            logger.debug("🎵 Trying spectral separation...")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Compute STFT
            D = librosa.stft(y)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Create frequency-based masks
            freq_bins = magnitude.shape[0]
            
            # Vocal mask (mid-frequency range)
            vocal_mask = np.zeros_like(magnitude)
            vocal_start = int(80 * freq_bins / (sr / 2))
            vocal_end = int(1000 * freq_bins / (sr / 2))
            vocal_mask[vocal_start:vocal_end, :] = 1.0
            
            # Smooth the mask
            from scipy.ndimage import gaussian_filter
            vocal_mask = gaussian_filter(vocal_mask, sigma=1.0)
            
            # Background mask is complement
            background_mask = 1.0 - vocal_mask
            
            # Apply masks
            vocals_stft = magnitude * vocal_mask * np.exp(1j * phase)
            background_stft = magnitude * background_mask * np.exp(1j * phase)
            
            # Reconstruct audio
            vocals_audio = librosa.istft(vocals_stft)
            background_audio = librosa.istft(background_stft)
            
            # Save separated audio
            vocals_path = os.path.join(self.temp_dir, "spectral_vocals.wav")
            background_path = os.path.join(self.temp_dir, "spectral_background.wav")
            
            sf.write(vocals_path, vocals_audio, sr)
            sf.write(background_path, background_audio, sr)
            
            self.temp_files.extend([vocals_path, background_path])
            
            return {
                'success': True,
                'vocals_path': vocals_path,
                'background_path': background_path,
                'method': 'spectral_masking'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _validate_audio_properties(self, audio_path: str) -> Dict:
        """Validate audio file properties"""
        try:
            audio_data, sr = librosa.load(audio_path, sr=None)
            return {
                'duration': len(audio_data) / sr,
                'sample_rate': sr,
                'samples': len(audio_data),
                'channels': 1 if audio_data.ndim == 1 else audio_data.shape[0],
                'file_size': os.path.getsize(audio_path),
                'valid': True
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def get_silence_info(self, audio_id: str) -> Dict:
        """Get stored silence information for an audio ID"""
        return self.silence_info.get(audio_id, {})

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        return self.performance_metrics.generate_performance_report()
    
    def get_performance_table(self) -> pd.DataFrame:
        """Get performance metrics as a pandas DataFrame"""
        return self.performance_metrics.create_performance_table()
    
    def print_performance_summary(self):
        """Print a formatted performance summary"""
        report = self.get_performance_metrics()
        
        print("\n" + "="*80)
        print("🎵 AUDIO PROCESSOR PERFORMANCE METRICS")
        print("="*80)
        
        # Pipeline Performance
        pipeline = report.get('pipeline_performance', {})
        print(f"\n📊 PIPELINE PERFORMANCE:")
        print(f"   Total Pipelines Processed: {pipeline.get('total_pipelines_processed', 0)}")
        print(f"   Successful Pipelines: {pipeline.get('successful_pipelines', 0)}")
        print(f"   Failed Pipelines: {pipeline.get('failed_pipelines', 0)}")
        print(f"   Success Rate: {pipeline.get('pipeline_success_rate_percent', 0):.1f}%")
        print(f"   Average Processing Time: {pipeline.get('average_pipeline_time_s', 0):.2f}s")
        print(f"   Total Processing Time: {pipeline.get('total_processing_time_hours', 0):.2f}h")
        
        # Step Performance
        step_perf = report.get('step_performance', {})
        print(f"\n🔧 STEP PERFORMANCE:")
        
        for step_name, metrics in step_perf.items():
            if not metrics.get('no_data', False):
                print(f"\n   {step_name.replace('_', ' ').title()}:")
                print(f"      Success Rate: {metrics.get('success_rate_percent', 0):.1f}%")
                print(f"      Avg Time: {metrics.get('average_processing_time_ms', 0):.1f}ms")
                print(f"      Throughput: {metrics.get('throughput_ops_per_second', 0):.1f} ops/s")
                print(f"      Total Attempts: {metrics.get('total_attempts', 0)}")
                
                # Step-specific metrics
                if step_name == 'vocal_separation':
                    methods = metrics.get('separation_methods_used', {})
                    if methods:
                        print(f"      Methods Used: {methods}")
                    success_rates = metrics.get('method_success_rates', {})
                    if success_rates:
                        print(f"      Method Success Rates: {success_rates}")
        
        # Quality Metrics
        quality = report.get('quality_metrics', {})
        if quality:
            print(f"\n🎯 QUALITY METRICS:")
            print(f"   Average Input Quality: {quality.get('average_input_quality', 0):.3f}")
            print(f"   Average Output Quality: {quality.get('average_output_quality', 0):.3f}")
            print(f"   Quality Improvement: {quality.get('quality_improvement', 0):.3f}")
            print(f"   Quality Improvement %: {quality.get('quality_improvement_percent', 0):.1f}%")
            
            if 'average_snr_improvement_db' in quality:
                print(f"   SNR Improvement: {quality['average_snr_improvement_db']:.2f}dB")
        
        # Resource Metrics
        resources = report.get('resource_metrics', {})
        if resources:
            print(f"\n💻 RESOURCE USAGE:")
            print(f"   Average Memory Usage: {resources.get('average_memory_usage_mb', 0):.1f}MB")
            print(f"   Peak Memory Usage: {resources.get('peak_memory_usage_mb', 0):.1f}MB")
            print(f"   Average CPU Usage: {resources.get('average_cpu_usage_percent', 0):.1f}%")
            print(f"   Total Temp Files Created: {resources.get('total_temp_files_created', 0)}")
        
        # Error Summary
        errors = report.get('error_summary', {})
        total_errors = errors.get('total_errors', 0)
        if total_errors > 0:
            print(f"\n⚠️ ERROR SUMMARY:")
            print(f"   Total Errors: {total_errors}")
            
            for error_type, error_dict in errors.items():
                if error_type != 'total_errors' and error_dict:
                    print(f"   {error_type.replace('_', ' ').title()}:")
                    for error_msg, count in error_dict.items():
                        print(f"      {error_msg}: {count}")
        
        print("\n" + "="*80)
        
        # Performance Table
        try:
            df = self.get_performance_table()
            if not df.empty:
                print("\n📋 DETAILED PERFORMANCE TABLE:")
                print(df.to_string(index=False))
        except Exception as e:
            print(f"⚠️ Could not generate performance table: {e}")
        
        print("\n" + "="*80)
    
    def reset_performance_metrics(self):
        """Reset all performance metrics"""
        self.performance_metrics.reset_metrics()
        logger.info("🔄 Performance metrics reset")

    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            logger.info("🧹 Cleaning up audio processor...")
            
            # Remove temp directory
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            # Remove individual temp files
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    logger.debug(f"Temp file cleanup failed: {e}")
            
            # Clear data structures
            self.temp_files.clear()
            self.silence_info.clear()
            self.background_timestamps.clear()
            
            logger.info("✅ Audio processor cleanup completed")
            
        except Exception as e:
            logger.warning(f"Audio processor cleanup warning: {e}")

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Enhanced Audio Processor Pipeline v2.0 - WITH PERFORMANCE METRICS")
    print("=" * 80)

    # Initialize processor
    processor = EnhancedAudioProcessorV2({
        'sample_rate': 16000,  # Optimal for Whisper/Pyannote
        'preserve_background': True
    })

    print("\n🎯 Pipeline Features:")
    print("   ✅ Video to 16kHz mono audio extraction")
    print("   ✅ Demucs/Spleeter vocal separation")
    print("   ✅ Advanced noise reduction")
    print("   ✅ Speech-optimized filtering")
    print("   ✅ Silence detection and preservation")
    print("   ✅ TTS silence reinsertion")
    print("   ✅ Background music timestamping")
    print("   ✅ Volume normalization")
    print("   ✅ Comprehensive performance tracking")
    print("   ✅ Real-time metrics collection")

    print("\n🔧 Processing Pipeline:")
    print("   1. 🎞️ Extract Audio from Video (FFmpeg)")
    print("   2. 🎧 Separate Vocal and Background (Demucs)")
    print("   3. 🧹 Apply Audio Enhancement")
    print("   4. 🎚️ Volume Normalization")
    print("   5. 🧠 Background Timestamping")
    print("   6. 🔇 TTS Silence Reinsertion")

    print("\n📊 Performance Metrics:")
    print("   ✅ Pipeline success/failure rates")
    print("   ✅ Step-by-step processing times")
    print("   ✅ Audio quality improvements")
    print("   ✅ Resource usage monitoring")
    print("   ✅ Error tracking and analysis")
    print("   ✅ Separation method effectiveness")

    print("\n📊 Ready for:")
    print("   ✅ Speaker Diarization (Pyannote)")
    print("   ✅ Accurate Transcription (Whisper)")
    print("   ✅ Emotion/Gender Detection")
    print("   ✅ TTS and Lip-sync")
    print("   ✅ Performance optimization")

    print("\n" + "=" * 80)
    print("🎬 Enhanced Audio Processor v2.0 with Performance Metrics Ready!")
