"""
ENHANCED Speech Recognition v4.1 - WITH COMPREHENSIVE PERFORMANCE METRICS
Improved segment coverage and retry mechanisms with detailed performance tracking
"""

import whisper
import torch
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa
from transformers import pipeline
import warnings
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter
import psutil
import threading

warnings.filterwarnings("ignore")

from utils.gpu_utils import gpu_manager

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionStepMetrics:
    """Performance metrics for individual transcription steps"""
    step_name: str
    processing_time: float = 0.0
    success: bool = False
    input_duration: float = 0.0
    output_word_count: int = 0
    confidence_score: float = 0.0
    method_used: str = ""
    error_message: str = ""
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for different models and strategies"""
    model_name: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    average_processing_time: float = 0.0
    total_processing_time: float = 0.0
    success_rate: float = 0.0
    average_confidence: float = 0.0
    average_words_per_second: float = 0.0
    quality_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.quality_distribution is None:
            self.quality_distribution = {"high": 0, "medium": 0, "low": 0, "poor": 0, "empty": 0}

@dataclass
class SegmentAnalysisMetrics:
    """Detailed analysis metrics for segments"""
    total_segments: int = 0
    successful_transcriptions: int = 0
    empty_segments: int = 0
    retry_attempts: int = 0
    successful_retries: int = 0
    average_segment_duration: float = 0.0
    average_confidence: float = 0.0
    quality_distribution: Dict[str, int] = None
    failure_reasons: Dict[str, int] = None
    strategy_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.quality_distribution is None:
            self.quality_distribution = {"high": 0, "medium": 0, "low": 0, "poor": 0, "empty": 0}
        if self.failure_reasons is None:
            self.failure_reasons = {}
        if self.strategy_usage is None:
            self.strategy_usage = {}

@dataclass
class OverallPerformanceMetrics:
    """Overall speech recognition performance metrics"""
    total_files_processed: int = 0
    successful_files: int = 0
    total_processing_time: float = 0.0
    total_audio_duration: float = 0.0
    real_time_factor: float = 0.0  # processing_time / audio_duration
    average_transcription_accuracy: float = 0.0
    total_words_transcribed: int = 0
    average_words_per_minute: float = 0.0
    peak_memory_usage_mb: float = 0.0
    average_gpu_utilization: float = 0.0

class ResourceMonitor:
    """Monitor system resource usage during transcription"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        stats = {
            'avg_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'max_cpu_percent': np.max(self.cpu_samples) if self.cpu_samples else 0,
            'avg_memory_mb': np.mean(self.memory_samples) if self.memory_samples else 0,
            'max_memory_mb': np.max(self.memory_samples) if self.memory_samples else 0,
            'avg_gpu_memory_mb': np.mean(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
            'max_gpu_memory_mb': np.max(self.gpu_memory_samples) if self.gpu_memory_samples else 0
        }
        return stats
    
    def _monitor_resources(self):
        """Background resource monitoring"""
        process = psutil.Process()
        while self.monitoring:
            try:
                self.cpu_samples.append(process.cpu_percent())
                self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
                
                # Monitor GPU memory if available
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    self.gpu_memory_samples.append(gpu_memory)
                
                time.sleep(0.1)  # Sample every 100ms
            except:
                break

class SpeechRecognitionPerformanceTracker:
    """Comprehensive performance tracking for speech recognition"""
    
    def __init__(self):
        self.session_start_time = time.time()
        self.step_metrics = []
        self.model_metrics = {}
        self.segment_metrics = SegmentAnalysisMetrics()
        self.overall_metrics = OverallPerformanceMetrics()
        self.transcription_history = []
        self.resource_monitor = ResourceMonitor()
        
        # Initialize model metrics
        models = ['faster_whisper', 'standard_whisper', 'enhanced_transcription', 
                 'alternative_processing', 'retry_context', 'full_alignment']
        for model in models:
            self.model_metrics[model] = ModelPerformanceMetrics(model_name=model)
    
    def start_transcription_tracking(self, audio_path: str, segments_count: int) -> Dict:
        """Start tracking a transcription session"""
        # Get audio duration
        try:
            audio_duration = librosa.get_duration(filename=audio_path)
        except:
            audio_duration = 0.0
        
        session_info = {
            'audio_path': audio_path,
            'segments_count': segments_count,
            'audio_duration': audio_duration,
            'start_time': time.time()
        }
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        return session_info
    
    def end_transcription_tracking(self, session_info: Dict, success: bool, 
                                 result: Dict) -> None:
        """End tracking a transcription session"""
        processing_time = time.time() - session_info['start_time']
        audio_duration = session_info.get('audio_duration', 0)
        
        # Stop resource monitoring
        resource_stats = self.resource_monitor.stop_monitoring()
        
        # Update overall metrics
        self.overall_metrics.total_files_processed += 1
        self.overall_metrics.total_processing_time += processing_time
        self.overall_metrics.total_audio_duration += audio_duration
        
        if success:
            self.overall_metrics.successful_files += 1
        
        # Calculate real-time factor
        if audio_duration > 0:
            rtf = processing_time / audio_duration
            self.overall_metrics.real_time_factor = (
                (self.overall_metrics.real_time_factor * (self.overall_metrics.total_files_processed - 1) + rtf) /
                self.overall_metrics.total_files_processed
            )
        
        # Update memory usage
        self.overall_metrics.peak_memory_usage_mb = max(
            self.overall_metrics.peak_memory_usage_mb,
            resource_stats.get('max_memory_mb', 0)
        )
        
        # Update GPU utilization
        if resource_stats.get('avg_gpu_memory_mb', 0) > 0:
            self.overall_metrics.average_gpu_utilization = resource_stats.get('avg_gpu_memory_mb', 0)
        
        # Process segment results
        if result.get('segments'):
            self._process_segment_results(result['segments'], processing_time)
        
        # Store transcription history
        self.transcription_history.append({
            'timestamp': datetime.now().isoformat(),
            'audio_path': session_info['audio_path'],
            'processing_time': processing_time,
            'audio_duration': audio_duration,
            'success': success,
            'segments_count': session_info['segments_count'],
            'resource_stats': resource_stats,
            'result_stats': result.get('stats', {})
        })
    
    def record_step_performance(self, step_name: str, processing_time: float, 
                              success: bool, method_used: str = "", 
                              confidence: float = 0.0, word_count: int = 0,
                              input_duration: float = 0.0, error_msg: str = "") -> None:
        """Record performance of individual processing steps"""
        
        # Update model metrics
        if method_used and method_used in self.model_metrics:
            metrics = self.model_metrics[method_used]
            metrics.total_attempts += 1
            metrics.total_processing_time += processing_time
            
            if success:
                metrics.successful_attempts += 1
                if confidence > 0:
                    total_conf = metrics.average_confidence * (metrics.successful_attempts - 1) + confidence
                    metrics.average_confidence = total_conf / metrics.successful_attempts
                
                if input_duration > 0 and word_count > 0:
                    wps = word_count / input_duration
                    total_wps = metrics.average_words_per_second * (metrics.successful_attempts - 1) + wps
                    metrics.average_words_per_second = total_wps / metrics.successful_attempts
            else:
                metrics.failed_attempts += 1
            
            # Update success rate and average processing time
            metrics.success_rate = (metrics.successful_attempts / metrics.total_attempts) * 100
            metrics.average_processing_time = metrics.total_processing_time / metrics.total_attempts
        
        # Create step metrics
        step_metrics = TranscriptionStepMetrics(
            step_name=step_name,
            processing_time=processing_time,
            success=success,
            input_duration=input_duration,
            output_word_count=word_count,
            confidence_score=confidence,
            method_used=method_used,
            error_message=error_msg
        )
        
        self.step_metrics.append(step_metrics)
    
    def record_segment_analysis(self, segment: Dict, strategy_used: str = "") -> None:
        """Record analysis of individual segments"""
        self.segment_metrics.total_segments += 1
        
        # Check if segment has text
        if segment.get('text', '').strip():
            self.segment_metrics.successful_transcriptions += 1
            
            # Update word count
            word_count = len(segment.get('text', '').split())
            self.overall_metrics.total_words_transcribed += word_count
        else:
            self.segment_metrics.empty_segments += 1
            
            # Track failure reason
            failure_reason = segment.get('failure_reason', 'unknown')
            self.segment_metrics.failure_reasons[failure_reason] = (
                self.segment_metrics.failure_reasons.get(failure_reason, 0) + 1
            )
        
        # Track quality
        quality = segment.get('quality', 'unknown')
        if quality in self.segment_metrics.quality_distribution:
            self.segment_metrics.quality_distribution[quality] += 1
        
        # Track strategy usage
        method = segment.get('method', 'unknown')
        self.segment_metrics.strategy_usage[method] = (
            self.segment_metrics.strategy_usage.get(method, 0) + 1
        )
        
        # Update averages
        duration = segment.get('duration', 0)
        confidence = segment.get('confidence', 0)
        
        total_duration = (self.segment_metrics.average_segment_duration * 
                         (self.segment_metrics.total_segments - 1) + duration)
        self.segment_metrics.average_segment_duration = total_duration / self.segment_metrics.total_segments
        
        if confidence > 0:
            total_conf = (self.segment_metrics.average_confidence * 
                         (self.segment_metrics.total_segments - 1) + confidence)
            self.segment_metrics.average_confidence = total_conf / self.segment_metrics.total_segments
    
    def record_retry_attempt(self, success: bool) -> None:
        """Record retry attempt statistics"""
        self.segment_metrics.retry_attempts += 1
        if success:
            self.segment_metrics.successful_retries += 1
    
    def _process_segment_results(self, segments: List[Dict], total_processing_time: float) -> None:
        """Process segment results for comprehensive analysis"""
        for segment in segments:
            self.record_segment_analysis(segment)
        
        # Calculate words per minute
        if self.overall_metrics.total_audio_duration > 0:
            self.overall_metrics.average_words_per_minute = (
                self.overall_metrics.total_words_transcribed / 
                (self.overall_metrics.total_audio_duration / 60)
            )
        
        # Calculate transcription accuracy (based on confidence scores)
        confidences = [s.get('confidence', 0) for s in segments if s.get('text', '').strip()]
        if confidences:
            session_accuracy = np.mean(confidences)
            total_accuracy = (self.overall_metrics.average_transcription_accuracy * 
                            (self.overall_metrics.total_files_processed - 1) + session_accuracy)
            self.overall_metrics.average_transcription_accuracy = (
                total_accuracy / self.overall_metrics.total_files_processed
            )
    
    def calculate_model_comparison(self) -> Dict:
        """Calculate comparative performance metrics between models"""
        comparison = {}
        
        for model_name, metrics in self.model_metrics.items():
            if metrics.total_attempts > 0:
                # Calculate efficiency score
                efficiency_score = 0
                if metrics.average_processing_time > 0:
                    efficiency_score = metrics.success_rate / metrics.average_processing_time
                
                comparison[model_name] = {
                    'success_rate': metrics.success_rate,
                    'average_processing_time': metrics.average_processing_time,
                    'average_confidence': metrics.average_confidence,
                    'average_words_per_second': metrics.average_words_per_second,
                    'total_attempts': metrics.total_attempts,
                    'efficiency_score': efficiency_score,
                    'reliability_score': metrics.success_rate * metrics.average_confidence / 100
                }
        
        return comparison
    
    def calculate_strategy_effectiveness(self) -> Dict:
        """Calculate effectiveness of different processing strategies"""
        strategy_stats = {}
        
        for strategy, count in self.segment_metrics.strategy_usage.items():
            if count > 0:
                # Calculate success rate for this strategy
                successful_with_strategy = 0
                total_with_strategy = 0
                
                for step in self.step_metrics:
                    if step.method_used == strategy:
                        total_with_strategy += 1
                        if step.success:
                            successful_with_strategy += 1
                
                success_rate = (successful_with_strategy / total_with_strategy * 100) if total_with_strategy > 0 else 0
                
                strategy_stats[strategy] = {
                    'usage_count': count,
                    'success_rate': success_rate,
                    'usage_percentage': (count / self.segment_metrics.total_segments * 100)
                }
        
        return strategy_stats
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        session_duration = time.time() - self.session_start_time
        
        # Model comparison
        model_comparison = self.calculate_model_comparison()
        
        # Strategy effectiveness
        strategy_effectiveness = self.calculate_strategy_effectiveness()
        
        # Find best performing model
        best_model = None
        best_score = 0
        for model, metrics in model_comparison.items():
            if metrics['reliability_score'] > best_score:
                best_score = metrics['reliability_score']
                best_model = model
        
        # Calculate processing efficiency
        processing_efficiency = {}
        if self.overall_metrics.total_audio_duration > 0:
            processing_efficiency = {
                'real_time_factor': self.overall_metrics.real_time_factor,
                'words_per_minute': self.overall_metrics.average_words_per_minute,
                'hours_processed_per_hour': 1 / self.overall_metrics.real_time_factor if self.overall_metrics.real_time_factor > 0 else 0,
                'average_transcription_accuracy': self.overall_metrics.average_transcription_accuracy
            }
        
        # Quality analysis
        quality_analysis = {
            'segment_success_rate': (self.segment_metrics.successful_transcriptions / 
                                   self.segment_metrics.total_segments * 100) if self.segment_metrics.total_segments > 0 else 0,
            'retry_success_rate': (self.segment_metrics.successful_retries / 
                                 self.segment_metrics.retry_attempts * 100) if self.segment_metrics.retry_attempts > 0 else 0,
            'average_segment_confidence': self.segment_metrics.average_confidence,
            'quality_distribution': dict(self.segment_metrics.quality_distribution),
            'failure_analysis': dict(self.segment_metrics.failure_reasons)
        }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'session_duration_seconds': session_duration,
            'overall_metrics': asdict(self.overall_metrics),
            'segment_analysis': asdict(self.segment_metrics),
            'model_comparison': model_comparison,
            'strategy_effectiveness': strategy_effectiveness,
            'best_performing_model': {
                'model_name': best_model,
                'reliability_score': best_score
            },
            'processing_efficiency': processing_efficiency,
            'quality_analysis': quality_analysis,
            'individual_step_metrics': [asdict(metric) for metric in self.step_metrics],
            'transcription_history': self.transcription_history,
            'recommendations': self._generate_recommendations(model_comparison, strategy_effectiveness, quality_analysis)
        }
        
        return report
    
    def _generate_recommendations(self, model_comparison: Dict, strategy_effectiveness: Dict, 
                                quality_analysis: Dict) -> List[str]:
        """Generate performance-based recommendations"""
        recommendations = []
        
        # Model recommendations
        if model_comparison:
            best_model = max(model_comparison.items(), key=lambda x: x[1]['reliability_score'])
            recommendations.append(
                f"Best performing model: {best_model[0]} "
                f"(reliability score: {best_model[1]['reliability_score']:.2f})"
            )
            
            # Check for underperforming models
            poor_models = [
                name for name, metrics in model_comparison.items()
                if metrics['success_rate'] < 70 and metrics['total_attempts'] > 5
            ]
            if poor_models:
                recommendations.append(
                    f"Consider optimizing or replacing underperforming models: {', '.join(poor_models)}"
                )
        
        # Strategy recommendations
        if strategy_effectiveness:
            best_strategy = max(strategy_effectiveness.items(), key=lambda x: x[1]['success_rate'])
            recommendations.append(
                f"Most effective strategy: {best_strategy[0]} "
                f"(success rate: {best_strategy[1]['success_rate']:.1f}%)"
            )
        
        # Quality recommendations
        if quality_analysis['segment_success_rate'] < 80:
            recommendations.append(
                "Low segment success rate - consider adjusting confidence thresholds or preprocessing"
            )
        
        if quality_analysis['retry_success_rate'] < 50 and self.segment_metrics.retry_attempts > 10:
            recommendations.append(
                "Low retry success rate - review retry strategies and parameters"
            )
        
        # Performance recommendations
        if self.overall_metrics.real_time_factor > 2.0:
            recommendations.append(
                "High real-time factor - consider using faster models or optimizing processing pipeline"
            )
        
        if self.overall_metrics.average_transcription_accuracy < 0.7:
            recommendations.append(
                "Low transcription accuracy - consider using larger models or improving audio preprocessing"
            )
        
        return recommendations

class EnhancedSpeechRecognition:
    def __init__(self, model_size: str = "large-v3"):
        self.device = gpu_manager.get_device()
        self.model_size = model_size
        self.model = None
        self.faster_whisper = None
        self.confidence_threshold = 0.4  # Lowered from 0.6 for better coverage
        self.min_segment_duration = 0.1  # Reduced from 0.3 for shorter segments
        self.max_retries = 3  # Added retry mechanism
        
        # Initialize performance tracking
        self.performance_tracker = SpeechRecognitionPerformanceTracker()
        
        self._load_models()
    
    def _load_models(self):
        """Load enhanced speech recognition models"""
        try:
            logger.info(f"🎤 Loading Enhanced Speech Recognition v4.1 with PERFORMANCE METRICS...")
            logger.info(f"   Model: Whisper {self.model_size}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Min segment duration: {self.min_segment_duration}s")
            logger.info(f"   Confidence threshold: {self.confidence_threshold}")
            
            # Clear GPU cache before loading
            gpu_manager.clear_gpu_cache()
            
            # Try to load faster-whisper first (more efficient)
            try:
                from faster_whisper import WhisperModel
                self.faster_whisper = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "float32"
                )
                logger.info("✅ Faster-Whisper loaded successfully")
            except ImportError:
                logger.info("📦 Faster-Whisper not available, using standard Whisper")
            
            # Load standard Whisper as fallback
            self.model = gpu_manager.safe_model_load(
                whisper.load_model,
                self.model_size,
                device=self.device
            )
            
            logger.info("✅ Enhanced Speech Recognition with Performance Metrics ready!")
            
        except Exception as e:
            logger.error(f"Failed to load speech recognition models: {e}")
            # Try smaller model as fallback
            if self.model_size != "base":
                logger.info("🔄 Trying smaller model as fallback...")
                self.model_size = "base"
                self._load_models()
            else:
                raise Exception(f"Speech recognition model loading failed: {e}")
    
    def transcribe_enhanced(self, audio_path: str, diarization_segments: List[Dict], 
                          language: Optional[str] = None) -> Dict:
        """
        ENHANCED: Transcribe audio with improved segment coverage and retry mechanisms
        """
        # Start performance tracking
        session_info = self.performance_tracker.start_transcription_tracking(
            audio_path, len(diarization_segments)
        )
        
        try:
            if not self.model and not self.faster_whisper:
                raise Exception("No speech recognition model loaded")
            
            logger.info(f"🎤 Enhanced transcription starting v4.1...")
            logger.info(f"   Segments to process: {len(diarization_segments)}")
            logger.info(f"   Language: {language or 'auto-detect'}")
            
            # Get full transcription for reference and language detection
            full_transcription_start = time.time()
            full_transcription = self._get_enhanced_transcription(audio_path, language)
            full_transcription_time = time.time() - full_transcription_start
            
            # Record full transcription performance
            self.performance_tracker.record_step_performance(
                "full_transcription", full_transcription_time, True, 
                "faster_whisper" if self.faster_whisper else "standard_whisper",
                0.8, len(full_transcription.get("text", "").split()),
                session_info.get('audio_duration', 0)
            )
            
            detected_language = full_transcription.get("language", language or "en")
            
            logger.info(f"🌍 Detected language: {detected_language}")
            
            # Process each segment with enhanced methods and retry logic
            processed_segments = []
            successful_transcriptions = 0
            failed_segments = []
            retry_segments = []
            
            # First pass: Process all segments
            for i, diar_segment in enumerate(diarization_segments):
                try:
                    logger.debug(f"Processing segment {i+1}/{len(diarization_segments)}")
                    
                    # Enhanced segment processing with multiple strategies
                    segment_start_time = time.time()
                    enhanced_segment = self._process_segment_with_strategies(
                        audio_path, diar_segment, i, detected_language
                    )
                    segment_processing_time = time.time() - segment_start_time
                    
                    # Record segment performance
                    self.performance_tracker.record_step_performance(
                        f"segment_{i}", segment_processing_time, 
                        bool(enhanced_segment["text"].strip()),
                        enhanced_segment.get("method", "unknown"),
                        enhanced_segment.get("confidence", 0),
                        len(enhanced_segment.get("text", "").split()),
                        enhanced_segment.get("duration", 0)
                    )
                    
                    # Record segment analysis
                    self.performance_tracker.record_segment_analysis(
                        enhanced_segment, enhanced_segment.get("method", "")
                    )
                    
                    if enhanced_segment["text"].strip():
                        successful_transcriptions += 1
                        processed_segments.append(enhanced_segment)
                    else:
                        # Mark for retry with different parameters
                        retry_segments.append((i, diar_segment))
                        processed_segments.append(enhanced_segment)
                    
                except Exception as e:
                    logger.warning(f"Segment {i} processing failed: {e}")
                    # Create fallback segment and mark for retry
                    fallback_segment = self._create_fallback_segment(diar_segment, i)
                    processed_segments.append(fallback_segment)
                    failed_segments.append((i, diar_segment, str(e)))
                    
                    # Record failed segment
                    self.performance_tracker.record_segment_analysis(fallback_segment)
            
            # Second pass: Retry failed segments with different strategies
            if retry_segments or failed_segments:
                logger.info(f"🔄 Retrying {len(retry_segments + failed_segments)} failed segments...")
                
                for segment_info in retry_segments + [(i, seg, "initial_failure") for i, seg, _ in failed_segments]:
                    if len(segment_info) == 2:
                        i, diar_segment = segment_info
                        error = "empty_result"
                    else:
                        i, diar_segment, error = segment_info
                    
                    try:
                        # Record retry attempt
                        self.performance_tracker.record_retry_attempt(False)  # Will update if successful
                        
                        # Retry with more aggressive parameters
                        retry_start_time = time.time()
                        retry_result = self._retry_segment_transcription(
                            audio_path, diar_segment, i, detected_language
                        )
                        retry_processing_time = time.time() - retry_start_time
                        
                        if retry_result and retry_result["text"].strip():
                            # Update the processed segment
                            processed_segments[i] = retry_result
                            successful_transcriptions += 1
                            
                            # Record successful retry
                            self.performance_tracker.record_retry_attempt(True)
                            self.performance_tracker.record_step_performance(
                                f"retry_segment_{i}", retry_processing_time, True,
                                retry_result.get("method", "retry"),
                                retry_result.get("confidence", 0),
                                len(retry_result.get("text", "").split()),
                                retry_result.get("duration", 0)
                            )
                            
                            logger.info(f"✅ Retry successful for segment {i}")
                        else:
                            logger.debug(f"⚠️ Retry failed for segment {i}")
                            
                            # Record failed retry
                            self.performance_tracker.record_step_performance(
                                f"retry_segment_{i}", retry_processing_time, False,
                                "retry_failed", 0, 0, 0, "retry_unsuccessful"
                            )
                    
                    except Exception as e:
                        logger.debug(f"Retry failed for segment {i}: {e}")
                        self.performance_tracker.record_step_performance(
                            f"retry_segment_{i}", 0, False, "retry_exception", 0, 0, 0, str(e)
                        )
            
            # Third pass: Use full transcription alignment for remaining empty segments
            alignment_start_time = time.time()
            processed_segments = self._align_with_full_transcription(
                processed_segments, full_transcription, audio_path
            )
            alignment_time = time.time() - alignment_start_time
            
            # Record alignment performance
            self.performance_tracker.record_step_performance(
                "full_alignment", alignment_time, True, "full_alignment"
            )
            
            # Recount successful transcriptions
            successful_transcriptions = sum(1 for s in processed_segments if s.get('text', '').strip())
            
            # Clear GPU cache after transcription
            gpu_manager.clear_gpu_cache()
            
            # Calculate final statistics
            stats = {
                'total_segments': len(processed_segments),
                'successful_transcriptions': successful_transcriptions,
                'success_rate': (successful_transcriptions/len(processed_segments)*100) if processed_segments else 0,
                'average_confidence': np.mean([s.get('confidence', 0) for s in processed_segments]) if processed_segments else 0,
                'failed_segments': len(processed_segments) - successful_transcriptions,
                'retry_attempts': len(retry_segments) + len(failed_segments)
            }
            
            logger.info(f"✅ Enhanced transcription completed!")
            logger.info(f"   Total segments: {len(processed_segments)}")
            logger.info(f"   Successful transcriptions: {successful_transcriptions}")
            logger.info(f"   Success rate: {stats['success_rate']:.1f}%")
            logger.info(f"   Failed segments: {stats['failed_segments']}")
            
            result = {
                'success': True,
                'segments': processed_segments,
                'detected_language': detected_language,
                'stats': stats,
                'performance_metrics': self.get_performance_metrics()
            }
            
            # End performance tracking
            self.performance_tracker.end_transcription_tracking(session_info, True, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced transcription failed: {e}")
            gpu_manager.clear_gpu_cache()
            
            # Return fallback result
            fallback_segments = self._create_fallback_segments(diarization_segments)
            
            result = {
                'success': False,
                'error': str(e),
                'segments': fallback_segments,
                'detected_language': language or 'en',
                'stats': {
                    'total_segments': len(fallback_segments),
                    'successful_transcriptions': 0,
                    'success_rate': 0,
                    'average_confidence': 0
                },
                'performance_metrics': self.get_performance_metrics()
            }
            
            # End performance tracking
            self.performance_tracker.end_transcription_tracking(session_info, False, result)
            
            return result
    
    def _process_segment_with_strategies(self, audio_path: str, diar_segment: Dict, 
                                       segment_id: int, language: str) -> Dict:
        """Process segment with multiple strategies for better coverage"""
        try:
            start_time = float(diar_segment['start'])
            end_time = float(diar_segment['end'])
            duration = end_time - start_time
            
            # Strategy 1: Standard processing (reduced minimum duration)
            if duration >= self.min_segment_duration:
                strategy_start = time.time()
                result = self._process_segment_enhanced(audio_path, diar_segment, segment_id, language)
                strategy_time = time.time() - strategy_start
                
                self.performance_tracker.record_step_performance(
                    "standard_processing", strategy_time, bool(result["text"].strip()),
                    "enhanced_transcription", result.get("confidence", 0),
                    len(result.get("text", "").split()), duration
                )
                
                if result["text"].strip():
                    return result
            
            # Strategy 2: Extended segment (add padding for very short segments)
            if duration < 0.5:
                strategy_start = time.time()
                extended_segment = {
                    **diar_segment,
                    'start': max(0, start_time - 0.2),
                    'end': end_time + 0.2
                }
                result = self._process_segment_enhanced(audio_path, extended_segment, segment_id, language)
                strategy_time = time.time() - strategy_start
                
                self.performance_tracker.record_step_performance(
                    "extended_segment", strategy_time, bool(result["text"].strip()),
                    "extended_segment", result.get("confidence", 0),
                    len(result.get("text", "").split()), duration
                )
                
                if result["text"].strip():
                    # Restore original timing
                    result['start'] = start_time
                    result['end'] = end_time
                    result['duration'] = duration
                    result['method'] = 'extended_segment'
                    return result
            
            # Strategy 3: Different audio processing parameters
            strategy_start = time.time()
            result = self._process_segment_alternative(audio_path, diar_segment, segment_id, language)
            strategy_time = time.time() - strategy_start
            
            self.performance_tracker.record_step_performance(
                "alternative_processing", strategy_time, bool(result["text"].strip()),
                "alternative_processing", result.get("confidence", 0),
                len(result.get("text", "").split()), duration
            )
            
            if result["text"].strip():
                return result
            
            # Return empty segment if all strategies fail
            return self._create_empty_segment(diar_segment, segment_id, "all_strategies_failed")
            
        except Exception as e:
            logger.debug(f"All strategies failed for segment {segment_id}: {e}")
            return self._create_empty_segment(diar_segment, segment_id, "strategy_exception")
    
    def _process_segment_alternative(self, audio_path: str, diar_segment: Dict, 
                                   segment_id: int, language: str) -> Dict:
        """Alternative segment processing with different parameters"""
        try:
            start_time = float(diar_segment['start'])
            end_time = float(diar_segment['end'])
            duration = end_time - start_time
            
            # Extract segment with alternative parameters
            segment_audio = self._extract_segment_alternative(audio_path, start_time, end_time)
            
            if not segment_audio:
                return self._create_empty_segment(diar_segment, segment_id, "alt_extraction_failed")
            
            try:
                # Transcribe with more permissive parameters
                segment_text, confidence = self._transcribe_segment_permissive(segment_audio, language)
                
                # Clean up temp file
                Path(segment_audio).unlink(missing_ok=True)
                
                # Create segment with alternative method marker
                enhanced_segment = {
                    "id": segment_id,
                    "start": start_time,
                    "end": end_time,
                    "duration": duration,
                    "text": segment_text.strip(),
                    "speaker": diar_segment.get('speaker', f'SPEAKER_{segment_id:02d}'),
                    "confidence": confidence,
                    "language": language,
                    "method": "alternative_processing",
                    "quality": self._assess_transcription_quality(segment_text, confidence, duration)
                }
                
                return enhanced_segment
                
            except Exception as e:
                logger.debug(f"Alternative transcription failed: {e}")
                if segment_audio and Path(segment_audio).exists():
                    Path(segment_audio).unlink(missing_ok=True)
                return self._create_empty_segment(diar_segment, segment_id, "alt_transcription_failed")
        
        except Exception as e:
            logger.debug(f"Alternative processing failed: {e}")
            return self._create_empty_segment(diar_segment, segment_id, "alt_processing_failed")
    
    def _extract_segment_alternative(self, audio_path: str, start_time: float, end_time: float) -> Optional[str]:
        """Alternative audio extraction with different parameters"""
        try:
            duration = end_time - start_time
            if duration < 0.05:  # Very short segments
                return None
            
            temp_audio = tempfile.mktemp(suffix='.wav')
            
            # More aggressive audio processing for difficult segments
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(max(0, start_time - 0.05)),  # Minimal padding
                '-i', str(audio_path),
                '-t', str(duration + 0.1),  # Small padding
                '-af', 'highpass=f=50,lowpass=f=12000,volume=3.0,compand,dynaudnorm=g=3',  # More aggressive
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-loglevel', 'error',
                temp_audio
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if Path(temp_audio).exists() and Path(temp_audio).stat().st_size > 500:  # Lower size threshold
                return temp_audio
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Alternative extraction failed: {e}")
            return None
    
    def _transcribe_segment_permissive(self, audio_path: str, language: str) -> Tuple[str, float]:
        """Transcribe segment with more permissive parameters"""
        try:
            # Use faster-whisper if available with permissive settings
            if self.faster_whisper:
                segments, info = self.faster_whisper.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=False,
                    vad_filter=False,  # Disable VAD for difficult segments
                    temperature=[0.0, 0.2, 0.4],  # Multiple temperatures
                    compression_ratio_threshold=2.4,  # More permissive
                    log_prob_threshold=-1.5,  # More permissive
                    no_speech_threshold=0.8  # More permissive
                )
                
                text = " ".join([segment.text for segment in segments]).strip()
                # Calculate confidence from segment probabilities
                confidences = []
                for segment in segments:
                    if hasattr(segment, 'avg_logprob'):
                        confidences.append(np.exp(segment.avg_logprob))
                
                confidence = np.mean(confidences) if confidences else 0.5
                return text, float(confidence)
            
            # Fallback to standard Whisper with permissive settings
            with torch.no_grad():
                result = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=False,
                    verbose=False,
                    temperature=[0.0, 0.2, 0.4, 0.6],  # Multiple temperatures
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.5,
                    no_speech_threshold=0.8,
                    fp16=self.device == "cuda"
                )
            
            text = result.get("text", "").strip()
            
            # Calculate confidence from segments
            segments = result.get("segments", [])
            if segments:
                confidences = [np.exp(seg.get("avg_logprob", -1.0)) for seg in segments]
                confidence = np.mean(confidences)
            else:
                confidence = 0.3  # Lower default confidence
            
            return text, float(confidence)
            
        except Exception as e:
            logger.debug(f"Permissive transcription failed: {e}")
            return "", 0.1
    
    def _retry_segment_transcription(self, audio_path: str, diar_segment: Dict, 
                                   segment_id: int, language: str) -> Optional[Dict]:
        """Retry segment transcription with different strategies"""
        try:
            start_time = float(diar_segment['start'])
            end_time = float(diar_segment['end'])
            duration = end_time - start_time
            
            # Retry strategy 1: Larger context window
            context_start = max(0, start_time - 0.5)
            context_end = end_time + 0.5
            context_segment = {
                **diar_segment,
                'start': context_start,
                'end': context_end
            }
            
            result = self._process_segment_alternative(audio_path, context_segment, segment_id, language)
            if result["text"].strip():
                # Restore original timing
                result['start'] = start_time
                result['end'] = end_time
                result['duration'] = duration
                result['method'] = 'retry_context'
                return result
            
            # Retry strategy 2: Use full audio transcription alignment
            full_result = self._extract_from_full_transcription(
                audio_path, start_time, end_time, language, segment_id
            )
            if full_result and full_result["text"].strip():
                return full_result
            
            return None
            
        except Exception as e:
            logger.debug(f"Retry transcription failed: {e}")
            return None
    
    def _extract_from_full_transcription(self, audio_path: str, start_time: float, 
                                       end_time: float, language: str, segment_id: int) -> Optional[Dict]:
        """Extract text from full transcription based on timing"""
        try:
            # Get full transcription with word timestamps
            if self.faster_whisper:
                segments, info = self.faster_whisper.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=True
                )
                
                # Find words within the time range
                segment_words = []
                for segment in segments:
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            if start_time <= word.start <= end_time or start_time <= word.end <= end_time:
                                segment_words.append(word.word)
                
                if segment_words:
                    text = " ".join(segment_words).strip()
                    return {
                        "id": segment_id,
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "text": text,
                        "speaker": f'SPEAKER_{segment_id:02d}',
                        "confidence": 0.6,
                        "language": language,
                        "method": "full_transcription_alignment",
                        "quality": "medium"
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Full transcription extraction failed: {e}")
            return None
    
    def _align_with_full_transcription(self, segments: List[Dict], full_transcription: Dict, 
                                     audio_path: str) -> List[Dict]:
        """Align empty segments with full transcription"""
        try:
            if not full_transcription.get("segments"):
                return segments
            
            full_segments = full_transcription["segments"]
            
            for i, segment in enumerate(segments):
                if not segment.get("text", "").strip():
                    # Try to find overlapping text from full transcription
                    start_time = segment["start"]
                    end_time = segment["end"]
                    
                    overlapping_text = []
                    for full_seg in full_segments:
                        full_start = full_seg.get("start", 0)
                        full_end = full_seg.get("end", 0)
                        
                        # Check for overlap
                        if (start_time <= full_start <= end_time or 
                            start_time <= full_end <= end_time or
                            full_start <= start_time <= full_end):
                            
                            # Calculate overlap ratio
                            overlap_start = max(start_time, full_start)
                            overlap_end = min(end_time, full_end)
                            overlap_duration = overlap_end - overlap_start
                            
                            if overlap_duration > 0:
                                # Extract proportional text
                                full_text = full_seg.get("text", "")
                                if full_text.strip():
                                    overlapping_text.append(full_text.strip())
                    
                    if overlapping_text:
                        combined_text = " ".join(overlapping_text).strip()
                        segments[i]["text"] = combined_text
                        segments[i]["confidence"] = 0.5
                        segments[i]["method"] = "full_alignment"
                        segments[i]["quality"] = "medium"
            
            return segments
            
        except Exception as e:
            logger.debug(f"Full transcription alignment failed: {e}")
            return segments
    
    def _get_enhanced_transcription(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """Get enhanced full transcription with better accuracy"""
        try:
            # Use faster-whisper if available
            if self.faster_whisper:
                segments, info = self.faster_whisper.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),  # Reduced from 500
                    temperature=[0.0, 0.2],  # Multiple temperatures
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6
                )
                
                # Convert to standard format
                result = {
                    "text": " ".join([segment.text for segment in segments]),
                    "language": info.language,
                    "segments": [
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                            "words": [
                                {"word": word.word, "start": word.start, "end": word.end, "probability": word.probability}
                                for word in segment.words
                            ] if segment.words else []
                        }
                        for segment in segments
                    ]
                }
                return result
            
            # Fallback to standard Whisper
            with torch.no_grad():
                result = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=True,
                    verbose=False,
                    temperature=[0.0, 0.2, 0.4],  # Multiple temperatures
                    best_of=2,
                    beam_size=5 if self.device == "cuda" else 1,
                    patience=1.0,
                    length_penalty=1.0,
                    suppress_tokens="-1",
                    condition_on_previous_text=True,
                    fp16=self.device == "cuda",
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6
                )
            
            return result
            
        except Exception as e:
            logger.warning(f"Enhanced transcription failed: {e}")
            return {"text": "", "language": "en", "segments": []}
    
    def _process_segment_enhanced(self, audio_path: str, diar_segment: Dict, 
                                segment_id: int, language: str) -> Dict:
        """Process individual segment with enhanced methods"""
        try:
            start_time = float(diar_segment['start'])
            end_time = float(diar_segment['end'])
            duration = end_time - start_time
            
            # Reduced minimum duration threshold
            if duration < self.min_segment_duration:
                return self._create_empty_segment(diar_segment, segment_id, "too_short")
            
            # Extract and enhance audio segment
            segment_audio = self._extract_enhanced_segment(audio_path, start_time, end_time)
            
            if not segment_audio:
                return self._create_empty_segment(diar_segment, segment_id, "extraction_failed")
            
            try:
                # Transcribe with enhanced method
                segment_text, confidence = self._transcribe_segment_enhanced(segment_audio, language)
                
                # Clean up temp file
                Path(segment_audio).unlink(missing_ok=True)
                
                # Create enhanced segment
                enhanced_segment = {
                    "id": segment_id,
                    "start": start_time,
                    "end": end_time,
                    "duration": duration,
                    "text": segment_text.strip(),
                    "speaker": diar_segment.get('speaker', f'SPEAKER_{segment_id:02d}'),
                    "confidence": confidence,
                    "language": language,
                    "method": "enhanced_transcription",
                    "quality": self._assess_transcription_quality(segment_text, confidence, duration)
                }
                
                return enhanced_segment
                
            except Exception as e:
                logger.debug(f"Segment transcription failed: {e}")
                if segment_audio and Path(segment_audio).exists():
                    Path(segment_audio).unlink(missing_ok=True)
                return self._create_empty_segment(diar_segment, segment_id, "transcription_failed")
        
        except Exception as e:
            logger.debug(f"Enhanced segment processing failed: {e}")
            return self._create_empty_segment(diar_segment, segment_id, "processing_failed")
    
    def _extract_enhanced_segment(self, audio_path: str, start_time: float, end_time: float) -> Optional[str]:
        """Extract audio segment with enhanced preprocessing"""
        try:
            duration = end_time - start_time
            if duration < 0.05:  # Very short threshold
                return None
            
            temp_audio = tempfile.mktemp(suffix='.wav')
            
            # Enhanced FFmpeg command with noise reduction and normalization
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(max(0, start_time - 0.1)),  # Small padding
                '-i', str(audio_path),
                '-t', str(duration + 0.2),  # Small padding
                '-af', 'highpass=f=80,lowpass=f=8000,volume=2.0,dynaudnorm',  # Enhanced filtering
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-loglevel', 'error',
                temp_audio
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if Path(temp_audio).exists() and Path(temp_audio).stat().st_size > 500:  # Reduced minimum size
                return temp_audio
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Enhanced segment extraction failed: {e}")
            return None
    
    def _transcribe_segment_enhanced(self, audio_path: str, language: str) -> Tuple[str, float]:
        """Transcribe segment with enhanced accuracy"""
        try:
            # Use faster-whisper if available
            if self.faster_whisper:
                segments, info = self.faster_whisper.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=False,
                    vad_filter=True,
                    temperature=[0.0, 0.2],  # Multiple temperatures
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6
                )
                
                text = " ".join([segment.text for segment in segments]).strip()
                # Calculate confidence from segment probabilities
                confidences = []
                for segment in segments:
                    if hasattr(segment, 'avg_logprob'):
                        confidences.append(np.exp(segment.avg_logprob))
                
                confidence = np.mean(confidences) if confidences else 0.5
                return text, float(confidence)
            
            # Fallback to standard Whisper
            with torch.no_grad():
                result = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    word_timestamps=False,
                    verbose=False,
                    temperature=[0.0, 0.2],  # Multiple temperatures
                    fp16=self.device == "cuda",
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6
                )
            
            text = result.get("text", "").strip()
            
            # Calculate confidence from segments
            segments = result.get("segments", [])
            if segments:
                confidences = [np.exp(seg.get("avg_logprob", -1.0)) for seg in segments]
                confidence = np.mean(confidences)
            else:
                confidence = 0.4  # Lower default confidence
            
            return text, float(confidence)
            
        except Exception as e:
            logger.debug(f"Enhanced segment transcription failed: {e}")
            return "", 0.1
    
    def _assess_transcription_quality(self, text: str, confidence: float, duration: float) -> str:
        """Assess transcription quality with enhanced metrics"""
        try:
            if not text.strip():
                return "empty"
            
            # Length-based quality assessment
            words = text.split()
            words_per_second = len(words) / duration if duration > 0 else 0
            
            # Quality scoring
            quality_score = confidence
            
            # Adjust for speech rate (normal: 2-4 words/second)
            if 1.5 <= words_per_second <= 5.0:
                quality_score += 0.1
            elif words_per_second > 8.0 or words_per_second < 0.5:
                quality_score -= 0.2
            
            # Adjust for text length
            if len(words) >= 2:  # Reduced from 3
                quality_score += 0.1
            
            # Determine quality level with lower thresholds
            if quality_score >= 0.7:
                return "high"
            elif quality_score >= 0.5:
                return "medium"
            elif quality_score >= 0.3:
                return "low"
            else:
                return "poor"
                
        except Exception:
            return "unknown"
    
    def _create_empty_segment(self, diar_segment: Dict, segment_id: int, reason: str) -> Dict:
        """Create empty segment with enhanced metadata"""
        return {
            "id": segment_id,
            "start": float(diar_segment['start']),
            "end": float(diar_segment['end']),
            "duration": float(diar_segment['end'] - diar_segment['start']),
            "text": "",
            "speaker": diar_segment.get('speaker', f'SPEAKER_{segment_id:02d}'),
            "confidence": 0.1,
            "language": "unknown",
            "method": f"empty_segment_{reason}",
            "quality": "empty",
            "failure_reason": reason
        }
    
    def _create_fallback_segment(self, diar_segment: Dict, segment_id: int) -> Dict:
        """Create fallback segment with enhanced error handling"""
        return {
            "id": segment_id,
            "start": float(diar_segment['start']),
            "end": float(diar_segment['end']),
            "duration": float(diar_segment['end'] - diar_segment['start']),
            "text": "",
            "speaker": diar_segment.get('speaker', f'SPEAKER_{segment_id:02d}'),
            "confidence": 0.1,
            "language": "unknown",
            "method": "fallback_segment",
            "quality": "fallback"
        }
    
    def _create_fallback_segments(self, diarization_segments: List[Dict]) -> List[Dict]:
        """Create fallback segments when transcription completely fails"""
        try:
            fallback_segments = []
            
            for i, diar_segment in enumerate(diarization_segments):
                segment = self._create_fallback_segment(diar_segment, i)
                fallback_segments.append(segment)
            
            logger.info(f"✅ Created {len(fallback_segments)} fallback segments")
            return fallback_segments
            
        except Exception as e:
            logger.error(f"Fallback segment creation failed: {e}")
            return []
    
    def get_transcription_stats(self, segments: List[Dict]) -> Dict:
        """Get enhanced transcription statistics"""
        try:
            total_segments = len(segments)
            segments_with_text = sum(1 for seg in segments if seg.get("text", "").strip())
            
            # Quality distribution
            quality_counts = {}
            confidence_scores = []
            failure_reasons = {}
            
            for segment in segments:
                quality = segment.get("quality", "unknown")
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
                confidence_scores.append(segment.get("confidence", 0))
                
                # Track failure reasons
                if not segment.get("text", "").strip():
                    reason = segment.get("failure_reason", "unknown")
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            return {
                "total_segments": total_segments,
                "segments_with_text": segments_with_text,
                "segments_empty": total_segments - segments_with_text,
                "success_rate": (segments_with_text / total_segments * 100) if total_segments > 0 else 0,
                "average_confidence": np.mean(confidence_scores) if confidence_scores else 0,
                "quality_distribution": quality_counts,
                "failure_reasons": failure_reasons,
                "total_duration": sum(seg.get("duration", 0) for seg in segments)
            }
            
        except Exception as e:
            logger.error(f"Stats calculation failed: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.performance_tracker.generate_performance_report()
    
    def get_model_comparison(self) -> Dict:
        """Get detailed comparison of models and strategies"""
        return self.performance_tracker.calculate_model_comparison()
    
    def get_strategy_effectiveness(self) -> Dict:
        """Get effectiveness analysis of processing strategies"""
        return self.performance_tracker.calculate_strategy_effectiveness()
    
    def export_performance_report(self, filepath: str = None) -> str:
        """Export comprehensive performance report to JSON file"""
        try:
            report = self.get_performance_metrics()
            
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"speech_recognition_performance_report_{timestamp}.json"
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Performance report exported to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export performance report: {e}")
            return ""

    def print_performance_summary(self):
        """Print a formatted performance summary to console"""
        try:
            report = self.get_performance_metrics()
            
            print("\n" + "="*80)
            print("🎤 SPEECH RECOGNITION PERFORMANCE SUMMARY")
            print("="*80)
            
            # Overall metrics
            overall = report['overall_metrics']
            print(f"\n📊 OVERALL PERFORMANCE:")
            print(f"   Total Files Processed: {overall['total_files_processed']}")
            print(f"   Successful Files: {overall['successful_files']}")
            print(f"   Total Processing Time: {overall['total_processing_time']:.2f}s")
            print(f"   Total Audio Duration: {overall['total_audio_duration']:.2f}s")
            print(f"   Real-time Factor: {overall['real_time_factor']:.2f}x")
            print(f"   Average Transcription Accuracy: {overall['average_transcription_accuracy']:.2f}")
            print(f"   Total Words Transcribed: {overall['total_words_transcribed']}")
            print(f"   Peak Memory Usage: {overall['peak_memory_usage_mb']:.1f} MB")
            
            # Segment analysis
            segment_analysis = report['segment_analysis']
            print(f"\n🎯 SEGMENT ANALYSIS:")
            print(f"   Total Segments: {segment_analysis['total_segments']}")
            print(f"   Successful Transcriptions: {segment_analysis['successful_transcriptions']}")
            print(f"   Empty Segments: {segment_analysis['empty_segments']}")
            print(f"   Retry Attempts: {segment_analysis['retry_attempts']}")
            print(f"   Successful Retries: {segment_analysis['successful_retries']}")
            print(f"   Average Confidence: {segment_analysis['average_confidence']:.2f}")
            
            # Model comparison
            model_comp = report['model_comparison']
            if model_comp:
                print(f"\n🏆 MODEL PERFORMANCE COMPARISON:")
                sorted_models = sorted(
                    model_comp.items(), 
                    key=lambda x: x[1]['reliability_score'], 
                    reverse=True
                )
                for model_name, metrics in sorted_models:
                    print(f"   {model_name}:")
                    print(f"      Success Rate: {metrics['success_rate']:.1f}%")
                    print(f"      Avg Processing Time: {metrics['average_processing_time']:.3f}s")
                    print(f"      Avg Confidence: {metrics['average_confidence']:.2f}")
                    print(f"      Reliability Score: {metrics['reliability_score']:.2f}")
            
            # Best model
            best_model = report['best_performing_model']
            if best_model['model_name']:
                print(f"\n🥇 BEST PERFORMING MODEL:")
                print(f"   Model: {best_model['model_name']}")
                print(f"   Reliability Score: {best_model['reliability_score']:.2f}")
            
            # Processing efficiency
            efficiency = report['processing_efficiency']
            if efficiency:
                print(f"\n⚡ PROCESSING EFFICIENCY:")
                print(f"   Real-time Factor: {efficiency['real_time_factor']:.2f}x")
                print(f"   Words per Minute: {efficiency['words_per_minute']:.1f}")
                print(f"   Hours Processed per Hour: {efficiency['hours_processed_per_hour']:.2f}")
            
            # Quality analysis
            quality = report['quality_analysis']
            print(f"\n🎧 QUALITY ANALYSIS:")
            print(f"   Segment Success Rate: {quality['segment_success_rate']:.1f}%")
            print(f"   Retry Success Rate: {quality['retry_success_rate']:.1f}%")
            print(f"   Average Segment Confidence: {quality['average_segment_confidence']:.2f}")
            
            # Quality distribution
            quality_dist = quality['quality_distribution']
            print(f"   Quality Distribution:")
            for qual, count in quality_dist.items():
                print(f"      {qual.capitalize()}: {count}")
            
            # Recommendations
            recommendations = report['recommendations']
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            print("\n" + "="*80)
            
        except Exception as e:
            logger.error(f"Failed to print performance summary: {e}")
    
    def cleanup(self):
        """Enhanced cleanup of speech recognition resources"""
        try:
            logger.info("🧹 Cleaning up Enhanced Speech Recognition...")
            
            # Export final performance report
            try:
                self.export_performance_report()
                self.print_performance_summary()
            except Exception as e:
                logger.warning(f"Performance report export failed: {e}")
            
            if self.model:
                gpu_manager.cleanup_model(self.model)
                self.model = None
            
            if self.faster_whisper:
                del self.faster_whisper
                self.faster_whisper = None
            
            gpu_manager.clear_gpu_cache()
            logger.info("✅ Enhanced Speech Recognition cleanup completed")
            
        except Exception as e:
            logger.warning(f"Enhanced Speech Recognition cleanup warning: {e}")
