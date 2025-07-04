"""
ENHANCED Emotion Detector v2.0 - Advanced Multi-Model Emotion Detection
Uses multiple pretrained models for accurate, efficient, and robust emotion detection
"""

import os
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import librosa
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
warnings.filterwarnings("ignore")

# GPU optimization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Disable TensorFlow GPU to avoid conflicts
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print("✅ TensorFlow GPU disabled safely")
except Exception as e:
    print(f"⚠️ TensorFlow setup skipped: {e}")

# Import GPU manager if available
try:
    from utils.gpu_utils import gpu_manager
except ImportError:
    # Fallback for standalone usage
    class DummyGPUManager:
        def get_device(self): return "cuda" if torch.cuda.is_available() else "cpu"
        def clear_gpu_cache(self): 
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        def safe_model_load(self, loader_fn, *args, **kwargs): return loader_fn(*args, **kwargs)
        def cleanup_model(self, model): del model
        def monitor_memory(self): 
            if torch.cuda.is_available():
                return {"gpu_memory_free": torch.cuda.get_device_properties(0).total_memory / 1e9}
            return {"gpu_memory_free": 0}
        def get_optimal_batch_size(self, default_size): return default_size
    
    gpu_manager = DummyGPUManager()

logger = logging.getLogger(__name__)

class EnhancedEmotionDetectorV2:
    """
    Enhanced Emotion Detector v2.0 with multiple pretrained models
    Focuses on accuracy, efficiency, and robustness
    """
    
    def __init__(self, model_size: str = "medium", use_gpu: bool = True):
        """
        Initialize the enhanced emotion detector
        
        Args:
            model_size: Size of models to use ("small", "medium", "large")
            use_gpu: Whether to use GPU acceleration
        """
        self.device = gpu_manager.get_device() if use_gpu else "cpu"
        self.model_size = model_size
        self.confidence_threshold = 0.4
        self.max_audio_duration = 30  # Limit audio processing for efficiency
        
        # Model containers
        self.audio_emotion_models = {}
        self.text_emotion_models = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance tracking
        self.processing_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "total_time": 0,
            "avg_time_per_analysis": 0
        }
        
        # Enhanced emotion mapping for TTS and other applications
        self.emotion_mappings = {
            'tts': {
                'neutral': 'neutral', 'calm': 'neutral',
                'happy': 'cheerful', 'joy': 'cheerful', 'excited': 'excited',
                'sad': 'sad', 'sadness': 'sad',
                'angry': 'angry', 'anger': 'angry',
                'fearful': 'serious', 'fear': 'serious',
                'surprised': 'excited', 'surprise': 'excited',
                'disgust': 'unfriendly',
                'unknown': 'neutral'
            },
            'normalized': {
                'joy': 'happy', 'happiness': 'happy',
                'sadness': 'sad', 'sorrow': 'sad',
                'anger': 'angry', 'rage': 'angry',
                'fear': 'fearful', 'anxiety': 'fearful',
                'surprise': 'surprised', 'amazement': 'surprised',
                'neutral': 'calm', 'peace': 'calm',
                'excitement': 'excited', 'enthusiasm': 'excited'
            }
        }
        
        # Load models based on size preference
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize enhanced emotion detection models"""
        try:
            logger.info(f"🔄 Loading Enhanced Emotion Detector v2.0...")
            logger.info(f"   Model size: {self.model_size}")
            logger.info(f"   Device: {self.device}")
            
            # Clear GPU cache
            gpu_manager.clear_gpu_cache()
            
            # Load audio emotion models based on size
            self._load_audio_emotion_models()
            
            # Load text emotion models
            self._load_text_emotion_models()
            
            logger.info("✅ Enhanced Emotion Detector v2.0 ready!")
            
        except Exception as e:
            logger.error(f"Enhanced emotion detector initialization failed: {e}")
            raise
    
    def _load_audio_emotion_models(self):
        """Load audio emotion detection models"""
        try:
            logger.info("🎵 Loading audio emotion models...")
            
            # Primary audio emotion model (wav2vec2-based)
            try:
                self.audio_emotion_models['primary'] = pipeline(
                    "audio-classification",
                    model="superb/wav2vec2-base-superb-er",
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                logger.info("✅ Primary audio emotion model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Primary audio emotion model failed: {e}")
                self.audio_emotion_models['primary'] = None
            
            # Secondary audio emotion model (enhanced)
            if self.model_size in ["medium", "large"]:
                try:
                    self.audio_emotion_models['secondary'] = pipeline(
                        "audio-classification",
                        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                        device=-1,  # Use CPU to save GPU memory
                        return_all_scores=True
                    )
                    logger.info("✅ Secondary audio emotion model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Secondary audio emotion model failed: {e}")
                    self.audio_emotion_models['secondary'] = None
            
            # Tertiary model for large configuration
            if self.model_size == "large":
                try:
                    self.audio_emotion_models['tertiary'] = pipeline(
                        "audio-classification",
                        model="facebook/wav2vec2-large-xlsr-53",
                        device=-1,
                        return_all_scores=True
                    )
                    logger.info("✅ Tertiary audio emotion model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Tertiary audio emotion model failed: {e}")
                    self.audio_emotion_models['tertiary'] = None
            
        except Exception as e:
            logger.error(f"Audio emotion model loading failed: {e}")
    
    def _load_text_emotion_models(self):
        """Load text emotion detection models"""
        try:
            logger.info("📝 Loading text emotion models...")
            
            # Primary text emotion model
            try:
                self.text_emotion_models['primary'] = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=-1,
                    return_all_scores=True
                )
                logger.info("✅ Primary text emotion model loaded")
            except Exception as e:
                logger.warning(f"⚠️ Primary text emotion model failed: {e}")
                self.text_emotion_models['primary'] = None
            
            # Secondary text emotion model
            if self.model_size in ["medium", "large"]:
                try:
                    self.text_emotion_models['secondary'] = pipeline(
                        "text-classification",
                        model="cardiffnlp/twitter-roberta-base-emotion",
                        device=-1,
                        return_all_scores=True
                    )
                    logger.info("✅ Secondary text emotion model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Secondary text emotion model failed: {e}")
                    self.text_emotion_models['secondary'] = None
            
            # Advanced model for large configuration
            if self.model_size == "large":
                try:
                    self.text_emotion_models['advanced'] = pipeline(
                        "text-classification",
                        model="SamLowe/roberta-base-go_emotions",
                        device=-1,
                        return_all_scores=True
                    )
                    logger.info("✅ Advanced text emotion model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Advanced text emotion model failed: {e}")
                    self.text_emotion_models['advanced'] = None
            
        except Exception as e:
            logger.error(f"Text emotion model loading failed: {e}")
    
    def detect_emotion_from_audio(self, audio_path: str) -> Dict:
        """
        Enhanced audio emotion detection using multiple models
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with emotion detection results
        """
        try:
            start_time = time.time()
            
            logger.debug(f"🎵 Enhanced audio emotion analysis: {audio_path}")
            
            if not os.path.exists(audio_path):
                logger.warning(f"⚠️ Audio file not found: {audio_path}")
                return self._create_unknown_result("file_not_found")
            
            # Load and preprocess audio
            audio_data = self._load_and_preprocess_audio(audio_path)
            if not audio_data:
                return self._create_unknown_result("audio_load_failed")
            
            y, sr = audio_data
            
            # Analyze with multiple models
            results = []
            
            # Model-based analysis
            for model_name, model in self.audio_emotion_models.items():
                if model:
                    try:
                        model_result = self._analyze_audio_with_model(y, sr, model, model_name)
                        if model_result:
                            results.append(model_result)
                    except Exception as e:
                        logger.debug(f"Audio model {model_name} failed: {e}")
            
            # Enhanced acoustic analysis
            try:
                acoustic_result = self._enhanced_acoustic_emotion_analysis(y, sr)
                results.append(acoustic_result)
            except Exception as e:
                logger.debug(f"Acoustic analysis failed: {e}")
            
            # Fuse results
            if not results:
                return self._create_unknown_result("all_methods_failed")
            
            final_result = self._fuse_audio_emotion_results(results)
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, True)
            
            logger.info(f"🎵 Audio emotion: {final_result['emotion']} "
                       f"(confidence: {final_result['confidence']:.2f}, "
                       f"time: {processing_time:.3f}s)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Audio emotion detection failed: {e}")
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            self._update_processing_stats(processing_time, False)
            return self._create_error_result(str(e))
    
    def detect_emotion_from_text(self, text: str) -> Dict:
        """
        Enhanced text emotion detection using multiple models
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion detection results
        """
        try:
            start_time = time.time()
            
            logger.debug(f"📝 Enhanced text emotion analysis: {text[:50]}...")
            
            if not text or not text.strip():
                return self._create_unknown_result("no_text")
            
            # Preprocess text
            processed_text = self._preprocess_text_for_emotion(text)
            
            results = []
            
            # Model-based analysis
            for model_name, model in self.text_emotion_models.items():
                if model:
                    try:
                        model_result = self._analyze_text_with_model(processed_text, model, model_name)
                        if model_result:
                            results.append(model_result)
                    except Exception as e:
                        logger.debug(f"Text model {model_name} failed: {e}")
            
            # Enhanced keyword analysis
            try:
                keyword_result = self._enhanced_keyword_emotion_analysis(processed_text)
                results.append(keyword_result)
            except Exception as e:
                logger.debug(f"Keyword analysis failed: {e}")
            
            # Linguistic pattern analysis
            try:
                linguistic_result = self._linguistic_emotion_analysis(processed_text)
                results.append(linguistic_result)
            except Exception as e:
                logger.debug(f"Linguistic analysis failed: {e}")
            
            # Fuse results
            if not results:
                return self._create_unknown_result("all_methods_failed")
            
            final_result = self._fuse_text_emotion_results(results)
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, True)
            
            logger.info(f"📝 Text emotion: {final_result['emotion']} "
                       f"(confidence: {final_result['confidence']:.2f}, "
                       f"time: {processing_time:.3f}s)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Text emotion detection failed: {e}")
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            self._update_processing_stats(processing_time, False)
            return self._create_error_result(str(e))
    
    def detect_speakers_emotion(self, speaker_segments: Dict) -> Dict:
        """
        Enhanced speaker emotion detection for multiple speakers
        
        Args:
            speaker_segments: Dictionary mapping speaker IDs to their segments
            
        Returns:
            Dictionary mapping speaker IDs to emotion detection results
        """
        try:
            logger.info(f"🎭 Enhanced speaker emotion detection for {len(speaker_segments)} speakers...")
            
            results = {}
            
            for speaker_id, segments in speaker_segments.items():
                try:
                    logger.debug(f"🔍 Processing speaker {speaker_id} with {len(segments)} segments...")
                    
                    if not segments:
                        results[speaker_id] = self._create_unknown_result("no_segments")
                        continue
                    
                    # Select best segments for analysis
                    best_segments = self._select_best_segments_for_emotion(segments)
                    
                    if not best_segments:
                        results[speaker_id] = self._create_unknown_result("no_suitable_segments")
                        continue
                    
                    # Analyze segments
                    speaker_result = self._analyze_speaker_emotion(best_segments, speaker_id)
                    results[speaker_id] = speaker_result
                    
                    logger.info(f"✅ Speaker {speaker_id}: {speaker_result['emotion']} "
                              f"(confidence: {speaker_result['confidence']:.2f})")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing speaker {speaker_id}: {e}")
                    results[speaker_id] = self._create_error_result(str(e))
            
            # Calculate summary statistics
            total_speakers = len(results)
            successful_detections = sum(1 for r in results.values() 
                                      if r.get('emotion', 'unknown') != 'unknown')
            
            logger.info(f"🎭 Speaker emotion detection summary:")
            logger.info(f"   Total speakers: {total_speakers}")
            logger.info(f"   Successful detections: {successful_detections}")
            logger.info(f"   Success rate: {(successful_detections/total_speakers*100):.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Speaker emotion detection failed: {e}")
            return {}
    
    def detect_emotion_from_transcription_segments(self, segments: List[Dict], 
                                                 audio_path: str = None) -> Dict:
        """
        Detect emotions from transcription segments with optional audio analysis
        
        Args:
            segments: List of transcription segments
            audio_path: Optional path to audio file for enhanced analysis
            
        Returns:
            Dictionary with emotion detection results
        """
        try:
            logger.info(f"🎬 Enhanced emotion detection from {len(segments)} transcription segments...")
            
            # Group segments by speaker
            speaker_segments = {}
            for segment in segments:
                speaker_id = segment.get('speaker', 'UNKNOWN')
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = []
                speaker_segments[speaker_id].append(segment)
            
            # Detect emotions for each speaker
            speaker_emotions = self.detect_speakers_emotion(speaker_segments)
            
            # Enhance with audio analysis if available
            if audio_path and os.path.exists(audio_path):
                try:
                    audio_emotions = self._analyze_audio_segments_emotion(segments, audio_path)
                    speaker_emotions = self._combine_audio_text_emotions(speaker_emotions, audio_emotions)
                except Exception as e:
                    logger.warning(f"Audio enhancement failed: {e}")
            
            # Calculate statistics
            total_speakers = len(speaker_emotions)
            detected_speakers = sum(1 for e in speaker_emotions.values() 
                                  if e.get('emotion', 'unknown') != 'unknown')
            
            return {
                'success': True,
                'speaker_emotions': speaker_emotions,
                'stats': {
                    'total_speakers': total_speakers,
                    'detected_speakers': detected_speakers,
                    'detection_rate': (detected_speakers / total_speakers * 100) if total_speakers > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Transcription segment emotion detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'speaker_emotions': {}
            }
    
    def _load_and_preprocess_audio(self, audio_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Load and preprocess audio for emotion analysis"""
        try:
            # Load audio with optimal parameters
            y, sr = librosa.load(
                audio_path, 
                sr=16000, 
                duration=self.max_audio_duration,
                mono=True
            )
            
            if len(y) < 1600:  # Less than 0.1 seconds
                logger.warning(f"Audio too short: {len(y)} samples")
                return None
            
            # Normalize audio
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # Remove silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) > 1600:
                y = y_trimmed
            
            return y, sr
            
        except Exception as e:
            logger.warning(f"Audio loading failed: {e}")
            return None
    
    def _analyze_audio_with_model(self, y: np.ndarray, sr: int, 
                                model, model_name: str) -> Optional[Dict]:
        """Analyze audio with a specific model"""
        try:
            # Clear GPU cache before analysis
            gpu_manager.clear_gpu_cache()
            
            # Get model predictions
            results = model(y, sampling_rate=sr)
            
            if not results:
                return None
            
            # Process results
            processed_results = []
            for result in results[:3]:  # Top 3 predictions
                emotion = self._normalize_emotion_name(result.get('label', 'unknown'))
                confidence = result.get('score', 0.0)
                
                if confidence > 0.2:  # Only consider confident predictions
                    processed_results.append({
                        'emotion': emotion,
                        'confidence': confidence,
                        'raw_label': result.get('label', '')
                    })
            
            if not processed_results:
                return None
            
            # Return best result
            best_result = max(processed_results, key=lambda x: x['confidence'])
            
            return {
                'method': f'audio_model_{model_name}',
                'emotion': best_result['emotion'],
                'confidence': best_result['confidence'],
                'all_predictions': processed_results
            }
            
        except Exception as e:
            logger.debug(f"Audio model {model_name} analysis failed: {e}")
            return None
    
    def _analyze_text_with_model(self, text: str, model, model_name: str) -> Optional[Dict]:
        """Analyze text with a specific model"""
        try:
            # Get model predictions
            results = model(text)
            
            if not results:
                return None
            
            # Process results
            processed_results = []
            for result in results[:3]:  # Top 3 predictions
                emotion = self._normalize_emotion_name(result.get('label', 'unknown'))
                confidence = result.get('score', 0.0)
                
                if confidence > 0.2:
                    processed_results.append({
                        'emotion': emotion,
                        'confidence': confidence,
                        'raw_label': result.get('label', '')
                    })
            
            if not processed_results:
                return None
            
            # Return best result
            best_result = max(processed_results, key=lambda x: x['confidence'])
            
            return {
                'method': f'text_model_{model_name}',
                'emotion': best_result['emotion'],
                'confidence': best_result['confidence'],
                'all_predictions': processed_results
            }
            
        except Exception as e:
            logger.debug(f"Text model {model_name} analysis failed: {e}")
            return None
    
    def _enhanced_acoustic_emotion_analysis(self, y: np.ndarray, sr: int) -> Dict:
        """Enhanced acoustic emotion analysis with comprehensive features"""
        try:
            # Extract comprehensive features
            features = self._extract_comprehensive_emotion_features(y, sr)
            
            # Emotion classification based on acoustic patterns
            emotions = {
                "angry": 0.0, "happy": 0.0, "sad": 0.0, "fearful": 0.0,
                "surprised": 0.0, "calm": 0.0, "excited": 0.0
            }
            
            # Energy-based classification
            energy = features.get('energy_mean', 0)
            energy_std = features.get('energy_std', 0)
            
            if energy > 0.05:  # Very high energy
                emotions["angry"] += 0.6
                emotions["excited"] += 0.5
            elif energy > 0.03:  # High energy
                emotions["excited"] += 0.5
                emotions["happy"] += 0.4
            elif energy > 0.015:  # Medium energy
                emotions["happy"] += 0.4
                emotions["surprised"] += 0.3
            elif energy < 0.005:  # Very low energy
                emotions["sad"] += 0.6
                emotions["calm"] += 0.4
            else:
                emotions["calm"] += 0.4
                emotions["sad"] += 0.2
            
            # Pitch variation analysis
            f0_std = features.get('f0_std', 0)
            f0_range = features.get('f0_range', 0)
            
            if f0_std > 40:  # Very high variation
                emotions["excited"] += 0.5
                emotions["surprised"] += 0.4
            elif f0_std > 25:  # High variation
                emotions["happy"] += 0.4
                emotions["excited"] += 0.3
            elif f0_std < 5:  # Very low variation
                emotions["sad"] += 0.5
                emotions["calm"] += 0.4
            
            # Spectral characteristics
            spectral_centroid = features.get('spectral_centroid', 0)
            spectral_rolloff = features.get('spectral_rolloff', 0)
            
            if spectral_centroid > 3000:  # Very bright
                emotions["angry"] += 0.4
                emotions["fearful"] += 0.3
            elif spectral_centroid > 2200:  # Bright
                emotions["happy"] += 0.3
                emotions["excited"] += 0.2
            elif spectral_centroid < 1200:  # Dark
                emotions["sad"] += 0.4
                emotions["calm"] += 0.2
            
            # Tempo analysis
            tempo = features.get('tempo', 120)
            if tempo > 140:  # Fast
                emotions["excited"] += 0.3
                emotions["happy"] += 0.2
            elif tempo < 80:  # Slow
                emotions["sad"] += 0.3
                emotions["calm"] += 0.2
            
            # MFCC-based timbre analysis
            mfcc_mean = features.get('mfcc_mean', 0)
            if mfcc_mean > 5:  # Harsh timbre
                emotions["angry"] += 0.2
            elif mfcc_mean < -5:  # Soft timbre
                emotions["calm"] += 0.2
                emotions["sad"] += 0.1
            
            # Find best emotion
            best_emotion = max(emotions, key=emotions.get)
            confidence = min(0.85, emotions[best_emotion])
            
            if confidence < 0.25:
                best_emotion = "calm"
                confidence = 0.4
            
            return {
                'method': 'enhanced_acoustic_analysis',
                'emotion': best_emotion,
                'confidence': confidence,
                'features': features,
                'emotion_scores': emotions
            }
            
        except Exception as e:
            logger.debug(f"Enhanced acoustic analysis failed: {e}")
            return {
                'method': 'acoustic_fallback',
                'emotion': 'calm',
                'confidence': 0.3
            }
    
    def _extract_comprehensive_emotion_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive features for emotion analysis"""
        features = {}
        
        try:
            # Energy features
            rms_energy = librosa.feature.rms(y=y)[0]
            features['energy_mean'] = float(np.mean(rms_energy))
            features['energy_std'] = float(np.std(rms_energy))
            features['energy_max'] = float(np.max(rms_energy))
            features['energy_range'] = float(np.max(rms_energy) - np.min(rms_energy))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            features['spectral_contrast'] = float(np.mean(spectral_contrast))
            
            # Pitch features
            f0 = librosa.yin(y, fmin=50, fmax=400)
            f0_values = f0[~np.isnan(f0) & (f0 > 0)]
            
            if len(f0_values) > 0:
                features['f0_mean'] = float(np.mean(f0_values))
                features['f0_std'] = float(np.std(f0_values))
                features['f0_range'] = float(np.max(f0_values) - np.min(f0_values))
                features['f0_median'] = float(np.median(f0_values))
            else:
                features['f0_mean'] = features['f0_std'] = 0.0
                features['f0_range'] = features['f0_median'] = 0.0
            
            # Rhythm and tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = float(np.mean(mfccs))
            features['mfcc_std'] = float(np.std(mfccs))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
        
        return features
    
    def _enhanced_keyword_emotion_analysis(self, text: str) -> Dict:
        """Enhanced keyword-based emotion analysis"""
        try:
            # Comprehensive emotion keywords
            emotion_keywords = {
                'happy': ['happy', 'joy', 'joyful', 'excited', 'cheerful', 'glad', 'wonderful', 
                         'great', 'amazing', 'fantastic', 'awesome', 'brilliant', 'excellent'],
                'sad': ['sad', 'depressed', 'unhappy', 'disappointed', 'heartbroken', 'miserable',
                       'sorrowful', 'melancholy', 'gloomy', 'downhearted', 'dejected'],
                'angry': ['angry', 'mad', 'furious', 'irritated', 'frustrated', 'hate', 'rage',
                         'annoyed', 'outraged', 'livid', 'enraged', 'irate'],
                'fearful': ['afraid', 'scared', 'worried', 'nervous', 'anxious', 'terrified',
                           'frightened', 'panicked', 'alarmed', 'concerned'],
                'surprised': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable',
                             'astonished', 'stunned', 'astounded'],
                'calm': ['calm', 'peaceful', 'relaxed', 'quiet', 'gentle', 'serene', 'tranquil',
                        'composed', 'placid'],
                'excited': ['excited', 'thrilled', 'enthusiastic', 'pumped', 'energetic', 'eager',
                           'exhilarated', 'elated']
            }
            
            text_lower = text.lower()
            emotion_scores = {}
            
            # Count keyword occurrences
            for emotion, keywords in emotion_keywords.items():
                score = 0
                for keyword in keywords:
                    count = text_lower.count(keyword)
                    if count > 0:
                        # Weight longer keywords more heavily
                        weight = len(keyword) / 5.0
                        score += count * weight
                
                if score > 0:
                    emotion_scores[emotion] = score
            
            # Punctuation analysis
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
            
            # Adjust scores based on punctuation
            if exclamation_count > 0:
                emotion_scores['excited'] = emotion_scores.get('excited', 0) + exclamation_count * 0.5
                emotion_scores['happy'] = emotion_scores.get('happy', 0) + exclamation_count * 0.3
            
            if question_count > 0:
                emotion_scores['surprised'] = emotion_scores.get('surprised', 0) + question_count * 0.3
            
            if caps_ratio > 0.3:  # Lots of caps
                emotion_scores['angry'] = emotion_scores.get('angry', 0) + caps_ratio * 2
                emotion_scores['excited'] = emotion_scores.get('excited', 0) + caps_ratio * 1
            
            # Determine final emotion
            if emotion_scores:
                emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = min(0.8, emotion_scores[emotion] * 0.15 + 0.3)
            else:
                emotion = "calm"
                confidence = 0.4
            
            return {
                'method': 'enhanced_keyword_analysis',
                'emotion': emotion,
                'confidence': confidence,
                'keyword_scores': emotion_scores
            }
            
        except Exception as e:
            logger.debug(f"Enhanced keyword analysis failed: {e}")
            return {
                'method': 'keyword_fallback',
                'emotion': 'calm',
                'confidence': 0.3
            }
    
    def _linguistic_emotion_analysis(self, text: str) -> Dict:
        """Linguistic pattern analysis for emotion detection"""
        try:
            # Analyze sentence structure and patterns
            sentences = text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # Word complexity analysis
            complex_words = [word for word in text.split() if len(word) > 6]
            complexity_ratio = len(complex_words) / max(1, len(text.split()))
            
            # Emotional indicators
            emotion_scores = {
                'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fearful': 0.0,
                'surprised': 0.0, 'calm': 0.0, 'excited': 0.0
            }
            
            # Sentence length patterns
            if avg_sentence_length > 15:  # Long sentences
                emotion_scores['calm'] += 0.3
            elif avg_sentence_length < 5:  # Short sentences
                emotion_scores['excited'] += 0.3
                emotion_scores['angry'] += 0.2
            
            # Complexity patterns
            if complexity_ratio > 0.3:  # High complexity
                emotion_scores['calm'] += 0.2
            elif complexity_ratio < 0.1:  # Low complexity
                emotion_scores['excited'] += 0.2
            
            # Repetition patterns
            words = text.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.7:  # High repetition
                    emotion_scores['angry'] += 0.2
                    emotion_scores['excited'] += 0.1
            
            # Find best emotion
            if any(score > 0 for score in emotion_scores.values()):
                emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = min(0.6, emotion_scores[emotion] + 0.3)
            else:
                emotion = "calm"
                confidence = 0.4
            
            return {
                'method': 'linguistic_analysis',
                'emotion': emotion,
                'confidence': confidence,
                'linguistic_features': {
                    'avg_sentence_length': avg_sentence_length,
                    'complexity_ratio': complexity_ratio,
                    'unique_word_ratio': unique_ratio if 'unique_ratio' in locals() else 0
                }
            }
            
        except Exception as e:
            logger.debug(f"Linguistic analysis failed: {e}")
            return {
                'method': 'linguistic_fallback',
                'emotion': 'calm',
                'confidence': 0.3
            }
    
    def _fuse_audio_emotion_results(self, results: List[Dict]) -> Dict:
        """Fuse audio emotion results with intelligent weighting"""
        if not results:
            return self._create_unknown_result("no_audio_results")
        
        # Enhanced method weights
        method_weights = {
            'audio_model_primary': 0.4,
            'audio_model_secondary': 0.3,
            'audio_model_tertiary': 0.2,
            'enhanced_acoustic_analysis': 0.1
        }
        
        return self._fuse_emotion_results(results, method_weights, "audio_multimodel_fusion")
    
    def _fuse_text_emotion_results(self, results: List[Dict]) -> Dict:
        """Fuse text emotion results with intelligent weighting"""
        if not results:
            return self._create_unknown_result("no_text_results")
        
        # Enhanced method weights
        method_weights = {
            'text_model_primary': 0.35,
            'text_model_secondary': 0.25,
            'text_model_advanced': 0.2,
            'enhanced_keyword_analysis': 0.15,
            'linguistic_analysis': 0.05
        }
        
        return self._fuse_emotion_results(results, method_weights, "text_multimodel_fusion")
    
    def _fuse_emotion_results(self, results: List[Dict], method_weights: Dict, 
                            fusion_method: str) -> Dict:
        """Generic emotion result fusion with enhanced logic"""
        try:
            emotion_scores = {}
            confidence_weights = {}
            total_weight = 0.0
            
            for result in results:
                method = result.get('method', 'unknown')
                emotion = result.get('emotion', 'unknown')
                confidence = result.get('confidence', 0.0)
                weight = method_weights.get(method, 0.05)
                
                # Adjust weight based on confidence
                adjusted_weight = weight * (0.5 + confidence * 0.5)
                
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.0
                    confidence_weights[emotion] = 0.0
                
                emotion_scores[emotion] += confidence * adjusted_weight
                confidence_weights[emotion] += adjusted_weight
                total_weight += adjusted_weight
            
            # Normalize scores
            if total_weight > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= total_weight
            
            # Find best emotion
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            best_confidence = emotion_scores[best_emotion]
            
            # Apply consensus bonus
            emotion_counts = {}
            for result in results:
                emotion = result.get('emotion', 'unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if emotion_counts.get(best_emotion, 0) > 1:
                # Multiple methods agree - boost confidence
                best_confidence = min(0.95, best_confidence + 0.1)
            
            # Apply minimum confidence threshold
            if best_confidence < 0.25:
                best_emotion = "calm"
                best_confidence = 0.4
            
            # Get TTS emotion
            tts_emotion = self._map_to_tts_emotion(best_emotion)
            
            return {
                'emotion': best_emotion,
                'confidence': float(best_confidence),
                'tts_emotion': tts_emotion,
                'method': fusion_method,
                'individual_results': results,
                'emotion_scores': emotion_scores,
                'methods_used': len(results),
                'consensus_count': emotion_counts.get(best_emotion, 0)
            }
            
        except Exception as e:
            logger.debug(f"Emotion fusion failed: {e}")
            return self._create_unknown_result("fusion_failed")
    
    def _select_best_segments_for_emotion(self, segments: List[Dict], 
                                        max_segments: int = 3) -> List[Dict]:
        """Select the best segments for emotion analysis"""
        try:
            if not segments:
                return []
            
            # Score segments based on multiple criteria
            scored_segments = []
            
            for segment in segments:
                score = 0.0
                
                # Duration score (prefer 1-5 second segments)
                duration = segment.get('duration', 0)
                if 1.0 <= duration <= 5.0:
                    score += 3.0
                elif 0.5 <= duration <= 8.0:
                    score += 2.0
                elif duration > 0.3:
                    score += 1.0
                
                # Text content score
                text = segment.get('text', '')
                if text and len(text.strip()) > 10:
                    score += 2.0
                elif text and len(text.strip()) > 3:
                    score += 1.0
                
                # Audio file availability
                audio_file = segment.get('audio_file', '')
                if audio_file and os.path.exists(audio_file):
                    score += 2.0
                
                # Emotional content indicators
                if text:
                    emotional_indicators = ['!', '?', 'very', 'really', 'so', 'extremely']
                    if any(indicator in text.lower() for indicator in emotional_indicators):
                        score += 1.0
                
                # Confidence score
                confidence = segment.get('confidence', 0)
                score += confidence * 2.0
                
                scored_segments.append((score, segment))
            
            # Sort by score and return top segments
            scored_segments.sort(key=lambda x: x[0], reverse=True)
            best_segments = [seg for score, seg in scored_segments[:max_segments]]
            
            logger.debug(f"Selected {len(best_segments)} best segments from {len(segments)} total")
            return best_segments
            
        except Exception as e:
            logger.debug(f"Segment selection failed: {e}")
            return segments[:max_segments] if segments else []
    
    def _analyze_speaker_emotion(self, segments: List[Dict], speaker_id: str) -> Dict:
        """Analyze emotion for a specific speaker"""
        try:
            audio_results = []
            text_results = []
            
            # Analyze each segment
            for segment in segments:
                # Audio analysis
                audio_file = segment.get('audio_file')
                if audio_file and os.path.exists(audio_file):
                    try:
                        audio_result = self.detect_emotion_from_audio(audio_file)
                        if audio_result.get('confidence', 0) > 0.3:
                            audio_results.append(audio_result)
                    except Exception as e:
                        logger.debug(f"Audio analysis failed for segment: {e}")
                
                # Text analysis
                text = segment.get('text', '')
                if text and text.strip():
                    try:
                        text_result = self.detect_emotion_from_text(text)
                        if text_result.get('confidence', 0) > 0.3:
                            text_results.append(text_result)
                    except Exception as e:
                        logger.debug(f"Text analysis failed for segment: {e}")
            
            # Combine results
            all_results = audio_results + text_results
            
            if not all_results:
                return self._create_unknown_result("no_analysis_results")
            
            # Fuse all results
            method_weights = {
                'audio_multimodel_fusion': 0.6,
                'text_multimodel_fusion': 0.4
            }
            
            final_result = self._fuse_emotion_results(all_results, method_weights, "speaker_multimodal_fusion")
            final_result['speaker_id'] = speaker_id
            final_result['segments_analyzed'] = len(segments)
            final_result['audio_segments'] = len(audio_results)
            final_result['text_segments'] = len(text_results)
            
            return final_result
            
        except Exception as e:
            logger.debug(f"Speaker emotion analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _preprocess_text_for_emotion(self, text: str) -> str:
        """Preprocess text for better emotion analysis"""
        try:
            # Remove extra whitespace
            processed = ' '.join(text.split())
            
            # Preserve important punctuation
            processed = processed.replace('!', ' ! ')
            processed = processed.replace('?', ' ? ')
            
            # Remove excessive punctuation
            import re
            processed = re.sub(r'[^\w\s!?.,;:-]', ' ', processed)
            
            # Normalize whitespace again
            processed = ' '.join(processed.split())
            
            return processed
            
        except Exception:
            return text
    
    def _normalize_emotion_name(self, emotion: str) -> str:
        """Normalize emotion names from different models"""
        emotion = emotion.lower().strip()
        
        # Use the normalized mapping
        return self.emotion_mappings['normalized'].get(emotion, emotion)
    
    def _map_to_tts_emotion(self, emotion: str) -> str:
        """Map detected emotion to TTS-compatible emotion style"""
        return self.emotion_mappings['tts'].get(emotion.lower(), 'neutral')
    
    def _create_unknown_result(self, reason: str) -> Dict:
        """Create a standardized unknown result"""
        return {
            'emotion': 'unknown',
            'confidence': 0.0,
            'tts_emotion': 'neutral',
            'method': f'unknown_{reason}',
            'reason': reason
        }
    
    def _create_error_result(self, error: str) -> Dict:
        """Create a standardized error result"""
        return {
            'emotion': 'calm',
            'confidence': 0.3,
            'tts_emotion': 'neutral',
            'method': 'error_fallback',
            'error': error
        }
    
    def _update_processing_stats(self, processing_time: float, success: bool):
        """Update processing statistics"""
        try:
            self.processing_stats["total_analyses"] += 1
            if success:
                self.processing_stats["successful_analyses"] += 1
            
            self.processing_stats["total_time"] += processing_time
            self.processing_stats["avg_time_per_analysis"] = (
                self.processing_stats["total_time"] / self.processing_stats["total_analyses"]
                if self.processing_stats["total_analyses"] > 0 else 0
            )
        except Exception as e:
            logger.debug(f"Stats update failed: {e}")
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return self.processing_stats.copy()
    
    def cleanup(self):
        """Clean up emotion detection resources"""
        try:
            logger.info("🧹 Cleaning up Enhanced Emotion Detector v2.0...")
            
            # Clean up audio models
            for model_name, model in self.audio_emotion_models.items():
                if model:
                    del model
            self.audio_emotion_models.clear()
            
            # Clean up text models
            for model_name, model in self.text_emotion_models.items():
                if model:
                    del model
            self.text_emotion_models.clear()
            
            # Shutdown executor
            self.executor.shutdown(wait=False)
            
            # Clear GPU cache
            gpu_manager.clear_gpu_cache()
            
            logger.info("✅ Enhanced Emotion Detector v2.0 cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# Factory function for easy import
def create_enhanced_emotion_detector(model_size: str = "medium", use_gpu: bool = True) -> EnhancedEmotionDetectorV2:
    """Factory function to create an enhanced emotion detector"""
    return EnhancedEmotionDetectorV2(model_size=model_size, use_gpu=use_gpu)

# Test the enhanced emotion detector
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Testing Enhanced Emotion Detector v2.0")
    print("=" * 70)
    
    # Create detector
    detector = create_enhanced_emotion_detector(model_size="medium")
    
    # Test text emotion detection
    print("\n📝 Testing text emotion detection...")
    test_texts = [
        "I am so happy and excited about this!",
        "This is really sad and disappointing.",
        "I'm very angry about what happened!",
        "I feel calm and peaceful today.",
        "Wow, that's absolutely amazing and surprising!"
    ]
    
    for text in test_texts:
        result = detector.detect_emotion_from_text(text)
        print(f"  Text: '{text}'")
        print(f"  Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
        print(f"  TTS: {result['tts_emotion']}")
        print()
    
    # Test with mock speaker segments
    print("🎭 Testing speaker emotion detection...")
    mock_segments = {
        "SPEAKER_00": [
            {
                "start": 0.0,
                "end": 3.0,
                "duration": 3.0,
                "text": "Hello everyone! I'm so excited to be here today!",
                "confidence": 0.9
            },
            {
                "start": 3.0,
                "end": 6.0,
                "duration": 3.0,
                "text": "This is going to be an amazing presentation.",
                "confidence": 0.8
            }
        ],
        "SPEAKER_01": [
            {
                "start": 6.0,
                "end": 9.0,
                "duration": 3.0,
                "text": "I'm feeling quite nervous about this.",
                "confidence": 0.7
            }
        ]
    }
    
    speaker_results = detector.detect_speakers_emotion(mock_segments)
    
    for speaker_id, result in speaker_results.items():
        print(f"  {speaker_id}: {result['emotion']} (confidence: {result['confidence']:.2f})")
    
    # Get statistics
    stats = detector.get_processing_stats()
    print(f"\n📊 Processing statistics:")
    print(f"  Total analyses: {stats['total_analyses']}")
    print(f"  Successful analyses: {stats['successful_analyses']}")
    print(f"  Average time per analysis: {stats['avg_time_per_analysis']:.3f}s")
    
    # Cleanup
    detector.cleanup()
    print("\n" + "=" * 70)
