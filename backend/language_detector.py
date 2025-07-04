"""
Advanced language detection for video content
"""

import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import librosa
import numpy as np
from transformers import pipeline
import torch

from utils.gpu_utils import gpu_manager

logger = logging.getLogger(__name__)

class AdvancedLanguageDetector:
    def __init__(self):
        self.device = gpu_manager.get_device()
        self.text_classifier = None
        self.audio_classifier = None
        
        self._load_models()
    
    def _load_models(self):
        """Load language detection models"""
        try:
            logger.info("Loading language detection models...")
            
            # Text-based language detection
            try:
                self.text_classifier = pipeline(
                    "text-classification",
                    model="papluca/xlm-roberta-base-language-detection",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("✅ Text language classifier loaded")
            except Exception as e:
                logger.warning(f"Text classifier loading failed: {e}")
            
            # Audio-based language detection (using Whisper's language detection)
            logger.info("✅ Audio language detection ready (using Whisper)")
            
        except Exception as e:
            logger.error(f"Language detection model loading failed: {e}")
    
    def detect_language_from_video(self, video_path: str, sample_duration: int = 60) -> Dict:
        """Comprehensive language detection from video"""
        try:
            logger.info(f"🔍 Detecting language from video: {video_path}")
            
            # Extract audio sample for analysis
            audio_sample = self._extract_audio_sample(video_path, sample_duration)
            
            # Method 1: Audio-based detection using Whisper
            audio_detection = self._detect_from_audio(audio_sample)
            
            # Method 2: Text-based detection from transcription
            text_detection = None
            if audio_detection.get("transcription"):
                text_detection = self._detect_from_text(audio_detection["transcription"])
            
            # Method 3: Statistical analysis of audio features
            acoustic_detection = self._detect_from_acoustic_features(audio_sample)
            
            # Combine results
            final_result = self._combine_detection_results(
                audio_detection, text_detection, acoustic_detection
            )
            
            # Cleanup temp file
            if audio_sample.exists():
                audio_sample.unlink()
            
            logger.info(f"🎯 Final detection: {final_result['language']} (confidence: {final_result['confidence']:.2f})")
            return final_result
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                "language": "en",
                "confidence": 0.5,
                "method": "fallback",
                "error": str(e)
            }
    
    def _extract_audio_sample(self, video_path: str, duration: int) -> Path:
        """Extract audio sample from video for analysis"""
        try:
            temp_audio = Path(tempfile.mktemp(suffix=".wav"))
            
            # Extract audio sample from middle of video
            cmd = [
                "ffmpeg", "-y",
                "-ss", "30",  # Start from 30 seconds
                "-i", str(video_path),
                "-t", str(duration),  # Duration
                "-acodec", "pcm_s16le",
                "-ac", "1",  # Mono
                "-ar", "16000",  # 16kHz sample rate
                "-loglevel", "error",
                str(temp_audio)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not temp_audio.exists() or temp_audio.stat().st_size == 0:
                raise Exception("Audio extraction failed")
            
            return temp_audio
            
        except Exception as e:
            logger.error(f"Audio sample extraction failed: {e}")
            raise
    
    def _detect_from_audio(self, audio_path: Path) -> Dict:
        """Detect language using audio-based methods"""
        try:
            import whisper
            
            # Load a small Whisper model for quick detection
            model = whisper.load_model("base", device=self.device)
            
            # Transcribe with language detection
            result = model.transcribe(
                str(audio_path),
                language=None,  # Auto-detect
                word_timestamps=False,
                verbose=False
            )
            
            detected_lang = result.get("language", "en")
            transcription = result.get("text", "").strip()
            
            # Calculate confidence based on transcription quality
            confidence = self._calculate_audio_confidence(result)
            
            return {
                "language": detected_lang,
                "confidence": confidence,
                "transcription": transcription,
                "method": "whisper_audio"
            }
            
        except Exception as e:
            logger.error(f"Audio-based detection failed: {e}")
            return {
                "language": "en",
                "confidence": 0.3,
                "method": "audio_fallback",
                "error": str(e)
            }
    
    def _detect_from_text(self, text: str) -> Optional[Dict]:
        """Detect language from transcribed text"""
        try:
            if not self.text_classifier or not text.strip():
                return None
            
            # Clean text for better detection
            cleaned_text = self._clean_text_for_detection(text)
            
            if len(cleaned_text) < 10:  # Too short for reliable detection
                return None
            
            # Detect language
            result = self.text_classifier(cleaned_text)
            
            if result and len(result) > 0:
                detected_lang = result[0]["label"].lower()
                confidence = result[0]["score"]
                
                # Map to our supported languages
                lang_mapping = {
                    "hindi": "hi", "hi": "hi",
                    "english": "en", "en": "en", "eng": "en",
                    "urdu": "ur", "ur": "ur",
                    "arabic": "ar", "ar": "ar", "ara": "ar"
                }
                
                mapped_lang = lang_mapping.get(detected_lang, detected_lang)
                
                return {
                    "language": mapped_lang,
                    "confidence": confidence,
                    "method": "text_classification",
                    "original_label": detected_lang
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Text-based detection failed: {e}")
            return None
    
    def _detect_from_acoustic_features(self, audio_path: Path) -> Dict:
        """Detect language using acoustic features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract acoustic features
            features = self._extract_acoustic_features(y, sr)
            
            # Simple rule-based classification based on acoustic properties
            language = self._classify_by_acoustic_features(features)
            
            return {
                "language": language,
                "confidence": 0.4,  # Lower confidence for acoustic-only detection
                "method": "acoustic_features",
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Acoustic feature detection failed: {e}")
            return {
                "language": "en",
                "confidence": 0.2,
                "method": "acoustic_fallback"
            }
    
    def _extract_acoustic_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract acoustic features for language classification"""
        try:
            # Fundamental frequency statistics
            f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr)
            f0_clean = f0[f0 > 0]
            
            if len(f0_clean) > 0:
                f0_mean = np.mean(f0_clean)
                f0_std = np.std(f0_clean)
                f0_range = np.max(f0_clean) - np.min(f0_clean)
            else:
                f0_mean = f0_std = f0_range = 0
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Rhythm and timing features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            onset_rate = len(onset_frames) / (len(audio) / sr)
            
            return {
                "f0_mean": float(f0_mean),
                "f0_std": float(f0_std),
                "f0_range": float(f0_range),
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "mfcc_mean": np.mean(mfccs, axis=1).tolist(),
                "tempo": float(tempo),
                "onset_rate": float(onset_rate)
            }
            
        except Exception as e:
            logger.error(f"Acoustic feature extraction failed: {e}")
            return {}
    
    def _classify_by_acoustic_features(self, features: Dict) -> str:
        """Simple rule-based language classification using acoustic features"""
        try:
            f0_mean = features.get("f0_mean", 150)
            spectral_centroid = features.get("spectral_centroid_mean", 2000)
            onset_rate = features.get("onset_rate", 5)
            
            # Simple heuristics (these would be better with trained models)
            if f0_mean > 200 and spectral_centroid > 2500:
                # Higher pitch and brighter spectrum - possibly Hindi/Urdu
                if onset_rate > 6:
                    return "hi"  # Hindi tends to be more rhythmic
                else:
                    return "ur"  # Urdu
            elif f0_mean < 150 and spectral_centroid < 2000:
                # Lower pitch and darker spectrum - possibly Arabic
                return "ar"
            else:
                # Default to English
                return "en"
                
        except Exception as e:
            logger.error(f"Acoustic classification failed: {e}")
            return "en"
    
    def _combine_detection_results(self, audio_result: Dict, 
                                 text_result: Optional[Dict],
                                 acoustic_result: Dict) -> Dict:
        """Combine multiple detection results into final decision"""
        try:
            results = [audio_result]
            
            if text_result:
                results.append(text_result)
            
            results.append(acoustic_result)
            
            # Weight the results based on confidence and method reliability
            weights = {
                "whisper_audio": 0.6,
                "text_classification": 0.8,
                "acoustic_features": 0.2,
                "audio_fallback": 0.3,
                "acoustic_fallback": 0.1
            }
            
            # Calculate weighted scores for each language
            language_scores = {}
            
            for result in results:
                lang = result["language"]
                confidence = result["confidence"]
                method = result["method"]
                weight = weights.get(method, 0.5)
                
                weighted_score = confidence * weight
                
                if lang in language_scores:
                    language_scores[lang] += weighted_score
                else:
                    language_scores[lang] = weighted_score
            
            # Find the language with highest score
            if language_scores:
                best_language = max(language_scores, key=language_scores.get)
                best_score = language_scores[best_language]
                
                # Normalize confidence
                total_weight = sum(weights[r["method"]] for r in results if r["method"] in weights)
                normalized_confidence = min(best_score / total_weight, 1.0)
                
                return {
                    "language": best_language,
                    "confidence": normalized_confidence,
                    "method": "combined",
                    "all_results": results,
                    "language_scores": language_scores
                }
            
            # Fallback to audio result
            return audio_result
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            return audio_result
    
    def _calculate_audio_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence for Whisper audio detection"""
        try:
            # Base confidence from Whisper
            base_confidence = 0.7
            
            # Adjust based on transcription quality
            text = whisper_result.get("text", "").strip()
            
            if len(text) < 10:
                return 0.3  # Very short transcription
            elif len(text) < 50:
                return 0.5  # Short transcription
            else:
                return min(base_confidence + len(text) / 1000, 0.9)
                
        except Exception:
            return 0.5
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-alphabetic characters except spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
