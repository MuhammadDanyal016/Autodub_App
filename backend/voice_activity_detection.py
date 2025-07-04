"""
Enhanced Voice Activity Detection (VAD) for speaker analysis
"""

import numpy as np
import librosa
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import webrtcvad
import wave
import struct

logger = logging.getLogger(__name__)

class EnhancedVAD:
    def __init__(self):
        self.sample_rate = 16000  # WebRTC VAD requires 16kHz
        self.frame_duration = 30  # ms (10, 20, or 30)
        self.vad_mode = 3  # Aggressiveness (0-3, 3 is most aggressive)
        self.temp_files = []
        
        # Initialize WebRTC VAD
        try:
            self.vad = webrtcvad.Vad(self.vad_mode)
            self.webrtc_available = True
            logger.info("✅ WebRTC VAD initialized")
        except Exception as e:
            logger.warning(f"WebRTC VAD initialization failed: {e}")
            self.webrtc_available = False
    
    def detect_speech_segments(self, audio_path: str, min_speech_duration: float = 0.5,
                              min_silence_duration: float = 0.3) -> List[Dict]:
        """Detect speech segments in audio with enhanced accuracy"""
        try:
            logger.info(f"🎤 Detecting speech segments in: {audio_path}")
            
            # Convert audio to required format for WebRTC VAD
            converted_audio = self._convert_audio_for_vad(audio_path)
            
            if self.webrtc_available:
                # Use WebRTC VAD for primary detection
                speech_segments = self._webrtc_vad_detection(converted_audio)
            else:
                # Fallback to energy-based VAD
                speech_segments = self._energy_based_vad(converted_audio)
            
            # Post-process segments
            processed_segments = self._post_process_segments(
                speech_segments, min_speech_duration, min_silence_duration
            )
            
            # Cleanup temp file
            if converted_audio != audio_path:
                Path(converted_audio).unlink(missing_ok=True)
            
            logger.info(f"✅ Detected {len(processed_segments)} speech segments")
            return processed_segments
            
        except Exception as e:
            logger.error(f"Speech segment detection failed: {e}")
            return self._create_fallback_segments(audio_path)
    
    def _convert_audio_for_vad(self, audio_path: str) -> str:
        """Convert audio to format required by WebRTC VAD"""
        try:
            # Check if already in correct format
            with wave.open(audio_path, 'rb') as wav_file:
                if (wav_file.getframerate() == self.sample_rate and 
                    wav_file.getnchannels() == 1 and 
                    wav_file.getsampwidth() == 2):
                    return audio_path
        except:
            pass
        
        # Convert using ffmpeg
        temp_audio = tempfile.mktemp(suffix=".wav")
        self.temp_files.append(temp_audio)
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(audio_path),
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', str(self.sample_rate),
            '-loglevel', 'error',
            temp_audio
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        if not Path(temp_audio).exists():
            raise Exception("Audio conversion for VAD failed")
        
        return temp_audio
    
    def _webrtc_vad_detection(self, audio_path: str) -> List[Dict]:
        """Perform VAD using WebRTC"""
        try:
            # Read audio file
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                
            # Calculate frame size in samples
            frame_size = int(sample_rate * self.frame_duration / 1000)
            
            # Process audio in frames
            speech_frames = []
            for i in range(0, len(frames), frame_size * 2):  # 2 bytes per sample
                frame = frames[i:i + frame_size * 2]
                
                # Ensure frame is correct size
                if len(frame) < frame_size * 2:
                    frame += b'\x00' * (frame_size * 2 - len(frame))
                
                # Check if frame contains speech
                try:
                    is_speech = self.vad.is_speech(frame, sample_rate)
                    timestamp = i / (2 * sample_rate)  # Convert to seconds
                    speech_frames.append({
                        'timestamp': timestamp,
                        'is_speech': is_speech,
                        'frame_duration': self.frame_duration / 1000
                    })
                except Exception as e:
                    logger.debug(f"VAD frame processing failed: {e}")
                    continue
            
            # Convert frame-level decisions to segments
            segments = self._frames_to_segments(speech_frames)
            
            return segments
            
        except Exception as e:
            logger.error(f"WebRTC VAD detection failed: {e}")
            return self._energy_based_vad(audio_path)
    
    def _energy_based_vad(self, audio_path: str) -> List[Dict]:
        """Fallback energy-based VAD"""
        try:
            logger.info("Using energy-based VAD fallback")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Calculate frame-level energy
            frame_length = int(sr * self.frame_duration / 1000)
            hop_length = frame_length // 2
            
            # RMS energy
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Calculate dynamic threshold
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            threshold = rms_mean + 0.5 * rms_std
            
            # Detect speech frames
            speech_frames = []
            for i, energy in enumerate(rms):
                timestamp = i * hop_length / sr
                is_speech = energy > threshold
                speech_frames.append({
                    'timestamp': timestamp,
                    'is_speech': is_speech,
                    'frame_duration': hop_length / sr,
                    'energy': float(energy)
                })
            
            # Convert to segments
            segments = self._frames_to_segments(speech_frames)
            
            return segments
            
        except Exception as e:
            logger.error(f"Energy-based VAD failed: {e}")
            return []
    
    def _frames_to_segments(self, speech_frames: List[Dict]) -> List[Dict]:
        """Convert frame-level speech decisions to segments"""
        try:
            if not speech_frames:
                return []
            
            segments = []
            current_segment = None
            
            for frame in speech_frames:
                timestamp = frame['timestamp']
                is_speech = frame['is_speech']
                frame_duration = frame['frame_duration']
                
                if is_speech:
                    if current_segment is None:
                        # Start new speech segment
                        current_segment = {
                            'start': timestamp,
                            'end': timestamp + frame_duration,
                            'type': 'speech'
                        }
                    else:
                        # Extend current segment
                        current_segment['end'] = timestamp + frame_duration
                else:
                    if current_segment is not None:
                        # End current speech segment
                        current_segment['duration'] = current_segment['end'] - current_segment['start']
                        segments.append(current_segment)
                        current_segment = None
            
            # Handle last segment
            if current_segment is not None:
                current_segment['duration'] = current_segment['end'] - current_segment['start']
                segments.append(current_segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Frame to segment conversion failed: {e}")
            return []
    
    def _post_process_segments(self, segments: List[Dict], min_speech_duration: float,
                              min_silence_duration: float) -> List[Dict]:
        """Post-process segments to remove short segments and merge close ones"""
        try:
            if not segments:
                return []
            
            # Filter out short speech segments
            filtered_segments = []
            for segment in segments:
                if segment.get('duration', 0) >= min_speech_duration:
                    filtered_segments.append(segment)
            
            if not filtered_segments:
                return []
            
            # Merge segments that are close together
            merged_segments = []
            current_segment = filtered_segments[0].copy()
            
            for next_segment in filtered_segments[1:]:
                gap = next_segment['start'] - current_segment['end']
                
                if gap <= min_silence_duration:
                    # Merge segments
                    current_segment['end'] = next_segment['end']
                    current_segment['duration'] = current_segment['end'] - current_segment['start']
                else:
                    # Add current segment and start new one
                    merged_segments.append(current_segment)
                    current_segment = next_segment.copy()
            
            # Add last segment
            merged_segments.append(current_segment)
            
            # Add segment IDs and additional info
            for i, segment in enumerate(merged_segments):
                segment['id'] = i
                segment['confidence'] = 0.8  # Default confidence for VAD
            
            return merged_segments
            
        except Exception as e:
            logger.error(f"Segment post-processing failed: {e}")
            return segments
    
    def _create_fallback_segments(self, audio_path: str) -> List[Dict]:
        """Create fallback segments when VAD fails"""
        try:
            # Get audio duration
            duration = librosa.get_duration(path=audio_path)
            
            # Create segments every 10 seconds
            segments = []
            segment_duration = 10.0
            
            for i, start in enumerate(np.arange(0, duration, segment_duration)):
                end = min(start + segment_duration, duration)
                segment = {
                    'id': i,
                    'start': float(start),
                    'end': float(end),
                    'duration': float(end - start),
                    'type': 'speech',
                    'confidence': 0.5,
                    'method': 'fallback'
                }
                segments.append(segment)
            
            logger.info(f"✅ Created {len(segments)} fallback segments")
            return segments
            
        except Exception as e:
            logger.error(f"Fallback segment creation failed: {e}")
            return [{
                'id': 0,
                'start': 0.0,
                'end': 60.0,
                'duration': 60.0,
                'type': 'speech',
                'confidence': 0.3,
                'method': 'default'
            }]
    
    def analyze_speech_quality(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        """Analyze speech quality for each segment"""
        try:
            logger.info("🔍 Analyzing speech quality...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            enhanced_segments = []
            
            for segment in segments:
                try:
                    start_sample = int(segment['start'] * sr)
                    end_sample = int(segment['end'] * sr)
                    segment_audio = audio[start_sample:end_sample]
                    
                    if len(segment_audio) == 0:
                        continue
                    
                    # Calculate quality metrics
                    quality_metrics = self._calculate_quality_metrics(segment_audio, sr)
                    
                    # Add quality info to segment
                    enhanced_segment = segment.copy()
                    enhanced_segment.update(quality_metrics)
                    enhanced_segments.append(enhanced_segment)
                    
                except Exception as e:
                    logger.debug(f"Quality analysis failed for segment {segment.get('id', 'unknown')}: {e}")
                    enhanced_segments.append(segment)
            
            return enhanced_segments
            
        except Exception as e:
            logger.error(f"Speech quality analysis failed: {e}")
            return segments
    
    def _calculate_quality_metrics(self, audio: np.ndarray, sr: int) -> Dict:
        """Calculate speech quality metrics"""
        try:
            # Signal-to-noise ratio estimation
            snr = self._estimate_snr(audio)
            
            # Zero crossing rate (speech clarity indicator)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = float(np.mean(zcr))
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_centroid_mean = float(np.mean(spectral_centroid))
            
            # RMS energy (volume)
            rms = librosa.feature.rms(y=audio)[0]
            rms_mean = float(np.mean(rms))
            
            # Overall quality score (0-1)
            quality_score = self._calculate_quality_score(snr, zcr_mean, rms_mean)
            
            return {
                'snr_db': snr,
                'zero_crossing_rate': zcr_mean,
                'spectral_centroid': spectral_centroid_mean,
                'rms_energy': rms_mean,
                'quality_score': quality_score,
                'quality_level': self._get_quality_level(quality_score)
            }
            
        except Exception as e:
            logger.debug(f"Quality metrics calculation failed: {e}")
            return {
                'quality_score': 0.5,
                'quality_level': 'medium'
            }
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        try:
            # Simple SNR estimation using signal variance
            signal_power = np.var(audio)
            
            # Estimate noise from quiet portions (bottom 10% of energy)
            frame_energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
            noise_threshold = np.percentile(frame_energy, 10)
            noise_frames = frame_energy <= noise_threshold
            
            if np.any(noise_frames):
                noise_power = np.mean(frame_energy[noise_frames]) ** 2
                if noise_power > 0:
                    snr_linear = signal_power / noise_power
                    snr_db = 10 * np.log10(snr_linear)
                    return float(np.clip(snr_db, -20, 40))  # Reasonable range
            
            return 10.0  # Default moderate SNR
            
        except Exception:
            return 10.0
    
    def _calculate_quality_score(self, snr: float, zcr: float, rms: float) -> float:
        """Calculate overall quality score"""
        try:
            # Normalize metrics to 0-1 range
            snr_score = np.clip((snr + 10) / 30, 0, 1)  # -10 to 20 dB range
            zcr_score = np.clip(zcr / 0.1, 0, 1)  # Typical ZCR range
            rms_score = np.clip(rms / 0.1, 0, 1)  # Typical RMS range
            
            # Weighted combination
            quality_score = (0.5 * snr_score + 0.3 * rms_score + 0.2 * zcr_score)
            
            return float(np.clip(quality_score, 0, 1))
            
        except Exception:
            return 0.5
    
    def _get_quality_level(self, score: float) -> str:
        """Convert quality score to level"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'medium'
        elif score >= 0.2:
            return 'poor'
        else:
            return 'very_poor'
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            for temp_file in self.temp_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception as e:
                    logger.debug(f"VAD temp file cleanup failed: {e}")
            self.temp_files.clear()
        except Exception as e:
            logger.warning(f"VAD cleanup failed: {e}")
