"""
Enhanced video processing with optimization and format handling
"""

import cv2
import numpy as np
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

logger = logging.getLogger(__name__)

class EnhancedVideoProcessor:
    def __init__(self):
        self.temp_files = []
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']
        self.target_fps = 25
        self.max_resolution = (1280, 720)
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get comprehensive video information"""
        try:
            logger.info(f"📹 Analyzing video: {video_path}")
            
            # Use OpenCV to get basic info
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            # Basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Use ffprobe for detailed info
            detailed_info = self._get_ffprobe_info(video_path)
            
            video_info = {
                'file_path': str(video_path),
                'file_size': Path(video_path).stat().st_size,
                'file_size_mb': round(Path(video_path).stat().st_size / (1024*1024), 2),
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'aspect_ratio': round(width / height, 2) if height > 0 else 0,
                'format': Path(video_path).suffix.lower(),
                'detailed_info': detailed_info
            }
            
            logger.info(f"✅ Video info: {video_info['resolution']} @ {fps:.1f}fps, {duration:.1f}s")
            return video_info
            
        except Exception as e:
            logger.error(f"Video info extraction failed: {e}")
            return {
                'file_path': str(video_path),
                'error': str(e),
                'duration': 0,
                'fps': 25,
                'width': 0,
                'height': 0
            }
    
    def _get_ffprobe_info(self, video_path: str) -> Dict:
        """Get detailed video info using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
            
        except Exception as e:
            logger.debug(f"ffprobe failed: {e}")
            return {}
    
    def optimize_video_for_processing(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Optimize video for processing pipeline"""
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".mp4")
            
            self.temp_files.append(output_path)
            
            logger.info(f"🔧 Optimizing video for processing...")
            
            # Get video info
            video_info = self.get_video_info(video_path)
            
            # Determine optimization parameters
            width = video_info.get('width', 1280)
            height = video_info.get('height', 720)
            fps = video_info.get('fps', 25)
            
            # Calculate target resolution (maintain aspect ratio)
            target_width, target_height = self._calculate_target_resolution(width, height)
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', f'scale={target_width}:{target_height}',
                '-r', str(self.target_fps),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-loglevel', 'error',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not Path(output_path).exists():
                raise Exception("Video optimization failed")
            
            optimized_info = self.get_video_info(output_path)
            logger.info(f"✅ Video optimized: {optimized_info['resolution']} @ {optimized_info['fps']:.1f}fps")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Video optimization failed: {e}")
            return video_path
    
    def _calculate_target_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Calculate target resolution maintaining aspect ratio"""
        try:
            max_width, max_height = self.max_resolution
            
            # If already within limits, keep original
            if width <= max_width and height <= max_height:
                return width, height
            
            # Calculate scaling factor
            width_scale = max_width / width
            height_scale = max_height / height
            scale = min(width_scale, height_scale)
            
            # Calculate new dimensions (ensure even numbers for video encoding)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Ensure even numbers
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            return new_width, new_height
            
        except Exception as e:
            logger.error(f"Resolution calculation failed: {e}")
            return 1280, 720
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_interval: int = 30) -> List[str]:
        """Extract frames from video at specified intervals"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"🖼️ Extracting frames every {frame_interval} frames...")
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            frame_paths = []
            frame_number = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % frame_interval == 0:
                    frame_path = output_dir / f"frame_{extracted_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
                    extracted_count += 1
                
                frame_number += 1
            
            cap.release()
            
            logger.info(f"✅ Extracted {len(frame_paths)} frames")
            return frame_paths
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    def create_video_from_frames(self, frame_paths: List[str], output_path: str, 
                                fps: float = 25) -> str:
        """Create video from frame sequence"""
        try:
            if not frame_paths:
                raise Exception("No frames provided")
            
            logger.info(f"🎬 Creating video from {len(frame_paths)} frames...")
            
            # Get frame dimensions
            first_frame = cv2.imread(frame_paths[0])
            if first_frame is None:
                raise Exception("Could not read first frame")
            
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
            
            out.release()
            
            if not Path(output_path).exists():
                raise Exception("Video creation failed")
            
            logger.info(f"✅ Video created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            raise
    
    def combine_video_audio(self, video_path: str, audio_path: str, 
                           output_path: Optional[str] = None) -> str:
        """Combine video and audio tracks"""
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".mp4")
            
            self.temp_files.append(output_path)
            
            logger.info(f"🎵 Combining video and audio...")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                '-loglevel', 'error',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not Path(output_path).exists():
                raise Exception("Video/audio combination failed")
            
            logger.info(f"✅ Video and audio combined: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video/audio combination failed: {e}")
            raise
    
    def trim_video(self, video_path: str, start_time: float, end_time: float,
                   output_path: Optional[str] = None) -> str:
        """Trim video to specified time range"""
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".mp4")
            
            self.temp_files.append(output_path)
            
            duration = end_time - start_time
            logger.info(f"✂️ Trimming video: {start_time:.2f}s to {end_time:.2f}s ({duration:.2f}s)")
            
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', str(video_path),
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-loglevel', 'error',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not Path(output_path).exists():
                raise Exception("Video trimming failed")
            
            logger.info(f"✅ Video trimmed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video trimming failed: {e}")
            raise
    
    def resize_video(self, video_path: str, width: int, height: int,
                     output_path: Optional[str] = None) -> str:
        """Resize video to specified dimensions"""
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".mp4")
            
            self.temp_files.append(output_path)
            
            logger.info(f"📐 Resizing video to {width}x{height}...")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', f'scale={width}:{height}',
                '-c:a', 'copy',
                '-loglevel', 'error',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not Path(output_path).exists():
                raise Exception("Video resizing failed")
            
            logger.info(f"✅ Video resized: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video resizing failed: {e}")
            raise
    
    def convert_format(self, video_path: str, target_format: str,
                       output_path: Optional[str] = None) -> str:
        """Convert video to different format"""
        try:
            if not target_format.startswith('.'):
                target_format = f'.{target_format}'
            
            if output_path is None:
                base_name = Path(video_path).stem
                output_path = tempfile.mktemp(suffix=target_format)
            
            self.temp_files.append(output_path)
            
            logger.info(f"🔄 Converting to {target_format} format...")
            
            # Format-specific settings
            codec_settings = self._get_codec_settings(target_format)
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path)
            ] + codec_settings + [
                '-loglevel', 'error',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not Path(output_path).exists():
                raise Exception("Format conversion failed")
            
            logger.info(f"✅ Format converted: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            raise
    
    def _get_codec_settings(self, format_ext: str) -> List[str]:
        """Get codec settings for different formats"""
        format_settings = {
            '.mp4': ['-c:v', 'libx264', '-c:a', 'aac', '-preset', 'fast'],
            '.avi': ['-c:v', 'libx264', '-c:a', 'mp3'],
            '.mov': ['-c:v', 'libx264', '-c:a', 'aac'],
            '.mkv': ['-c:v', 'libx264', '-c:a', 'aac'],
            '.webm': ['-c:v', 'libvpx-vp9', '-c:a', 'libopus'],
            '.flv': ['-c:v', 'libx264', '-c:a', 'aac']
        }
        
        return format_settings.get(format_ext, ['-c:v', 'libx264', '-c:a', 'aac'])
    
    def add_watermark(self, video_path: str, watermark_text: str,
                      output_path: Optional[str] = None) -> str:
        """Add text watermark to video"""
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".mp4")
            
            self.temp_files.append(output_path)
            
            logger.info(f"🏷️ Adding watermark: {watermark_text}")
            
            # Escape special characters for ffmpeg
            escaped_text = watermark_text.replace("'", "\\'").replace(":", "\\:")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', f"drawtext=text='{escaped_text}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=h-th-10",
                '-c:a', 'copy',
                '-loglevel', 'error',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not Path(output_path).exists():
                raise Exception("Watermark addition failed")
            
            logger.info(f"✅ Watermark added: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Watermark addition failed: {e}")
            raise
    
    def create_video_preview(self, video_path: str, preview_duration: int = 30,
                            output_path: Optional[str] = None) -> str:
        """Create a preview of the video"""
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix=".mp4")
            
            self.temp_files.append(output_path)
            
            # Get video info to determine preview segments
            video_info = self.get_video_info(video_path)
            total_duration = video_info.get('duration', 0)
            
            if total_duration <= preview_duration:
                # If video is shorter than preview duration, return original
                return video_path
            
            # Create preview from multiple segments
            segment_duration = min(10, preview_duration // 3)
            segments = [
                (0, segment_duration),  # Beginning
                (total_duration // 2 - segment_duration // 2, total_duration // 2 + segment_duration // 2),  # Middle
                (total_duration - segment_duration, total_duration)  # End
            ]
            
            logger.info(f"🎬 Creating {preview_duration}s preview from {total_duration:.1f}s video...")
            
            # Create filter complex for concatenating segments
            filter_parts = []
            input_parts = []
            
            for i, (start, end) in enumerate(segments):
                input_parts.extend(['-ss', str(start), '-t', str(end - start), '-i', str(video_path)])
                filter_parts.append(f'[{i}:v][{i}:a]')
            
            filter_complex = ''.join(filter_parts) + f'concat=n={len(segments)}:v=1:a=1[outv][outa]'
            
            cmd = [
                'ffmpeg', '-y'
            ] + input_parts + [
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-loglevel', 'error',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not Path(output_path).exists():
                raise Exception("Preview creation failed")
            
            logger.info(f"✅ Preview created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Preview creation failed: {e}")
            # Fallback: just trim from beginning
            return self.trim_video(video_path, 0, min(preview_duration, 30))
    
    def validate_video(self, video_path: str) -> Dict:
        """Validate video file and return status"""
        try:
            video_path = Path(video_path)
            
            # Check file existence
            if not video_path.exists():
                return {'valid': False, 'error': 'File does not exist'}
            
            # Check file size
            if video_path.stat().st_size == 0:
                return {'valid': False, 'error': 'File is empty'}
            
            # Check format
            if video_path.suffix.lower() not in self.supported_formats:
                return {
                    'valid': False, 
                    'error': f'Unsupported format. Supported: {self.supported_formats}'
                }
            
            # Try to get video info
            video_info = self.get_video_info(str(video_path))
            
            if 'error' in video_info:
                return {'valid': False, 'error': video_info['error']}
            
            # Check duration
            duration = video_info.get('duration', 0)
            if duration <= 0:
                return {'valid': False, 'error': 'Invalid video duration'}
            
            # Check resolution
            width = video_info.get('width', 0)
            height = video_info.get('height', 0)
            if width <= 0 or height <= 0:
                return {'valid': False, 'error': 'Invalid video resolution'}
            
            return {
                'valid': True,
                'info': video_info,
                'message': 'Video is valid and ready for processing'
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            for temp_file in self.temp_files:
                try:
                    if Path(temp_file).exists():
                        Path(temp_file).unlink()
                except Exception as e:
                    logger.debug(f"Temp file cleanup failed: {e}")
            self.temp_files.clear()
            logger.info("✅ Video processor cleanup completed")
        except Exception as e:
            logger.warning(f"Video processor cleanup failed: {e}")
