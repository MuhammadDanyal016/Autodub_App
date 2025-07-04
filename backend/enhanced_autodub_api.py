"""
Enhanced AutoDub API v4.0 - FIXED: Prevent model cleanup between requests
"""

import sys
import os
import logging
import asyncio
import tempfile
import uuid
import time
import json
import subprocess
import gc
import threading
import weakref
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException


# Import configuration
try:
    from config import Config
except ImportError:
    # Fallback configuration for Colab
    class Config:
        TEMP_DIR = Path("/tmp/autodub")
        OUTPUT_DIR = Path("/tmp/autodub/output")
        MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
        VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
        SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
        EDGE_TTS_VOICES = {}
        
        @staticmethod
        def setup_cuda_environment():
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import actual pipeline components
try:
    from core.enhanced_autodub_pipeline import EnhancedAutoDubPipeline
    from core.enhanced_autodub_processor import EnhancedAutoDubProcessor
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"⚠️ Pipeline components not available: {e}")

# Import utilities
try:
    from utils.gpu_utils import gpu_manager
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    print("⚠️ GPU utils not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/enhanced_autodub_api.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# Global variables - SINGLE SHARED INSTANCES
pipeline = None
processor = None
executor = ThreadPoolExecutor(max_workers=3)
processing_status = {}
cloudflare_tunnel = None
processor_lock = threading.Lock()  # Ensure thread-safe access to shared processor

# Create necessary directories
Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ProcessingSession:
    """Lightweight session wrapper for tracking individual processing jobs"""
    
    def __init__(self, process_id: str, video_path: str, target_language: str, source_language: Optional[str]):
        self.process_id = process_id
        self.video_path = video_path
        self.target_language = target_language
        self.source_language = source_language
        self.start_time = time.time()
        self.temp_files = []  # Track temporary files for cleanup
    
    def add_temp_file(self, file_path: str):
        """Add a temporary file to cleanup list"""
        self.temp_files.append(file_path)
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing"""
        for temp_file in self.temp_files:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
                    logger.info(f"🗑️ Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temp file {temp_file}: {e}")
        self.temp_files.clear()

class SharedProcessorWrapper:
    """Enhanced wrapper to prevent ALL model cleanup - both API and pipeline level"""
    
    def __init__(self, processor_instance):
        self.processor = processor_instance
        self.is_shared = True
        self.active_sessions = set()
        self.lock = threading.Lock()
        self._original_cleanup_methods = {}
        self._patch_pipeline_cleanup()
    
    def _patch_pipeline_cleanup(self):
        """Patch all pipeline component cleanup methods to prevent model unloading"""
        try:
            logger.info("🔧 Patching pipeline component cleanup methods...")
            
            # COMPREHENSIVE Speech Recognition Patching
            speech_recognizers = []
        
            # Find all possible speech recognizer instances
            if hasattr(self.processor, 'pipeline') and hasattr(self.processor.pipeline, 'speech_recognizer'):
                speech_recognizers.append(('pipeline_speech_recognizer', self.processor.pipeline.speech_recognizer))
        
            if hasattr(self.processor, 'speech_recognizer'):
                speech_recognizers.append(('direct_speech_recognizer', self.processor.speech_recognizer))
        
            # Also check for nested speech recognizers
            if hasattr(self.processor, 'pipeline'):
                for attr_name in dir(self.processor.pipeline):
                    attr = getattr(self.processor.pipeline, attr_name, None)
                    if attr and hasattr(attr, 'model') and hasattr(attr, 'faster_whisper'):
                        speech_recognizers.append((f'nested_{attr_name}', attr))
        
            # Patch all found speech recognizers
            for name, speech_recognizer in speech_recognizers:
                if hasattr(speech_recognizer, 'cleanup'):
                    self._original_cleanup_methods[name] = speech_recognizer.cleanup
                    speech_recognizer.cleanup = self._create_speech_recognition_dummy_cleanup(speech_recognizer, name)
                    logger.info(f"✅ Patched {name} cleanup with model preservation")
            
                # Also patch any model-specific cleanup methods
                if hasattr(speech_recognizer, '_cleanup_models'):
                    self._original_cleanup_methods[f'{name}_models'] = speech_recognizer._cleanup_models
                    speech_recognizer._cleanup_models = lambda: logger.info(f"🚫 Model cleanup intercepted for {name}")
            
                # Patch model property setters to prevent None assignment
                if hasattr(speech_recognizer, 'model'):
                    original_model = speech_recognizer.model
                    def create_model_property(original):
                        def model_setter(value):
                            if value is None:
                                logger.info(f"🚫 Prevented model=None assignment for {name}")
                                return  # Don't set to None
                            original = value
                        return model_setter
                
                    # Store original model reference
                    self._original_cleanup_methods[f'{name}_original_model'] = speech_recognizer.model
            
                # Patch faster_whisper property setters
                if hasattr(speech_recognizer, 'faster_whisper'):
                    original_faster_whisper = speech_recognizer.faster_whisper
                    def create_faster_whisper_property(original):
                        def faster_whisper_setter(value):
                            if value is None:
                                logger.info(f"🚫 Prevented faster_whisper=None assignment for {name}")
                                return  # Don't set to None
                            original = value
                        return faster_whisper_setter
                
                    # Store original faster_whisper reference
                    self._original_cleanup_methods[f'{name}_original_faster_whisper'] = speech_recognizer.faster_whisper
        
            # Patch other component cleanups
            components_to_patch = [
                'audio_processor',
                'speaker_diarizer', 
                'gender_detector',
                'emotion_detector',
                'translator',
                'tts_generator'
            ]
        
            for component_name in components_to_patch:
                if hasattr(self.processor, 'pipeline'):
                    component = getattr(self.processor.pipeline, component_name, None)
                    if component and hasattr(component, 'cleanup'):
                        self._original_cleanup_methods[component_name] = component.cleanup
                        component.cleanup = self._dummy_cleanup
                        logger.info(f"✅ Patched {component_name} cleanup")
        
            # Patch main processor cleanup
            if hasattr(self.processor, 'cleanup'):
                self._original_cleanup_methods['main_processor'] = self.processor.cleanup
                self.processor.cleanup = self._dummy_cleanup
                logger.info("✅ Patched main processor cleanup")
        
            # CRITICAL: Also patch gpu_manager cleanup calls
            try:
                from utils.gpu_utils import gpu_manager
                if hasattr(gpu_manager, 'cleanup_model'):
                    original_cleanup_model = gpu_manager.cleanup_model
                    def patched_cleanup_model(model):
                        logger.info("🚫 GPU manager model cleanup intercepted - preserving model")
                        # Don't actually cleanup the model
                        return
                    gpu_manager.cleanup_model = patched_cleanup_model
                    self._original_cleanup_methods['gpu_manager_cleanup_model'] = original_cleanup_model
                    logger.info("✅ Patched GPU manager cleanup_model")
            except Exception as e:
                logger.warning(f"Could not patch GPU manager: {e}")
        
            logger.info("🎯 All pipeline cleanup methods patched - models will persist")
        
        except Exception as e:
            logger.error(f"❌ Failed to patch cleanup methods: {e}")

    def _create_speech_recognition_dummy_cleanup(self, speech_recognizer, name):
        """Create a specialized dummy cleanup for speech recognition that preserves models"""
        def speech_recognition_dummy_cleanup():
            logger.info(f"🚫 Speech recognition cleanup intercepted for {name} - preserving models")
            try:
                # Only export performance report, don't unload models
                if hasattr(speech_recognizer, 'export_performance_report'):
                    speech_recognizer.export_performance_report()
                if hasattr(speech_recognizer, 'print_performance_summary'):
                    speech_recognizer.print_performance_summary()
                logger.info(f"📊 Performance metrics exported for {name}, models preserved")
            except Exception as e:
                logger.warning(f"Performance export warning for {name}: {e}")
        
            # CRITICAL: Ensure models are NOT set to None
            if hasattr(speech_recognizer, 'model') and speech_recognizer.model is not None:
                logger.info(f"🔒 Model preserved for {name}: {type(speech_recognizer.model)}")
            if hasattr(speech_recognizer, 'faster_whisper') and speech_recognizer.faster_whisper is not None:
                logger.info(f"🔒 Faster-whisper preserved for {name}: {type(speech_recognizer.faster_whisper)}")
        
            # DO NOT call gpu_manager.cleanup_model() or set models to None
            # DO NOT clear GPU cache aggressively
            logger.info(f"✅ {name} cleanup completed with model preservation")

        return speech_recognition_dummy_cleanup
    
    def _dummy_cleanup(self):
        """Dummy cleanup that does nothing to preserve models"""
        logger.info("🚫 Cleanup call intercepted - models preserved")
        pass
    
    def start_session(self, session_id: str):
        """Register a new processing session"""
        with self.lock:
            self.active_sessions.add(session_id)
            logger.info(f"🎯 Session {session_id} started (active: {len(self.active_sessions)})")
    
    def end_session(self, session_id: str):
        """End a processing session"""
        with self.lock:
            self.active_sessions.discard(session_id)
            logger.info(f"🎯 Session {session_id} ended (active: {len(self.active_sessions)})")
    
    def process_video(self, *args, **kwargs):
        """Process video with session tracking"""
        return self.processor.process_video(*args, **kwargs)
    
    def should_cleanup_models(self) -> bool:
        """Check if models should be cleaned up (only if no active sessions)"""
        with self.lock:
            return len(self.active_sessions) == 0
    
    def lightweight_cleanup(self):
        """Perform lightweight cleanup without touching models"""
        try:
            logger.info("🧹 Performing lightweight cleanup (models completely preserved)...")
            
            # Only clean up temporary files and cache, NOT models
            if hasattr(self.processor, 'cleanup_temp_files'):
                self.processor.cleanup_temp_files()
            
            # Light GPU cache cleanup
            if GPU_UTILS_AVAILABLE:
                try:
                    gpu_manager.clear_gpu_cache()
                except Exception as e:
                    logger.warning(f"⚠️ GPU cache cleanup warning: {e}")
            
            # Light garbage collection
            gc.collect()
            
            logger.info("✅ Lightweight cleanup completed (ALL models preserved)")
            
        except Exception as e:
            logger.error(f"❌ Lightweight cleanup error: {e}")
    
    def restore_cleanup_methods(self):
        """Restore original cleanup methods (only called during shutdown)"""
        try:
            logger.info("🔄 Restoring original cleanup methods for shutdown...")
        
            # Restore GPU manager
            if 'gpu_manager_cleanup_model' in self._original_cleanup_methods:
                try:
                    from utils.gpu_utils import gpu_manager
                    gpu_manager.cleanup_model = self._original_cleanup_methods['gpu_manager_cleanup_model']
                    logger.info("✅ Restored GPU manager cleanup_model")
                except Exception as e:
                    logger.warning(f"Could not restore GPU manager: {e}")
        
            # Restore speech recognizer cleanup
            for name, original_cleanup in self._original_cleanup_methods.items():
                if 'speech_recognizer' in name and not name.endswith('_original_model') and not name.endswith('_original_faster_whisper'):
                    # Find the component and restore its cleanup
                    if name == 'pipeline_speech_recognizer':
                        if hasattr(self.processor, 'pipeline') and hasattr(self.processor.pipeline, 'speech_recognizer'):
                            self.processor.pipeline.speech_recognizer.cleanup = original_cleanup
                    elif name == 'direct_speech_recognizer':
                        if hasattr(self.processor, 'speech_recognizer'):
                            self.processor.speech_recognizer.cleanup = original_cleanup
        
            # Restore other component cleanups
            for component_name, original_cleanup in self._original_cleanup_methods.items():
                if component_name == 'main_processor':
                    self.processor.cleanup = original_cleanup
                elif hasattr(self.processor, 'pipeline'):
                    component = getattr(self.processor.pipeline, component_name, None)
                    if component:
                        component.cleanup = original_cleanup
        
            logger.info("✅ Original cleanup methods restored")
        
        except Exception as e:
            logger.error(f"❌ Failed to restore cleanup methods: {e}")
    
    def full_cleanup(self):
        """Perform full cleanup including models (only when shutting down)"""
        try:
            logger.info("🧹 Performing FULL cleanup (restoring and calling original cleanup methods)...")
            
            # Restore original cleanup methods
            self.restore_cleanup_methods()
            
            # Now call the original cleanup
            if hasattr(self.processor, 'cleanup'):
                self.processor.cleanup()
            
            logger.info("✅ Full cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Full cleanup error: {e}")

class CloudflareTunnel:
    """Enhanced Cloudflare Tunnel manager with reconnection support"""
    
    def __init__(self):
        self.process = None
        self.tunnel_url = None
        self.tunnel_name = f"autodub-{int(time.time())}"
        self.is_healthy = False
        self.last_health_check = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
    
    def setup_tunnel(self, port: int = 8000) -> Optional[str]:
        """Setup Cloudflare Tunnel with enhanced error handling"""
        try:
            # Check if cloudflared is installed
            result = subprocess.run(['cloudflared', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("❌ cloudflared not installed. Install with: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb && sudo dpkg -i cloudflared.deb")
                return None
            
            logger.info("✅ cloudflared found, setting up tunnel...")
            
            # Enhanced tunnel command with better stability options
            cmd = [
                'cloudflared', 'tunnel',
                '--url', f'http://localhost:{port}',
                '--no-autoupdate',
                '--metrics', '127.0.0.1:8081',
                '--loglevel', 'info'
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for tunnel URL with better parsing
            timeout = 45
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error(f"❌ Cloudflare tunnel failed: {stderr}")
                    return None
                
                # Read from stderr where cloudflared outputs tunnel info
                try:
                    line = self.process.stderr.readline()
                    if line:
                        logger.debug(f"Cloudflare output: {line.strip()}")
                        # Look for tunnel URL patterns
                        if 'https://' in line and ('trycloudflare.com' in line or 'cfargotunnel.com' in line):
                            import re
                            url_patterns = [
                                r'https://[^\s\|]+\.trycloudflare\.com',
                                r'https://[^\s\|]+\.cfargotunnel\.com'
                            ]
                            for pattern in url_patterns:
                                url_match = re.search(pattern, line)
                                if url_match:
                                    self.tunnel_url = url_match.group(0).strip()
                                    self.is_healthy = True
                                    self.last_health_check = time.time()
                                    logger.info(f"🌐 Cloudflare tunnel active: {self.tunnel_url}")
                                    return self.tunnel_url
                        
                        # Also check for the "Visit it at" message format
                        if "Visit it at" in line:
                            url_line = self.process.stderr.readline()
                            if url_line and 'https://' in url_line:
                                import re
                                url_match = re.search(r'https://[^\s\|]+\.trycloudflare\.com', url_line)
                                if url_match:
                                    self.tunnel_url = url_match.group(0).strip()
                                    self.is_healthy = True
                                    self.last_health_check = time.time()
                                    logger.info(f"🌐 Cloudflare tunnel active: {self.tunnel_url}")
                                    return self.tunnel_url
                except Exception as e:
                    logger.debug(f"Error reading tunnel output: {e}")
                
                time.sleep(1)
            
            logger.warning("⚠️ Timeout waiting for Cloudflare tunnel URL")
            return None
            
        except FileNotFoundError:
            logger.error("❌ cloudflared not found. Please install cloudflared first.")
            return None
        except Exception as e:
            logger.error(f"❌ Cloudflare tunnel setup failed: {e}")
            return None
    
    def check_tunnel_health(self) -> bool:
        """Check if tunnel is still healthy"""
        try:
            if not self.process or self.process.poll() is not None:
                self.is_healthy = False
                return False
        
            # Check metrics endpoint if available
            try:
                import requests
                response = requests.get('http://127.0.0.1:8081/metrics', timeout=5)
                self.is_healthy = response.status_code == 200
                self.last_health_check = time.time()
                return self.is_healthy
            except:
                # Fallback: assume healthy if process is running
                self.is_healthy = True
                self.last_health_check = time.time()
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Tunnel health check failed: {e}")
            self.is_healthy = False
            return False
    
    def reconnect_tunnel(self, port: int = 8000) -> Optional[str]:
        """Attempt to reconnect the tunnel"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"❌ Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return None
        
        logger.info(f"🔄 Attempting tunnel reconnection (attempt {self.reconnect_attempts + 1})")
        self.reconnect_attempts += 1
        
        # Close existing tunnel
        self.close_tunnel()
        
        # Wait before reconnecting
        time.sleep(5)
        
        # Setup new tunnel
        return self.setup_tunnel(port)
    
    def get_tunnel_url(self) -> Optional[str]:
        """Get current tunnel URL with health check"""
        current_time = time.time()
        
        # Perform health check every 30 seconds
        if current_time - self.last_health_check > 30:
            if not self.check_tunnel_health():
                logger.warning("⚠️ Tunnel health check failed, attempting reconnection")
                return self.reconnect_tunnel()
        
        return self.tunnel_url if self.is_healthy else None
    
    def close_tunnel(self):
        """Close the tunnel with proper cleanup"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                logger.info("🛑 Cloudflare tunnel closed")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.warning("⚠️ Forced kill of Cloudflare tunnel process")
            except Exception as e:
                logger.error(f"❌ Error closing tunnel: {e}")
            finally:
                self.process = None
                self.tunnel_url = None
                self.is_healthy = False

# Initialize Cloudflare tunnel manager
tunnel_manager = CloudflareTunnel()

# Global shared processor wrapper
shared_processor = None

def cleanup_after_processing(session: ProcessingSession):
    """Lightweight cleanup after processing - preserves shared models"""
    try:
        logger.info(f"🧹 Starting lightweight cleanup for session {session.process_id}")
        
        # End the session in shared processor
        if shared_processor:
            shared_processor.end_session(session.process_id)
        
        # Clean up temporary files created during this session
        session.cleanup_temp_files()
        
        # Only perform lightweight cleanup (preserve models)
        if shared_processor:
            shared_processor.lightweight_cleanup()
        
        logger.info(f"✅ Lightweight cleanup completed for session {session.process_id}")
        
    except Exception as e:
        logger.error(f"❌ Cleanup error for session {session.process_id}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan with SINGLE processor initialization"""
    global pipeline, processor, cloudflare_tunnel, shared_processor
    
    # Startup - Initialize ONCE
    try:
        logger.info("🚀 Initializing Enhanced AutoDub API v4.0...")
        
        # Setup environment
        Config.setup_cuda_environment()
        
        # Initialize GPU manager if available
        if GPU_UTILS_AVAILABLE:
            try:
                gpu_manager.optimize_memory()
                gpu_info = gpu_manager.monitor_memory()
                logger.info(f"💾 GPU Memory: {gpu_info.get('gpu_memory_free', 'N/A')}GB free")
            except Exception as e:
                logger.warning(f"⚠️ GPU manager initialization failed: {e}")
        
        # Initialize pipeline components ONCE
        if PIPELINE_AVAILABLE:
            try:
                logger.info("🔄 Initializing SINGLE shared processor instance...")
                processor = EnhancedAutoDubProcessor()
                
                # Wrap processor to prevent aggressive cleanup
                shared_processor = SharedProcessorWrapper(processor)
                
                logger.info("✅ Enhanced AutoDub Processor initialized")
                
                if hasattr(processor, 'pipeline'):
                    pipeline = processor.pipeline
                    logger.info("✅ Pipeline components ready")
                else:
                    logger.warning("⚠️ Pipeline not found in processor")
                
                logger.info("🎯 SHARED processor ready for all processing requests (models will persist)")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize processor: {e}")
                processor = None
                pipeline = None
                shared_processor = None
        else:
            logger.warning("⚠️ Pipeline components not available")
        
        # Setup Cloudflare tunnel
        try:
            port = int(os.getenv("API_PORT", "8000"))
            cloudflare_tunnel = tunnel_manager.setup_tunnel(port)
            if cloudflare_tunnel:
                logger.info(f"🌐 Cloudflare tunnel active: {cloudflare_tunnel}")
                logger.info(f"🔗 Public API URL: {cloudflare_tunnel}")
                logger.info(f"📚 Public Docs: {cloudflare_tunnel}/docs")
            else:
                logger.warning("⚠️ Cloudflare tunnel setup failed")
        except Exception as e:
            logger.warning(f"⚠️ Cloudflare tunnel setup error: {e}")
        
        logger.info("✅ Enhanced AutoDub API v4.0 started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {str(e)}")
        raise
    
    yield
    
    # Shutdown - Clean up ONCE
    try:
        logger.info("🛑 Starting API shutdown...")
        
        # Perform full cleanup only during shutdown
        if shared_processor:
            shared_processor.full_cleanup()
        
        if GPU_UTILS_AVAILABLE:
            try:
                gpu_manager.clear_gpu_cache()
            except:
                pass
        
        executor.shutdown(wait=True)
        
        # Close Cloudflare tunnel
        tunnel_manager.close_tunnel()
        
        # Final garbage collection
        gc.collect()
        
        logger.info("🛑 Enhanced AutoDub API v4.0 shutdown complete")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced AutoDub API v4.0",
    description="""
    🎬 Advanced AI-powered video dubbing system with Cloudflare Tunnel
    
    **Core Pipeline Components:**
    - Audio Processing & Enhancement
    - Speaker Diarization & Analysis  
    - Speech Recognition & Transcription
    - Language Translation
    - Text-to-Speech Synthesis
    - Lip Synchronization
    
    **Features:**
    - Real-time processing status
    - Public access via Cloudflare Tunnel
    - Comprehensive error handling
    - PERSISTENT shared models
    - Optimized for Google Colab
    """,
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory=str(Config.OUTPUT_DIR)), name="static")
except Exception as e:
    logger.warning(f"⚠️ Could not mount static files: {e}")

@app.get("/")
async def root():
    """Root endpoint with system information"""
    try:
        system_info = {
            "message": "🎬 Enhanced AutoDub API v4.0",
            "version": "4.0.0",
            "status": "ready" if shared_processor else "initializing",
            "pipeline_available": PIPELINE_AVAILABLE,
            "gpu_utils_available": GPU_UTILS_AVAILABLE,
            "supported_languages": Config.SUPPORTED_LANGUAGES,
            "cloudflare_tunnel": cloudflare_tunnel,
            "docs": "/docs",
            "environment": "Google Colab Optimized",
            "processor_shared": True,
            "models_persistent": True,  # Indicate models persist between requests
            "active_processes": len(processing_status),
            "active_sessions": len(shared_processor.active_sessions) if shared_processor else 0
        }
        
        if GPU_UTILS_AVAILABLE:
            try:
                gpu_info = gpu_manager.monitor_memory()
                system_info["system_info"] = {
                    "device": gpu_manager.get_device(),
                    "gpu_memory_free": gpu_info.get('gpu_memory_free', 'N/A'),
                    "cpu_memory_available": gpu_info.get('cpu_memory_available_gb', 'N/A')
                }
            except Exception as e:
                logger.warning(f"⚠️ GPU info retrieval failed: {e}")
        
        return system_info
    except Exception as e:
        logger.error(f"❌ Root endpoint error: {e}")
        return {"error": "Internal server error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "processor": "ready" if shared_processor else "not_available",
                "pipeline": "ready" if pipeline else "not_available",
                "gpu_utils": "available" if GPU_UTILS_AVAILABLE else "not_available",
                "cloudflare_tunnel": "active" if cloudflare_tunnel else "inactive"
            },
            "active_processes": len(processing_status),
            "active_sessions": len(shared_processor.active_sessions) if shared_processor else 0,
            "processor_shared": True,
            "models_persistent": True,
            "temp_dir_exists": Config.TEMP_DIR.exists(),
            "output_dir_exists": Config.OUTPUT_DIR.exists()
        }
        
        if GPU_UTILS_AVAILABLE:
            try:
                gpu_info = gpu_manager.monitor_memory()
                health_status["system_resources"] = gpu_info
            except Exception as e:
                health_status["gpu_error"] = str(e)
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: Optional[str] = Form(None)
):
    """Upload video for processing"""
    try:
        # Validate file format
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided"
            )
        
        if not file.filename.lower().endswith(tuple(Config.VIDEO_FORMATS)):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unsupported video format",
                    "supported_formats": Config.VIDEO_FORMATS,
                    "received_filename": file.filename
                }
            )
        
        # Validate languages
        if target_language not in Config.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Unsupported target language",
                    "supported_languages": Config.SUPPORTED_LANGUAGES,
                    "received_language": target_language
                }
            )
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        temp_file = Config.TEMP_DIR / f"upload_{upload_id}_{file.filename}"
        
        # Save uploaded file
        try:
            with open(temp_file, "wb") as buffer:
                content = await file.read()
                if not content:
                    raise HTTPException(status_code=400, detail="Empty file uploaded")
                buffer.write(content)
        except Exception as e:
            logger.error(f"❌ File save error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # Validate file size
        file_size = temp_file.stat().st_size
        if file_size > Config.MAX_FILE_SIZE:
            temp_file.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if file_size == 0:
            temp_file.unlink()
            raise HTTPException(status_code=400, detail="Empty file")
        
        logger.info(f"✅ File uploaded: {file.filename} ({file_size} bytes)")
        
        return {
            "upload_id": upload_id,
            "filename": file.filename,
            "temp_path": str(temp_file),
            "target_language": target_language,
            "source_language": source_language,
            "file_size_mb": round(file_size / (1024*1024), 2),
            "message": "File uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    temp_path: str = Form(...),
    target_language: str = Form(...),
    source_language: Optional[str] = Form(None)
):
    """Process uploaded video through the SHARED pipeline with persistent models"""
    
    if not shared_processor:
        raise HTTPException(
            status_code=500, 
            detail="Shared processor not initialized. Check server logs for initialization errors."
        )
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Uploaded file not found: {temp_path}"
        )
    
    # Generate processing ID
    process_id = str(uuid.uuid4())
    
    # Initialize status tracking
    processing_status[process_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Processing queued",
        "start_time": time.time(),
        "target_language": target_language,
        "source_language": source_language,
        "input_file": temp_path,
        "using_shared_processor": True,
        "models_persistent": True,  # Indicate models persist
        "pipeline_stages": {
            "audio_processing": {"status": "pending", "progress": 0},
            "speaker_diarization": {"status": "pending", "progress": 0},
            "speech_recognition": {"status": "pending", "progress": 0},
            "translation": {"status": "pending", "progress": 0},
            "text_to_speech": {"status": "pending", "progress": 0},
            "lip_sync": {"status": "pending", "progress": 0}
        }
    }
    
    # Start background processing with SHARED processor
    background_tasks.add_task(
        process_video_background,
        process_id,
        str(temp_path_obj),
        target_language,
        source_language
    )
    
    logger.info(f"🎬 Started processing: {process_id} (using PERSISTENT SHARED processor)")
    
    return {
        "process_id": process_id,
        "status": "queued",
        "message": "Processing started with persistent shared processor",
        "progress_url": f"/status/{process_id}",
        "estimated_time": "5-15 minutes depending on video length",
        "using_shared_processor": True,
        "models_persistent": True
    }

async def process_video_background(
    process_id: str, 
    video_path: str, 
    target_language: str, 
    source_language: Optional[str]
):
    """Background processing using SHARED processor with persistent models"""
    session = None
    
    try:
        logger.info(f"🎬 Starting background processing for {process_id} (PERSISTENT SHARED PROCESSOR)")
        
        # Create lightweight session for tracking
        session = ProcessingSession(process_id, video_path, target_language, source_language)
        session.add_temp_file(video_path)  # Track input file for cleanup
        
        # Register session with shared processor
        if shared_processor:
            shared_processor.start_session(process_id)
        
        # Update status to processing
        processing_status[process_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Using persistent shared processor - models already loaded",
            "current_stage": "initialization"
        })
        
        # Validate input file still exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Input video file not found: {video_path}")
        
        # CRITICAL: Use SHARED processor with thread safety
        with processor_lock:
            if not shared_processor:
                raise Exception("Shared processor not available")
            
            logger.info(f"🎯 Using PERSISTENT SHARED processor for {process_id} - models remain loaded")
            
            # Update progress
            processing_status[process_id].update({
                "progress": 25,
                "message": "Processing video through persistent shared pipeline...",
                "current_stage": "pipeline_processing"
            })
            
            # Run processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def process_with_persistent_shared_processor():
                try:
                    # Use the SHARED processor directly (models stay loaded)
                    return shared_processor.process_video(
                        video_path,
                        target_language,
                        source_language
                    )
                except Exception as e:
                    logger.error(f"❌ Persistent shared processor error: {e}")
                    return {"success": False, "error": str(e)}
            
            result = await loop.run_in_executor(executor, process_with_persistent_shared_processor)
        
        logger.info(f"📊 Processing result for {process_id}: {result}")
        
        if result and result.get("success"):
            # Extract output file path
            output_video = None
            possible_keys = ["output_video", "dubbed_video", "output_path", "final_video"]
            
            for key in possible_keys:
                if key in result and result[key]:
                    output_video = result[key]
                    break
            
            # Search for output files if not found in result
            if not output_video:
                temp_dir = Path(Config.TEMP_DIR)
                output_dir = Path(Config.OUTPUT_DIR)
                
                for directory in [temp_dir, output_dir]:
                    pattern = f"*{target_language}*.mp4"
                    matching_files = list(directory.glob(pattern))
                    if matching_files:
                        # Get the most recent file
                        output_video = str(max(matching_files, key=lambda p: p.stat().st_mtime))
                        break
            
            if output_video and Path(output_video).exists():
                processing_status[process_id].update({
                    "status": "completed",
                    "progress": 100,
                    "message": "Processing completed successfully with persistent shared processor",
                    "result": result,
                    "output_video": output_video,
                    "output_size_mb": round(Path(output_video).stat().st_size / (1024*1024), 2),
                    "end_time": time.time()
                })
                logger.info(f"✅ Process {process_id} completed. Output: {output_video}")
            else:
                processing_status[process_id].update({
                    "status": "failed",
                    "progress": 0,
                    "message": "Processing completed but output file not found",
                    "error": "Output file not generated",
                    "result": result,
                    "end_time": time.time()
                })
        else:
            error_msg = result.get('error', 'Unknown processing error') if result else 'No result returned'
            processing_status[process_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"Processing failed: {error_msg}",
                "error": error_msg,
                "end_time": time.time()
            })
    
    except Exception as e:
        logger.error(f"❌ Background processing failed for {process_id}: {e}")
        processing_status[process_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Processing failed: {str(e)}",
            "error": str(e),
            "end_time": time.time()
        })
    
    finally:
        # Lightweight cleanup - preserves shared models
        if session:
            cleanup_after_processing(session)

def convert_numpy_safe(obj):
    """
    Recursively converts numpy types and non-serializable types to native Python.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_safe(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_safe(i) for i in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif hasattr(obj, "__dict__"):
        return convert_numpy_safe(vars(obj))
    else:
        return obj

@app.get("/status/{process_id}")
async def get_processing_status(process_id: str):
    """Returns safe, serializable processing status."""
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Process ID not found")

    try:
        status_raw = processing_status[process_id].copy()

        # Add timing info
        if "start_time" in status_raw:
            current_time = time.time()
            start_time = float(status_raw["start_time"])
            end_time = float(status_raw.get("end_time", current_time))
            status_raw["elapsed_time"] = round(end_time - start_time, 2)

            # Estimate remaining
            progress = status_raw.get("progress", 0)
            if status_raw.get("status") == "processing" and progress > 10:
                estimated_total = status_raw["elapsed_time"] * (100 / progress)
                status_raw["estimated_remaining"] = max(0, round(estimated_total - status_raw["elapsed_time"], 2))

        # ✅ Deep conversion of all nested numpy objects
        cleaned = convert_numpy_safe(status_raw)

        return JSONResponse(content=jsonable_encoder(cleaned))

    except Exception as e:
        logger.error(f"❌ Status retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@app.get("/stream/{filename}")
async def stream_file(filename: str):
    """Stream large files in chunks"""
    try:
        # Handle both full paths and just filenames
        if filename.startswith('/'):
            file_path = Path(filename)
        else:
            # Look in multiple locations
            possible_paths = [
                Config.OUTPUT_DIR / filename,
                Config.TEMP_DIR / filename,
                Path(filename)
            ]
            
            file_path = None
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    break
            
            if not file_path:
                raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        def iterfile(file_path: Path, chunk_size: int = 8192):
            try:
                with open(file_path, "rb") as file:
                    while chunk := file.read(chunk_size):
                        yield chunk
            except Exception as e:
                logger.error(f"❌ File streaming error: {e}")
                raise
        
        file_size = file_path.stat().st_size
        
        return StreamingResponse(
            iterfile(file_path),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename={file_path.name}",
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Stream failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stream failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed files with enhanced streaming and error handling"""
    try:
        logger.info(f"📥 Download request for: {filename}")
        
        # Handle both full paths and just filenames
        if filename.startswith('/'):
            file_path = Path(filename)
        else:
            # Look in multiple locations with better pattern matching
            possible_paths = [
                Config.OUTPUT_DIR / filename,
                Config.TEMP_DIR / filename,
                Path(filename)
            ]
            
            # Also try with .mp4 extension if not present
            if not filename.endswith('.mp4'):
                possible_paths.extend([
                    Config.OUTPUT_DIR / f"{filename}.mp4",
                    Config.TEMP_DIR / f"{filename}.mp4"
                ])
            
            file_path = None
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    logger.info(f"✅ Found file at: {path}")
                    break
            
            if not file_path:
                logger.error(f"❌ File not found in any location: {filename}")
                # List available files for debugging
                available_files = []
                for dir_path in [Config.OUTPUT_DIR, Config.TEMP_DIR]:
                    if dir_path.exists():
                        available_files.extend([f.name for f in dir_path.glob("*")])
                
                raise HTTPException(
                    status_code=404, 
                    detail={
                        "error": f"File not found: {filename}",
                        "available_files": available_files[:10],  # Show first 10 files
                        "searched_paths": [str(p) for p in possible_paths]
                    }
                )
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        file_size = file_path.stat().st_size
        logger.info(f"📊 File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        # Check tunnel health before serving large files
        if file_size > 10 * 1024 * 1024:  # 10MB threshold
            tunnel_healthy = tunnel_manager.check_tunnel_health()
            if not tunnel_healthy:
                logger.warning("⚠️ Tunnel unhealthy, attempting reconnection before large file download")
                tunnel_manager.reconnect_tunnel()
        
        # Enhanced streaming function with smaller chunks for better tunnel compatibility
        def iterfile_enhanced():
            chunk_size = 4096  # Smaller chunks for better tunnel stability
            bytes_sent = 0
            
            try:
                with open(file_path, "rb") as file:
                    while True:
                        chunk = file.read(chunk_size)
                        if not chunk:
                            break
                        
                        bytes_sent += len(chunk)
                        
                        # Log progress for large files
                        if file_size > 50 * 1024 * 1024 and bytes_sent % (10 * 1024 * 1024) == 0:
                            progress = (bytes_sent / file_size) * 100
                            logger.info(f"📤 Download progress: {progress:.1f}% ({bytes_sent}/{file_size} bytes)")
                        
                        yield chunk
                        
                        # Small delay for very large files to prevent tunnel timeouts
                        if file_size > 100 * 1024 * 1024:
                            time.sleep(0.001)
                            
            
            except Exception as e:
                logger.error(f"❌ File streaming error: {e}")
                raise HTTPException(status_code=500, detail=f"File streaming failed: {str(e)}")
        
        # Determine content type
        content_type = "video/mp4" if filename.lower().endswith('.mp4') else "application/octet-stream"
        
        # Enhanced headers for better download compatibility
        headers = {
            "Content-Disposition": f"attachment; filename={file_path.name}",
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        
        # Use streaming for all files to ensure consistency
        return StreamingResponse(
            iterfile_enhanced(),
            media_type=content_type,
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/download-direct/{filename}")
async def download_file_direct(filename: str):
    """Direct download fallback when tunnel is unavailable"""
    try:
        logger.info(f"📥 Direct download request for: {filename}")
        
        # Same file resolution logic as download endpoint
        if filename.startswith('/'):
            file_path = Path(filename)
        else:
            possible_paths = [
                Config.OUTPUT_DIR / filename,
                Config.TEMP_DIR / filename,
                Path(filename)
            ]
            
            if not filename.endswith('.mp4'):
                possible_paths.extend([
                    Config.OUTPUT_DIR / f"{filename}.mp4",
                    Config.TEMP_DIR / f"{filename}.mp4"
                ])
            
            file_path = None
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    break
            
            if not file_path:
                raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Use FileResponse for direct downloads (no streaming)
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Direct download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Direct download failed: {str(e)}")

@app.get("/files/{process_id}")
async def list_process_files(process_id: str):
    """List available files for a completed process"""
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Process ID not found")
    
    try:
        status = processing_status[process_id]
        files = []
        
        # Get output video
        output_video = None
        if "output_video" in status:
            output_video = status["output_video"]
        elif "result" in status:
            result = status["result"]
            output_video = result.get("output_video") or result.get("dubbed_video")
        
        # Search for files if not found in status
        if not output_video:
            temp_dir = Path(Config.TEMP_DIR)
            output_dir = Path(Config.OUTPUT_DIR)
            
            for directory in [temp_dir, output_dir]:
                pattern = f"*{status['target_language']}*.mp4"
                matching_files = list(directory.glob(pattern))
                if matching_files:
                    output_video = str(max(matching_files, key=lambda p: p.stat().st_mtime))
                    break
        
        if output_video and Path(output_video).exists():
            file_path = Path(output_video)
            files.append({
                "type": "output_video",
                "filename": file_path.name,
                "full_path": str(file_path),
                "size_mb": round(file_path.stat().st_size / (1024*1024), 2),
                "download_url": f"/download/{file_path.name}",
                "download_direct_url": f"/download-direct/{file_path.name}",  # Fallback option
                "stream_url": f"/stream/{file_path.name}"
            })
        
        return {
            "process_id": process_id,
            "status": status["status"],
            "files": files,
            "total_files": len(files)
        }
    except Exception as e:
        logger.error(f"❌ File listing error: {e}")
        raise HTTPException(status_code=500, detail=f"File listing failed: {str(e)}")

@app.get("/tunnel/info")
async def get_tunnel_info():
    """Get Cloudflare tunnel information"""
    tunnel_url = tunnel_manager.get_tunnel_url()
    if tunnel_url:
        return {
            "tunnel_active": True,
            "public_url": tunnel_url,
            "docs_url": f"{tunnel_url}/docs",
            "tunnel_type": "Cloudflare Tunnel"
        }
    else:
        return {
            "tunnel_active": False,
            "message": "Cloudflare tunnel not active",
            "tunnel_type": "Cloudflare Tunnel"
        }

@app.get("/tunnel/health")
async def check_tunnel_health():
    """Check and report tunnel health status"""
    try:
        tunnel_url = tunnel_manager.get_tunnel_url()
        is_healthy = tunnel_manager.check_tunnel_health()
        
        return {
            "tunnel_active": tunnel_url is not None,
            "tunnel_healthy": is_healthy,
            "tunnel_url": tunnel_url,
            "last_health_check": tunnel_manager.last_health_check,
            "reconnect_attempts": tunnel_manager.reconnect_attempts,
            "process_running": tunnel_manager.process is not None and tunnel_manager.process.poll() is None
        }
    except Exception as e:
        logger.error(f"❌ Tunnel health check error: {e}")
        return {
            "tunnel_active": False,
            "tunnel_healthy": False,
            "error": str(e)
        }

@app.get("/languages")
async def get_supported_languages():
    """Get supported languages and configuration"""
    return {
        "supported_languages": Config.SUPPORTED_LANGUAGES,
        "video_formats": Config.VIDEO_FORMATS,
        "max_file_size_mb": Config.MAX_FILE_SIZE // (1024*1024),
        "features": {
            "auto_detection": True,
            "speaker_diarization": True,
            "emotion_detection": True,
            "cloudflare_tunnel": True,
            "shared_processor": True,
            "persistent_models": True
        }
    }

@app.delete("/cleanup/{process_id}")
async def cleanup_process(process_id: str):
    """Clean up completed process data"""
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Process ID not found")
    
    try:
        status = processing_status[process_id]
        if status["status"] == "processing":
            raise HTTPException(status_code=400, detail="Cannot cleanup active process")
        
        # Remove from status tracking
        del processing_status[process_id]
        
        # Light garbage collection
        gc.collect()
        
        logger.info(f"🗑️ Cleaned up process: {process_id}")
        return {"message": "Process data cleaned up successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/processor/status")
async def get_processor_status():
    """Get shared processor status"""
    try:
        processor_status = {
            "processor_available": shared_processor is not None,
            "pipeline_available": pipeline is not None,
            "shared_processor": True,
            "models_persistent": True,
            "active_processes": len(processing_status),
            "active_sessions": len(shared_processor.active_sessions) if shared_processor else 0,
            "processor_type": type(shared_processor.processor).__name__ if shared_processor else None
        }
        
        if GPU_UTILS_AVAILABLE:
            gpu_info = gpu_manager.monitor_memory()
            processor_status["gpu_info"] = gpu_info
        
        return processor_status
    except Exception as e:
        logger.error(f"❌ Processor status error: {e}")
        raise HTTPException(status_code=500, detail=f"Processor status failed: {str(e)}")

@app.post("/processor/clear-cache")
async def clear_processor_cache():
    """Clear GPU cache without reinitializing models"""
    try:
        logger.info("🧹 Clearing GPU cache (preserving models)...")
        
        if shared_processor:
            shared_processor.lightweight_cleanup()
        
        return {"message": "GPU cache cleared successfully (models preserved)"}
    except Exception as e:
        logger.error(f"❌ Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@app.get("/processes")
async def list_all_processes():
    """List all processing jobs"""
    try:
        processes = []
        for process_id, status in processing_status.items():
            process_info = {
                "process_id": process_id,
                "status": status["status"],
                "progress": status["progress"],
                "target_language": status.get("target_language"),
                "start_time": status.get("start_time"),
                "elapsed_time": None,
                "using_shared_processor": status.get("using_shared_processor", True),
                "models_persistent": status.get("models_persistent", True)
            }
            
            if "start_time" in status:
                current_time = time.time()
                if "end_time" in status:
                    process_info["elapsed_time"] = round(status["end_time"] - status["start_time"], 2)
                else:
                    process_info["elapsed_time"] = round(current_time - status["start_time"], 2)
            
            processes.append(process_info)
        
        return {
            "total_processes": len(processes),
            "processes": processes,
            "shared_processor": True,
            "models_persistent": True,
            "active_sessions": len(shared_processor.active_sessions) if shared_processor else 0
        }
    except Exception as e:
        logger.error(f"❌ Process listing error: {e}")
        raise HTTPException(status_code=500, detail=f"Process listing failed: {str(e)}")

if __name__ == "__main__":
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"🚀 Starting Enhanced AutoDub API v4.0 on {host}:{port}")
    print(f"📚 Documentation will be available at: http://{host}:{port}/docs")
    print(f"🎯 Using PERSISTENT shared processor for efficiency")
    print(f"🔒 Models will remain loaded between requests")
    
    # Run the API
    uvicorn.run(
        "enhanced_autodub_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
