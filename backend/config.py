import os
import torch
from pathlib import Path
from typing import Dict, List, Optional
import getpass
import logging
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Config:
    # Directories
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    TEMP_DIR = BASE_DIR / "temp"
    OUTPUT_DIR = BASE_DIR / "output"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Create directories
    for dir_path in [MODELS_DIR, TEMP_DIR, OUTPUT_DIR, CACHE_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Enhanced CUDA environment setup
    @classmethod
    def setup_cuda_environment(cls):
        """Setup CUDA environment to prevent libcuda.so errors"""
        try:
            cuda_env_vars = {
                'CUDA_VISIBLE_DEVICES': '0',
                'CUDA_LAUNCH_BLOCKING': '1',
                # Disable CUDA cache to prevent memory issues
                'CUDA_CACHE_DISABLE': '1',
                # More conservative memory settings
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128,garbage_collection_threshold:0.8',
                'TF_CPP_MIN_LOG_LEVEL': '2',
                'TOKENIZERS_PARALLELISM': 'false',
                'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                # ALSA fixes
                'ALSA_PCM_CARD': '0',
                'ALSA_PCM_DEVICE': '0',
                'PULSE_RUNTIME_PATH': '/tmp/pulse-runtime',
                'ALSA_CARD': 'default',
                'ALSA_DEVICE': '0'
            }
            
            for key, value in cuda_env_vars.items():
                os.environ[key] = value
            
            # Fix LD_LIBRARY_PATH for libcuda.so
            cuda_paths = [
                '/usr/local/cuda/lib64',
                '/usr/lib/x86_64-linux-gnu',
                '/usr/lib64',
                '/opt/cuda/lib64'
            ]
            
            current_path = os.environ.get('LD_LIBRARY_PATH', '')
            for path in cuda_paths:
                if os.path.exists(path) and path not in current_path:
                    current_path = f"{path}:{current_path}" if current_path else path
            
            os.environ['LD_LIBRARY_PATH'] = current_path
            logger.info("✅ CUDA environment setup completed")
        except Exception as e:
            logger.error(f"⚠️ CUDA environment setup failed: {e}")
    
    # Device configuration with fallback
    @classmethod
    def get_device(cls):
        """Get optimal device with comprehensive fallback"""
        try:
            if torch.cuda.is_available():
                # Test CUDA functionality with smaller tensor
                test_tensor = torch.randn(5, 5).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                logger.info("✅ CUDA is available and working")
                return "cuda"
        except Exception as e:
            logger.warning(f"⚠️ CUDA test failed: {e}")
            logger.info("⚠️ Falling back to CPU")
        
        return "cpu"
    
    # Memory management
    @classmethod
    def clear_gpu_memory(cls):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("✅ GPU memory cache cleared")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clear GPU memory: {e}")
    
    @classmethod
    def get_available_gpu_memory(cls) -> Optional[float]:
        """Get available GPU memory in GB"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_memory = torch.cuda.mem_get_info()[0] / 1024**3
                logger.info(f"📊 Available GPU memory: {free_memory:.2f} GB")
                return free_memory
            return None
        except Exception as e:
            logger.warning(f"⚠️ Failed to get GPU memory info: {e}")
            return None
    
    @classmethod
    def get_appropriate_whisper_model(cls) -> str:
        """Select appropriate Whisper model based on available memory"""
        free_memory = cls.get_available_gpu_memory()
        
        if free_memory is None or cls.DEVICE == "cpu":
            logger.info("⚠️ Using 'base' Whisper model (CPU mode)")
            return "base"
        
        # Model size thresholds in GB
        if free_memory > 10:
            logger.info("✅ Using 'large-v3' Whisper model")
            return "large-v3"
        elif free_memory > 6:
            logger.info("⚠️ Limited memory: Using 'medium' Whisper model")
            return "medium"
        elif free_memory > 3:
            logger.info("⚠️ Low memory: Using 'small' Whisper model")
            return "small"
        else:
            logger.info("⚠️ Very low memory: Using 'base' Whisper model")
            return "base"
    
    # Initialize as None, will be set after class definition
    DEVICE = None
    GPU_MEMORY_FRACTION = 0.6  # More conservative memory usage (reduced from 0.75)
    
    # Model configurations with dynamic fallbacks
    WHISPER_MODEL = None  # Will be set dynamically based on available memory
    PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
    TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
    
    # Enhanced Edge TTS configuration
    EDGE_TTS_VOICES = {
        "en": {
            "male": {
                "default": "en-US-AriaNeural",
                "cheerful": "en-US-AriaNeural",
                "sad": "en-US-GuyNeural",
                "angry": "en-US-ChristopherNeural",
                "excited": "en-US-AriaNeural"
            },
            "female": {
                "default": "en-US-JennyNeural",
                "cheerful": "en-US-JennyNeural",
                "sad": "en-US-AriaNeural",
                "angry": "en-US-SaraNeural",
                "excited": "en-US-JennyNeural"
            }
        },
        "ur": {
            "male": {
                "default": "ur-PK-AsadNeural",
                "cheerful": "ur-PK-AsadNeural",
                "sad": "ur-PK-AsadNeural",
                "angry": "ur-PK-AsadNeural",
                "excited": "ur-PK-AsadNeural"
            },
            "female": {
                "default": "ur-PK-UzmaNeural",
                "cheerful": "ur-PK-UzmaNeural",
                "sad": "ur-PK-UzmaNeural",
                "angry": "ur-PK-UzmaNeural",
                "excited": "ur-PK-UzmaNeural"
            }
        },
        "hi": {
            "male": {
                "default": "hi-IN-MadhurNeural",
                "cheerful": "hi-IN-MadhurNeural",
                "sad": "hi-IN-MadhurNeural",
                "angry": "hi-IN-MadhurNeural",
                "excited": "hi-IN-MadhurNeural"
            },
            "female": {
                "default": "hi-IN-SwaraNeural",
                "cheerful": "hi-IN-SwaraNeural",
                "sad": "hi-IN-SwaraNeural",
                "angry": "hi-IN-SwaraNeural",
                "excited": "hi-IN-SwaraNeural"
            }
        },
        "ar": {
            "male": {
                "default": "ar-SA-HamedNeural",
                "cheerful": "ar-SA-HamedNeural",
                "sad": "ar-SA-HamedNeural",
                "angry": "ar-SA-HamedNeural",
                "excited": "ar-SA-HamedNeural"
            },
            "female": {
                "default": "ar-SA-ZariyahNeural",
                "cheerful": "ar-SA-ZariyahNeural",
                "sad": "ar-SA-ZariyahNeural",
                "angry": "ar-SA-ZariyahNeural",
                "excited": "ar-SA-ZariyahNeural"
            }
        }
    }

    # Hugging Face configuration
    HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN", None)
    HF_TOKEN_FILE = BASE_DIR / ".hf_token"

    # Enhanced audio analysis settings
    NOISE_REDUCTION_ENABLED = True
    BACKGROUND_MUSIC_PRESERVATION = True
    ADVANCED_SPEAKER_ANALYSIS = True
    ENHANCED_GENDER_DETECTION = True
    EMOTION_DETECTION_ENABLED = True

    # Speaker analysis settings
    MIN_SPEAKER_DURATION = 1.0  # Minimum duration for speaker detection
    MAX_SPEAKERS = 8  # Maximum number of speakers to detect
    SPEAKER_CLUSTERING_THRESHOLD = 0.7

    # Audio enhancement settings
    NOISE_REDUCTION_STRENGTH = 0.6  # 0.0 to 1.0
    BACKGROUND_MUSIC_VOLUME = 0.25   # Volume level for background music
    VOCAL_ENHANCEMENT = True
    
    # Audio settings
    SAMPLE_RATE = 22050
    AUDIO_FORMAT = "wav"
    AUDIO_CHANNELS = 1
    
    # Video settings
    VIDEO_MAX_DURATION = 900  # 15 minutes (increased)
    VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]
    TARGET_FPS = 25
    MAX_RESOLUTION = (1280, 720)
    
    # Language settings
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "ur": "Urdu", 
        "hi": "Hindi",
        "ar": "Arabic"
    }
    
    # Processing settings - more conservative
    CHUNK_SIZE = 20  # seconds (reduced from 30)
    OVERLAP = 2  # seconds
    BATCH_SIZE = 2  # Reduced for better memory management (from 4)
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
    
    # Wav2Lip settings
    WAV2LIP_MODEL_PATH = MODELS_DIR / "wav2lip_gan.pth"
    FACE_DETECTION_MODEL_PATH = MODELS_DIR / "face_detection_model.pth"
    
    # Enhanced script format template
    SCRIPT_FORMAT = "[{timestamp}] ({speaker}, {gender}, {emotion}, conf:{confidence:.2f}): {text}"
    
    @classmethod
    def get_edge_voice(cls, language: str, gender: str = "female", emotion: str = "neutral") -> str:
        """Get Edge TTS voice for language, gender, and emotion"""
        voices = cls.EDGE_TTS_VOICES.get(language, cls.EDGE_TTS_VOICES["en"])
        gender_voices = voices.get(gender, voices["female"])
        return gender_voices.get(emotion, gender_voices["default"])

    @classmethod
    def setup_huggingface_token(cls):
        """Setup Hugging Face token for pyannote access"""
        if cls.HF_TOKEN:
            return cls.HF_TOKEN
        
        # Check if token file exists
        if cls.HF_TOKEN_FILE.exists():
            try:
                with open(cls.HF_TOKEN_FILE, 'r') as f:
                    token = f.read().strip()
                    if token:
                        cls.HF_TOKEN = token
                        os.environ["HUGGING_FACE_TOKEN"] = token
                        return token
            except Exception:
                pass
        
        # Prompt for token if not found
        print("\n🔑 Hugging Face Token Required for Advanced Speaker Diarization")
        print("Visit: https://hf.co/settings/tokens to create your token")
        print("Also accept conditions at: https://hf.co/pyannote/speaker-diarization-3.1")
        
        token = getpass.getpass("Enter your Hugging Face token (or press Enter to skip): ").strip()
        
        if token:
            # Save token for future use
            try:
                with open(cls.HF_TOKEN_FILE, 'w') as f:
                    f.write(token)
                cls.HF_TOKEN = token
                os.environ["HUGGING_FACE_TOKEN"] = token
                print("✅ Token saved successfully!")
            except Exception as e:
                print(f"⚠️ Could not save token: {e}")
            
            return token
        
        print("⚠️ Skipping advanced speaker diarization (will use fallback method)")
        return None

    @classmethod
    def get_optimal_batch_size(cls, base_size: int = 4) -> int:
        """Get optimal batch size based on available memory"""
        try:
            if cls.DEVICE == "cuda" and torch.cuda.is_available():
                free_memory = cls.get_available_gpu_memory()
                if free_memory is None:
                    return 1
                
                if free_memory > 10:  # 10GB+
                    return base_size
                elif free_memory > 6:  # 6-10GB
                    return max(1, base_size // 2)
                else:  # <6GB
                    return 1
            return 1
        except Exception:
            return 1
    
    @classmethod
    def initialize(cls):
        """Initialize all configuration settings"""
        try:
            logger.info("🚀 Initializing AutoDub configuration...")
            
            # Setup CUDA environment
            cls.setup_cuda_environment()
            
            # Set device
            cls.DEVICE = cls.get_device()
            logger.info(f"🖥️ Using device: {cls.DEVICE}")
            
            # Set appropriate Whisper model based on available memory
            cls.WHISPER_MODEL = cls.get_appropriate_whisper_model()
            
            # Set optimal batch size
            cls.BATCH_SIZE = cls.get_optimal_batch_size()
            logger.info(f"📦 Using batch size: {cls.BATCH_SIZE}")
            
            # Setup Hugging Face token
            cls.setup_huggingface_token()
            
            logger.info("✅ Configuration initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Configuration initialization failed: {e}")
            return False


# Initialize configuration
Config.initialize()