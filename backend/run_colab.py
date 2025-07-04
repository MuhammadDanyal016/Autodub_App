"""
Enhanced runner script for Google Colab with comprehensive error handling
"""

import os
import sys
import logging
from pathlib import Path
import subprocess
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_setup_status():
    """Check if setup has been completed"""
    try:
        # Check for key model files
        model_files = [
            Path("models/wav2lip_gan.pth"),
            Path("models/face_detection_model.pth")
        ]
        
        # Check for key Python packages
        import importlib
        key_packages = ["torch", "whisper", "edge_tts", "cv2", "librosa"]
        
        models_exist = all(f.exists() for f in model_files)
        packages_exist = True
        
        for package in key_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                packages_exist = False
                break
        
        return models_exist and packages_exist
        
    except Exception as e:
        logger.warning(f"Setup status check failed: {e}")
        return False

def run_setup():
    """Run the setup script"""
    try:
        logger.info("📦 Running Enhanced AutoDub setup...")
        result = subprocess.run([sys.executable, "setup_colab.py"], check=True)
        logger.info("✅ Setup completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Setup failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Setup error: {e}")
        return False

def start_api_server():
    """Start the Enhanced AutoDub API server"""
    try:
        logger.info("🌐 Starting Enhanced AutoDub API server...")
        logger.info("📍 API will be available at: http://localhost:8000")
        logger.info("📚 Documentation at: http://localhost:8000/docs")
        logger.info("🎬 Features enabled:")
        logger.info("   ✅ Enhanced gender detection")
        logger.info("   ✅ Emotion-aware TTS")
        logger.info("   ✅ GPU optimization")
        logger.info("   ✅ Advanced speaker diarization")
        logger.info("   ✅ Background music preservation")
        
        # Run the main application
        subprocess.run([sys.executable, "main.py"], check=True)
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Server failed: {e}")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

def setup_and_run():
    """Setup and run Enhanced AutoDub with comprehensive error handling"""
    start_time = time.time()
    
    logger.info("🚀 Starting Enhanced AutoDub for Google Colab...")
    logger.info("🔧 Version: 3.0 - Complete optimization and error fixes")
    
    try:
        # Check if setup is needed
        if not check_setup_status():
            logger.info("📦 Setup required - running initial setup...")
            if not run_setup():
                logger.error("❌ Setup failed - cannot continue")
                return False
        else:
            logger.info("✅ Setup already completed - starting server...")
        
        # Start the API server
        start_api_server()
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ Total runtime: {elapsed_time:.2f} seconds")
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
        return True
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_and_run()
    sys.exit(0 if success else 1)
