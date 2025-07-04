"""
ENHANCED Direct Processing Script v4.0 - Complete system with all improvements
"""

import sys
import os
import logging
import argparse
from pathlib import Path
import time

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_autodub.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import config first
from config import Config

# Setup environment
Config.setup_cuda_environment()
Config.setup_huggingface_token()

from core.enhanced_autodub_processor import EnhancedAutoDubProcessor
from utils.gpu_utils import gpu_manager

def main():
    """ENHANCED Direct processing with comprehensive features"""
    parser = argparse.ArgumentParser(
        description="ENHANCED AutoDub v4.0 - Direct Video Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ENHANCED FEATURES v4.0:
  ✅ ML-based gender detection (audio + video)
  ✅ Pretrained emotion recognition (Wav2Vec2 + DeepFace)
  ✅ Enhanced speaker diarization with embeddings
  ✅ Unique voice allocation per speaker
  ✅ Background music separation and preservation
  ✅ Advanced text cleaning and TTS processing
  ✅ Comprehensive error handling and fallbacks

EXAMPLES:
  python direct_process_enhanced.py video.mp4 ur
  python direct_process_enhanced.py video.mp4 hi --source en
  python direct_process_enhanced.py video.mp4 ar --output ./results
        """
    )
    
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('target_language', help='Target language code (e.g., ur, hi, ar)')
    parser.add_argument('--source', '--source-language', dest='source_language',
                       help='Source language code (auto-detect if not specified)')
    parser.add_argument('--output', '--output-dir', dest='output_dir',
                       help='Output directory (default: ./output)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Verbose logging enabled")
    
    try:
        # Display enhanced startup banner
        print("=" * 100)
        print("🚀 ENHANCED AUTODUB v4.0 - DIRECT PROCESSING")
        print("=" * 100)
        print("🔧 ENHANCED FEATURES:")
        print("   ✅ ML-based gender detection (audio + video analysis)")
        print("   ✅ Pretrained emotion recognition (Wav2Vec2 + DeepFace)")
        print("   ✅ Advanced speaker diarization with embeddings")
        print("   ✅ Unique voice allocation per speaker")
        print("   ✅ Background music separation and preservation")
        print("   ✅ Enhanced text cleaning and TTS processing")
        print("   ✅ Comprehensive error handling and fallbacks")
        print("=" * 100)
        print()
        
        # Validate inputs
        logger.info("🔍 Validating inputs...")
        
        video_path = Path(args.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if args.target_language not in Config.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported target language: {args.target_language}")
        
        if args.source_language and args.source_language not in Config.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {args.source_language}")
        
        # Display processing info
        logger.info(f"📁 Input video: {video_path}")
        logger.info(f"🌍 Target language: {args.target_language}")
        logger.info(f"🌍 Source language: {args.source_language or 'auto-detect'}")
        logger.info(f"📁 Output directory: {args.output_dir or 'default'}")
        
        # Initialize system
        logger.info("🚀 Initializing ENHANCED AutoDub system...")
        
        # Setup CUDA environment
        Config.setup_cuda_environment()
        
        # Monitor initial system state
        initial_memory = gpu_manager.monitor_memory()
        logger.info(f"💾 System status:")
        logger.info(f"   Device: {gpu_manager.get_device()}")
        logger.info(f"   CUDA available: {gpu_manager.is_gpu_available()}")
        logger.info(f"   GPU memory: {initial_memory.get('gpu_memory_free', 'N/A')}GB free")
        logger.info(f"   CPU memory: {initial_memory.get('cpu_memory_available_gb', 'N/A')}GB available")
        
        # Initialize enhanced processor
        processor = EnhancedAutoDubProcessor()
        
        # Start processing
        logger.info("🎬 Starting ENHANCED video processing...")
        start_time = time.time()
        
        result = processor.process_video(
            str(video_path),
            args.target_language,
            args.source_language,
            args.output_dir
        )
        
        processing_time = time.time() - start_time
        
        # Display results
        print("\n" + "=" * 100)
        if result['success']:
            print("🎉 ENHANCED PROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 100)
            print(f"⏱️  Total time: {processing_time:.2f} seconds")
            print(f"📹 Output video: {result.get('output_video', 'N/A')}")
            print(f"📝 Script file: {result.get('script_file', 'N/A')}")
            print(f"👥 Speakers found: {result.get('speakers_found', 'N/A')}")
            print(f"📊 Segments processed: {result.get('segments_processed', 'N/A')}")
            print(f"🎯 Success rate: {result.get('success_rate', 'N/A')}%")
            
            # Display enhanced features used
            features = result.get('features_used', {})
            if features:
                print("\n🔧 ENHANCED FEATURES USED:")
                for feature, enabled in features.items():
                    status = "✅" if enabled else "❌"
                    feature_name = feature.replace('_', ' ').title()
                    print(f"   {status} {feature_name}")
            
            # Display memory usage
            memory_usage = result.get('memory_usage', {})
            if memory_usage:
                print(f"\n💾 MEMORY USAGE:")
                initial = memory_usage.get('initial', {})
                final = memory_usage.get('final', {})
                print(f"   Initial GPU: {initial.get('gpu_memory_free', 'N/A')}GB")
                print(f"   Final GPU: {final.get('gpu_memory_free', 'N/A')}GB")
            
            print("=" * 100)
            
        else:
            print("❌ ENHANCED PROCESSING FAILED!")
            print("=" * 100)
            print(f"⏱️  Failed after: {processing_time:.2f} seconds")
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            # Display processing stages for debugging
            stages = result.get('processing_stages', {})
            if stages:
                print(f"\n📊 PROCESSING STAGES:")
                for stage, data in stages.items():
                    status = data.get('status', 'unknown')
                    duration = data.get('duration', 0)
                    print(f"   {stage}: {status} ({duration:.2f}s)")
            
            print("=" * 100)
            return 1
        
        # Cleanup
        processor.cleanup()
        gpu_manager.clear_gpu_cache()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Processing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ ENHANCED processing failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
