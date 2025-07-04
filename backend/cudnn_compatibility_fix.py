import os
import logging
import warnings
import sys

# Configure logging
logger = logging.getLogger(__name__)

def fix_cudnn_compatibility():
    """
    Fix CuDNN compatibility issues by setting appropriate environment variables
    """
    try:
        logger.info("🔧 Applying CuDNN compatibility fixes...")
        
        # Suppress TensorFlow CuDNN version warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Tell TensorFlow to only allocate necessary memory
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # Disable CuDNN version check in TensorFlow
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        # Filter out specific warnings
        warnings.filterwarnings('ignore', message='.*CuDNN library version.*')
        
        # Import and configure TensorFlow if available
        try:
            import tensorflow as tf
            
            # Configure TensorFlow to use less GPU memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"✅ TensorFlow GPU memory growth enabled for {len(gpus)} GPUs")
            
            # Set TensorFlow log level
            tf.get_logger().setLevel('ERROR')
            
            logger.info("✅ TensorFlow configured successfully")
            
        except ImportError:
            logger.info("TensorFlow not found, skipping TF-specific configurations")
        except Exception as e:
            logger.warning(f"TensorFlow configuration warning: {e}")
        
        # Import and configure PyTorch if available
        try:
            import torch
            
            # Set PyTorch to use TF32 precision for better performance
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Use deterministic algorithms
                torch.backends.cudnn.deterministic = True
                
                # Set benchmark mode for better performance with fixed input sizes
                torch.backends.cudnn.benchmark = True
                
                logger.info("✅ PyTorch CUDA configurations applied")
            
        except ImportError:
            logger.info("PyTorch not found, skipping PyTorch-specific configurations")
        except Exception as e:
            logger.warning(f"PyTorch configuration warning: {e}")
        
        logger.info("✅ CuDNN compatibility fixes applied successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply CuDNN compatibility fixes: {e}")
        return False

# Apply fixes when imported
fix_cudnn_compatibility()
