"""
Enhanced GPU utilities with libcuda.so error fixes and optimal resource management
"""

import torch
import psutil
import gc
import logging
import os
import subprocess
from typing import Dict, Optional, Any
import threading
import time

logger = logging.getLogger(__name__)

class GPUManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.device = self._get_optimal_device()
        self.gpu_available = torch.cuda.is_available()
        self.memory_reserved = 0
        self._setup_cuda_environment()
        
    def _setup_cuda_environment(self):
        """Setup CUDA environment to prevent libcuda.so errors"""
        try:
            # Set CUDA environment variables
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['CUDA_CACHE_DISABLE'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
            
            # Fix libcuda.so path issues
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
            
            # Initialize CUDA context properly
            if torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.empty_cache()
                
            logger.info("✅ CUDA environment configured successfully")
            
        except Exception as e:
            logger.warning(f"CUDA environment setup warning: {e}")
    
    def _get_optimal_device(self) -> str:
        """Get optimal device with comprehensive fallback"""
        try:
            if torch.cuda.is_available():
                # Test CUDA functionality thoroughly
                device_count = torch.cuda.device_count()
                logger.info(f"Found {device_count} CUDA device(s)")
                
                for i in range(device_count):
                    try:
                        torch.cuda.set_device(i)
                        test_tensor = torch.randn(100, 100, device=f'cuda:{i}')
                        _ = test_tensor @ test_tensor.T
                        del test_tensor
                        torch.cuda.empty_cache()
                        logger.info(f"✅ CUDA device {i} working properly")
                        return "cuda"
                    except Exception as e:
                        logger.warning(f"CUDA device {i} test failed: {e}")
                        continue
                        
            logger.info("Using CPU device")
            return "cpu"
            
        except Exception as e:
            logger.warning(f"Device detection failed: {e}, using CPU")
            return "cpu"
    
    def get_device(self) -> str:
        """Get current device"""
        return self.device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and working"""
        return self.gpu_available and self.device == "cuda"
    
    def get_gpu_info(self) -> Dict:
        """Get comprehensive GPU information"""
        if not self.is_gpu_available():
            return {"gpu_available": False, "device": "cpu"}
        
        try:
            with self._lock:
                current_device = torch.cuda.current_device()
                gpu_props = torch.cuda.get_device_properties(current_device)
                memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
                memory_total = gpu_props.total_memory / 1024**3
                
                return {
                    "gpu_available": True,
                    "device": f"cuda:{current_device}",
                    "gpu_name": gpu_props.name,
                    "gpu_memory_total": round(memory_total, 2),
                    "gpu_memory_allocated": round(memory_allocated, 2),
                    "gpu_memory_reserved": round(memory_reserved, 2),
                    "gpu_memory_free": round(memory_total - memory_reserved, 2),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                    "multiprocessor_count": gpu_props.multi_processor_count
                }
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return {"gpu_available": False, "error": str(e)}
    
    def clear_gpu_cache(self):
        """Comprehensive GPU cache clearing"""
        try:
            with self._lock:
                if self.is_gpu_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
                    
                # Force garbage collection
                gc.collect()
                
                # Additional cleanup for specific libraries
                try:
                    import transformers
                    if hasattr(transformers, 'utils'):
                        transformers.utils.cleanup()
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"GPU cache clear warning: {e}")
    
    def optimize_memory(self):
        """Comprehensive memory optimization"""
        try:
            self.clear_gpu_cache()
            
            if self.is_gpu_available():
                # Set memory fraction conservatively
                try:
                    torch.cuda.set_per_process_memory_fraction(0.75)
                    logger.info("✅ GPU memory fraction set to 75%")
                except Exception as e:
                    logger.warning(f"Memory fraction setting failed: {e}")
                
                # Enable memory mapping for large models
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
        except Exception as e:
            logger.warning(f"Memory optimization warning: {e}")
    
    def get_optimal_batch_size(self, base_batch_size: int = 8) -> int:
        """Get optimal batch size based on available memory"""
        if not self.is_gpu_available():
            return 1
        
        try:
            gpu_info = self.get_gpu_info()
            free_memory = gpu_info.get("gpu_memory_free", 0)
            
            if free_memory > 10:  # 10GB+
                return base_batch_size
            elif free_memory > 6:  # 6-10GB
                return max(1, base_batch_size // 2)
            elif free_memory > 3:  # 3-6GB
                return max(1, base_batch_size // 4)
            else:  # <3GB
                return 1
                
        except Exception as e:
            logger.warning(f"Batch size calculation failed: {e}")
            return 1
    
    def monitor_memory(self) -> Dict:
        """Comprehensive system memory monitoring"""
        try:
            cpu_memory = psutil.virtual_memory()
            
            result = {
                "timestamp": time.time(),
                "cpu_memory_total_gb": round(cpu_memory.total / 1024**3, 2),
                "cpu_memory_available_gb": round(cpu_memory.available / 1024**3, 2),
                "cpu_memory_used_gb": round(cpu_memory.used / 1024**3, 2),
                "cpu_memory_percent": cpu_memory.percent,
                "cpu_count": psutil.cpu_count(),
                "cpu_usage_percent": psutil.cpu_percent(interval=1)
            }
            
            if self.is_gpu_available():
                gpu_info = self.get_gpu_info()
                result.update(gpu_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            return {"error": str(e)}
    
    def cleanup_model(self, model: Any):
        """Safely cleanup a model and free memory"""
        try:
            if model is not None:
                # Move to CPU first if on GPU
                if hasattr(model, 'cpu'):
                    model.cpu()
                
                # Delete the model
                del model
                
                # Clear caches
                self.clear_gpu_cache()
                
        except Exception as e:
            logger.warning(f"Model cleanup warning: {e}")
    
    def safe_model_load(self, load_func, *args, **kwargs):
        """Safely load a model with memory management"""
        try:
            # Clear memory before loading
            self.clear_gpu_cache()
            
            # Load model
            model = load_func(*args, **kwargs)
            
            # Move to appropriate device
            if hasattr(model, 'to') and self.is_gpu_available():
                model = model.to(self.device)
            
            return model
            
        except Exception as e:
            logger.error(f"Safe model load failed: {e}")
            raise
    
    def get_device_context(self):
        """Get device context manager"""
        if self.is_gpu_available():
            return torch.cuda.device(self.device)
        else:
            return torch.device('cpu')

# Global GPU manager instance
gpu_manager = GPUManager()
