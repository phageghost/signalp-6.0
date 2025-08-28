#!/usr/bin/env python3
"""
Debug script to check environmental factors that might be causing the NaN issue.
"""

import os
import torch
import logging
import platform
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check environmental factors that might affect model training."""
    
    logger.info("=== Environment Check ===")
    
    # Check Python and PyTorch versions
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check platform
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    
    # Check PyTorch build info
    logger.info(f"PyTorch build: {torch.version.git_version}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # Check MPS availability
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        logger.info(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Check device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Selected device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"Selected device: MPS")
    else:
        device = torch.device("cpu")
        logger.info(f"Selected device: CPU")
    
    # Check tensor operations
    logger.info("\n=== Tensor Operation Tests ===")
    
    # Test basic tensor operations
    try:
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        z = torch.mm(x, y)
        logger.info(f"✅ Matrix multiplication on {device}: {z.shape}")
        
        # Test log operation
        log_z = torch.log(torch.softmax(z, dim=1))
        logger.info(f"✅ Log operation on {device}: {log_z.shape}")
        
        # Test exponential operation
        exp_z = torch.exp(z)
        logger.info(f"✅ Exponential operation on {device}: {exp_z.shape}")
        
    except Exception as e:
        logger.error(f"❌ Tensor operations failed on {device}: {e}")
        return False
    
    # Check for NaN/Inf in results
    if torch.isnan(z).any():
        logger.error(f"❌ NaN detected in matrix multiplication result")
        return False
    if torch.isinf(z).any():
        logger.error(f"❌ Inf detected in matrix multiplication result")
        return False
    
    if torch.isnan(log_z).any():
        logger.error(f"❌ NaN detected in log operation result")
        return False
    if torch.isinf(log_z).any():
        logger.error(f"❌ Inf detected in log operation result")
        return False
    
    if torch.isnan(exp_z).any():
        logger.error(f"❌ NaN detected in exponential operation result")
        return False
    if torch.isinf(exp_z).any():
        logger.error(f"❌ Inf detected in exponential operation result")
        return False
    
    logger.info("✅ All tensor operations completed successfully")
    
    # Check memory usage
    logger.info("\n=== Memory Check ===")
    if device.type == "cuda":
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    elif device.type == "mps":
        # MPS doesn't have memory info methods
        logger.info("MPS device - memory info not available")
    else:
        logger.info("CPU device - memory info not available")
    
    # Check random seed behavior
    logger.info("\n=== Random Seed Test ===")
    
    # Set a fixed seed
    torch.manual_seed(42)
    
    # Generate some random numbers
    rand1 = torch.randn(5, 5, device=device)
    rand2 = torch.randn(5, 5, device=device)
    
    # Reset seed and generate again
    torch.manual_seed(42)
    rand1_again = torch.randn(5, 5, device=device)
    rand2_again = torch.randn(5, 5, device=device)
    
    # Check if they're the same
    if torch.allclose(rand1, rand1_again) and torch.allclose(rand2, rand2_again):
        logger.info("✅ Random seed behavior is deterministic")
    else:
        logger.warning("⚠️  Random seed behavior is not deterministic")
    
    # Check for any NaN/Inf in random numbers
    if torch.isnan(rand1).any() or torch.isnan(rand2).any():
        logger.error("❌ NaN detected in random number generation")
        return False
    if torch.isinf(rand1).any() or torch.isinf(rand2).any():
        logger.error("❌ Inf detected in random number generation")
        return False
    
    logger.info("✅ Random number generation is working correctly")
    
    # Check environment variables
    logger.info("\n=== Environment Variables ===")
    relevant_vars = [
        'KMP_DUPLICATE_LIB_OK',
        'PYTORCH_ENABLE_MPS_FALLBACK',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'CUDA_VISIBLE_DEVICES',
    ]
    
    for var in relevant_vars:
        value = os.environ.get(var)
        if value is not None:
            logger.info(f"{var}: {value}")
        else:
            logger.info(f"{var}: Not set")
    
    logger.info("✅ Environment check completed successfully")
    return True

if __name__ == "__main__":
    success = check_environment()
    if success:
        logger.info("\n🎉 Environment check PASSED! No obvious issues found.")
    else:
        logger.error("\n💥 Environment check FAILED! Found potential issues.")
