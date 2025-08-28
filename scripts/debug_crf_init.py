#!/usr/bin/env python3
"""
Debug script to investigate CRF initialization and see where NaN values are coming from.
"""

import os
import sys
import torch
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signalp6.models.multi_tag_crf import CRF

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_crf_initialization():
    """Debug CRF initialization to see where NaN values come from."""
    
    logger.info("Debugging CRF initialization...")
    
    # Test on CPU first
    logger.info("Testing CRF initialization on CPU...")
    crf_cpu = CRF(num_tags=37, batch_first=True)
    
    logger.info(f"CRF transitions on CPU: shape={crf_cpu.transitions.shape}")
    logger.info(f"CRF transitions on CPU: has NaN={torch.isnan(crf_cpu.transitions).any()}")
    logger.info(f"CRF transitions on CPU: has Inf={torch.isinf(crf_cpu.transitions).any()}")
    if not torch.isnan(crf_cpu.transitions).any() and not torch.isinf(crf_cpu.transitions).any():
        logger.info(f"CRF transitions on CPU: min={crf_cpu.transitions.min():.6f}, max={crf_cpu.transitions.max():.6f}")
    
    logger.info(f"CRF constraint mask on CPU: shape={crf_cpu._constraint_mask.shape}")
    logger.info(f"CRF constraint mask on CPU: has NaN={torch.isnan(crf_cpu._constraint_mask).any()}")
    logger.info(f"CRF constraint mask on CPU: has Inf={torch.isinf(crf_cpu._constraint_mask).any()}")
    if not torch.isnan(crf_cpu._constraint_mask).any() and not torch.isinf(crf_cpu._constraint_mask).any():
        logger.info(f"CRF constraint mask on CPU: min={crf_cpu._constraint_mask.min():.6f}, max={crf_cpu._constraint_mask.max():.6f}")
    
    # Test on MPS if available
    if torch.backends.mps.is_available():
        logger.info("Testing CRF initialization on MPS...")
        device = torch.device('mps')
        
        # Create CRF on CPU first, then move to MPS
        crf_mps = CRF(num_tags=37, batch_first=True)
        crf_mps = crf_mps.to(device)
        
        logger.info(f"CRF transitions on MPS: shape={crf_mps.transitions.shape}")
        logger.info(f"CRF transitions on MPS: has NaN={torch.isnan(crf_mps.transitions).any()}")
        logger.info(f"CRF transitions on MPS: has Inf={torch.isinf(crf_mps.transitions).any()}")
        if not torch.isnan(crf_mps.transitions).any() and not torch.isinf(crf_mps.transitions).any():
            logger.info(f"CRF transitions on MPS: min={crf_mps.transitions.min():.6f}, max={crf_mps.transitions.max():.6f}")
        
        logger.info(f"CRF constraint mask on MPS: shape={crf_mps._constraint_mask.shape}")
        logger.info(f"CRF constraint mask on MPS: has NaN={torch.isnan(crf_mps._constraint_mask).any()}")
        logger.info(f"CRF constraint mask on MPS: has Inf={torch.isinf(crf_mps._constraint_mask).any()}")
        if not torch.isnan(crf_mps._constraint_mask).any() and not torch.isinf(crf_mps._constraint_mask).any():
            logger.info(f"CRF constraint mask on MPS: min={crf_mps._constraint_mask.min():.6f}, max={crf_mps._constraint_mask.max():.6f}")
        
        # Check if the issue is in the to() method
        logger.info("Testing CRF to() method...")
        crf_mps2 = CRF(num_tags=37, batch_first=True)
        logger.info(f"CRF transitions before to(): has NaN={torch.isnan(crf_mps2.transitions).any()}")
        crf_mps2 = crf_mps2.to(device)
        logger.info(f"CRF transitions after to(): has NaN={torch.isnan(crf_mps2.transitions).any()}")
        
    else:
        logger.info("MPS not available, skipping MPS test")
    
    logger.info("CRF initialization debugging complete.")

if __name__ == "__main__":
    debug_crf_initialization()
