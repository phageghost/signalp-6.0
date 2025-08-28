#!/usr/bin/env python3
"""
Debug script to investigate the CRF constraint mask issue.
The model has NaN values in crf._constraint_mask which is causing the training to fail.
"""

import os
import sys
import torch
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signalp6.models import BertSequenceTaggingCRF
from transformers import BertConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def investigate_crf_constraint_mask():
    """Investigate the CRF constraint mask issue."""
    
    logger.info("Investigating CRF constraint mask issue...")
    
    # Create a minimal config similar to what's used in training
    config = BertConfig.from_pretrained("Rostlab/prot_bert")
    
    # Set the same attributes that are set in training
    config.num_labels = 37
    config.num_global_labels = 6
    config.lm_output_dropout = 0.1
    config.lm_output_position_dropout = 0.1
    config.crf_scaling_factor = 1.0
    config.use_large_crf = True
    config.use_region_labels = True
    config.use_kingdom_id = False
    config.kingdom_embed_size = 0
    
    # Set CRF constraints (this is what's causing the issue)
    config.use_region_labels = True
    
    logger.info(f"Config created with:")
    logger.info(f"  - num_labels: {config.num_labels}")
    logger.info(f"  - num_global_labels: {config.num_global_labels}")
    logger.info(f"  - use_region_labels: {config.use_region_labels}")
    logger.info(f"  - use_large_crf: {config.use_large_crf}")
    
    # Create the model
    logger.info("Creating model...")
    model = BertSequenceTaggingCRF(config)
    
    # Check if the model has a CRF layer
    if hasattr(model, 'crf'):
        logger.info("Model has CRF layer")
        logger.info(f"CRF type: {type(model.crf)}")
        
        # Check all attributes of the CRF layer
        for attr_name in dir(model.crf):
            if not attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(model.crf, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    logger.info(f"CRF attribute {attr_name}: {type(attr_value)}, shape: {attr_value.shape}")
                    if torch.isnan(attr_value).any():
                        logger.error(f"  - Contains NaN values!")
                        logger.error(f"  - Min: {attr_value.min()}, Max: {attr_value.max()}")
                        logger.error(f"  - Sample values: {attr_value[:5, :5]}")
                    elif torch.isinf(attr_value).any():
                        logger.error(f"  - Contains Inf values!")
                        logger.error(f"  - Min: {attr_value.min()}, Max: {attr_value.max()}")
                    else:
                        logger.info(f"  - Values OK: min={attr_value.min()}, max={attr_value.max()}")
            except Exception as e:
                logger.warning(f"Could not access CRF attribute {attr_name}: {e}")
        
        # Specifically check the constraint mask
        if hasattr(model.crf, '_constraint_mask'):
            constraint_mask = model.crf._constraint_mask
            logger.info(f"Constraint mask shape: {constraint_mask.shape}")
            logger.info(f"Constraint mask dtype: {constraint_mask.dtype}")
            logger.info(f"Constraint mask device: {constraint_mask.device}")
            
            if torch.isnan(constraint_mask).any():
                logger.error("Constraint mask contains NaN values!")
                logger.error(f"NaN count: {torch.isnan(constraint_mask).sum()}")
                logger.error(f"Total elements: {constraint_mask.numel()}")
                
                # Check where the NaNs are
                nan_positions = torch.where(torch.isnan(constraint_mask))
                logger.error(f"NaN positions (first 10): {list(zip(nan_positions[0][:10], nan_positions[1][:10]))}")
                
                # Check if it's all NaN
                if torch.isnan(constraint_mask).all():
                    logger.error("ALL values in constraint mask are NaN!")
                else:
                    # Check some non-NaN values
                    non_nan_mask = ~torch.isnan(constraint_mask)
                    if non_nan_mask.any():
                        non_nan_values = constraint_mask[non_nan_mask]
                        logger.info(f"Non-NaN values: min={non_nan_values.min()}, max={non_nan_values.max()}")
                        logger.info(f"Non-NaN count: {non_nan_values.numel()}")
            else:
                logger.info("Constraint mask is OK (no NaN values)")
                
                # Check the actual values
                logger.info(f"Constraint mask values: min={constraint_mask.min()}, max={constraint_mask.max()}")
                logger.info(f"Unique values: {torch.unique(constraint_mask)}")
                
                # Show a sample of the mask
                logger.info(f"Sample of constraint mask (first 10x10):")
                logger.info(constraint_mask[:10, :10])
        else:
            logger.warning("CRF layer does not have _constraint_mask attribute")
            
        # Check other CRF-related attributes
        for attr_name in ['start_transitions', 'end_transitions', 'transitions']:
            if hasattr(model.crf, attr_name):
                attr_value = getattr(model.crf, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    logger.info(f"CRF {attr_name}: shape={attr_value.shape}, dtype={attr_value.dtype}")
                    if torch.isnan(attr_value).any():
                        logger.error(f"  - Contains NaN values!")
                    elif torch.isinf(attr_value).any():
                        logger.error(f"  - Contains Inf values!")
                    else:
                        logger.info(f"  - Values OK: min={attr_value.min()}, max={attr_value.max()}")
    else:
        logger.warning("Model does not have CRF layer")
    
    # Check the model's forward method signature
    logger.info("Model forward method signature:")
    import inspect
    sig = inspect.signature(model.forward)
    logger.info(f"  {sig}")
    
    # Check if the model has any other problematic parameters
    logger.info("Checking all model parameters for NaN/Inf...")
    has_problematic = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"Parameter '{name}' contains NaN values!")
            has_problematic = True
        if torch.isinf(param).any():
            logger.error(f"Parameter '{name}' contains Inf values!")
            has_problematic = True
    
    if not has_problematic:
        logger.info("All model parameters are OK")
    
    # Check buffers
    logger.info("Checking all model buffers for NaN/Inf...")
    has_problematic_buffer = False
    for name, buffer in model.named_buffers():
        if torch.isnan(buffer).any():
            logger.error(f"Buffer '{name}' contains NaN values!")
            has_problematic_buffer = True
        if torch.isinf(buffer).any():
            logger.error(f"Buffer '{name}' contains Inf values!")
            has_problematic_buffer = True
    
    if not has_problematic_buffer:
        logger.info("All model buffers are OK")
    
    return model

def test_model_forward_pass(model):
    """Test a simple forward pass to see where the NaN comes from."""
    logger.info("Testing model forward pass...")
    
    # Create dummy input
    batch_size = 2
    seq_len = 70
    
    input_ids = torch.randint(0, 30, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    global_targets = torch.randint(0, 6, (batch_size,))
    targets = torch.randint(0, 37, (batch_size, seq_len))
    
    logger.info(f"Input shapes:")
    logger.info(f"  - input_ids: {input_ids.shape}")
    logger.info(f"  - attention_mask: {attention_mask.shape}")
    logger.info(f"  - global_targets: {global_targets.shape}")
    logger.info(f"  - targets: {targets.shape}")
    
    try:
        # Try forward pass
        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(
                input_ids,
                global_targets=global_targets,
                targets=targets,
                input_mask=attention_mask
            )
            
        logger.info("Forward pass successful!")
        logger.info(f"  - Loss: {loss.shape}, value: {loss.item()}")
        logger.info(f"  - Global probs: {global_probs.shape}")
        logger.info(f"  - Position probs: {pos_probs.shape}")
        logger.info(f"  - Position predictions: {pos_preds.shape}")
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Starting CRF constraint mask investigation...")
    
    try:
        model = investigate_crf_constraint_mask()
        test_model_forward_pass(model)
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Investigation complete.")
