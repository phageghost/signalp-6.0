#!/usr/bin/env python3
"""
Debug script to test the specific scenario where a pretrained model is loaded
and then CRF constraints are added, which might be causing the NaN issue.
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

def test_pretrained_model_with_constraints():
    """Test loading a pretrained model and then adding CRF constraints."""
    
    logger.info("Testing pretrained model loading with CRF constraints...")
    
    # Step 1: Create initial config and load pretrained model
    logger.info("Step 1: Loading pretrained model...")
    config = BertConfig.from_pretrained("Rostlab/prot_bert")
    
    # Set basic attributes
    config.num_labels = 37
    config.num_global_labels = 6
    config.lm_output_dropout = 0.1
    config.lm_output_position_dropout = 0.1
    config.crf_scaling_factor = 1.0
    config.use_large_crf = True
    
    logger.info("Initial config created")
    
    # Load the model from pretrained weights
    model = BertSequenceTaggingCRF.from_pretrained("Rostlab/prot_bert", config=config)
    logger.info("Model loaded from pretrained weights")
    
    # Check CRF state before adding constraints
    logger.info("CRF state BEFORE adding constraints:")
    if hasattr(model, 'crf') and hasattr(model.crf, '_constraint_mask'):
        constraint_mask = model.crf._constraint_mask
        logger.info(f"  - Constraint mask shape: {constraint_mask.shape}")
        logger.info(f"  - Constraint mask has NaN: {torch.isnan(constraint_mask).any()}")
        if not torch.isnan(constraint_mask).any():
            logger.info(f"  - Constraint mask values: min={constraint_mask.min()}, max={constraint_mask.max()}")
            logger.info(f"  - Unique values: {torch.unique(constraint_mask)}")
    
    # Step 2: Add CRF constraints (this is what happens in training)
    logger.info("Step 2: Adding CRF constraints...")
    
    # Set region labels flag
    config.use_region_labels = True
    logger.info("Set use_region_labels = True")
    
    # Add CRF constraints (this is the problematic part)
    if hasattr(config, 'use_region_labels') and config.use_region_labels:
        logger.info("Setting up CRF constraints...")
        
        # This is the exact constraint setup from the training script
        allowed_transitions = [
            # NO_SP
            (0, 0), (0, 1), (1, 1), (1, 2), (1, 0), (2, 1), (2, 2),
            # SPI
            (3, 3), (3, 4), (4, 4), (4, 5), (5, 5), (5, 8), (8, 8),
            (8, 7), (7, 7), (7, 6), (6, 6), (6, 7), (7, 8),
            # SPII
            (9, 9), (9, 10), (10, 10), (10, 11), (11, 11), (11, 12),
            (12, 15), (15, 15), (15, 14), (14, 14), (14, 13), (13, 13),
            (13, 14), (14, 15),
            # TAT
            (16, 16), (16, 17), (17, 17), (17, 16), (16, 18), (18, 18),
            (18, 19), (19, 19), (19, 22), (22, 22), (22, 21), (21, 21),
            (21, 20), (20, 20), (20, 21), (21, 22),
            # TATLIPO
            (23, 23), (23, 24), (24, 24), (24, 23), (23, 25), (25, 25),
            (25, 26), (26, 26), (26, 27), (27, 30), (30, 30), (30, 29),
            (29, 29), (29, 28), (28, 28), (28, 29), (29, 30),
            # PILIN
            (31, 31), (31, 32), (32, 32), (32, 33), (33, 33), (33, 36),
            (36, 36), (36, 35), (35, 35), (35, 34), (34, 34), (34, 35), (35, 36),
        ]
        
        allowed_starts = [0, 2, 3, 9, 16, 23, 31]
        allowed_ends = [0, 1, 2, 13, 14, 15, 20, 21, 22, 28, 29, 30, 34, 35, 36]
        
        config.allowed_crf_transitions = allowed_transitions
        config.allowed_crf_starts = allowed_starts
        config.allowed_crf_ends = allowed_ends
        
        logger.info(f"Added {len(allowed_transitions)} allowed transitions")
        logger.info(f"Added {len(allowed_starts)} allowed starts")
        logger.info(f"Added {len(allowed_ends)} allowed ends")
        
        # Now try to apply these constraints to the model
        logger.info("Applying constraints to model...")
        
        # Check if the model has a method to apply constraints
        if hasattr(model.crf, 'set_constraints'):
            logger.info("Model has set_constraints method, calling it...")
            try:
                model.crf.set_constraints(allowed_transitions, allowed_starts, allowed_ends)
                logger.info("Constraints applied successfully")
            except Exception as e:
                logger.error(f"Failed to apply constraints: {e}")
        else:
            logger.warning("Model does not have set_constraints method")
            
            # Try to manually set the constraint mask
            logger.info("Attempting to manually set constraint mask...")
            try:
                # Create a new constraint mask based on allowed transitions
                constraint_mask = torch.zeros(37, 37, dtype=torch.float32)
                
                # Set allowed transitions to 1, others to -inf
                for from_state, to_state in allowed_transitions:
                    constraint_mask[from_state, to_state] = 1.0
                
                # Set disallowed transitions to -inf
                constraint_mask[constraint_mask == 0] = float('-inf')
                
                logger.info(f"Created constraint mask: shape={constraint_mask.shape}")
                logger.info(f"Constraint mask has NaN: {torch.isnan(constraint_mask).any()}")
                logger.info(f"Constraint mask has Inf: {torch.isinf(constraint_mask).any()}")
                logger.info(f"Constraint mask values: min={constraint_mask.min()}, max={constraint_mask.max()}")
                
                # Try to assign it to the model
                if hasattr(model.crf, '_constraint_mask'):
                    model.crf._constraint_mask.data = constraint_mask
                    logger.info("Manually assigned constraint mask to model")
                else:
                    logger.warning("Model CRF does not have _constraint_mask attribute")
                    
            except Exception as e:
                logger.error(f"Failed to manually set constraint mask: {e}")
    
    # Step 3: Check CRF state after adding constraints
    logger.info("CRF state AFTER adding constraints:")
    if hasattr(model, 'crf') and hasattr(model.crf, '_constraint_mask'):
        constraint_mask = model.crf._constraint_mask
        logger.info(f"  - Constraint mask shape: {constraint_mask.shape}")
        logger.info(f"  - Constraint mask has NaN: {torch.isnan(constraint_mask).any()}")
        logger.info(f"  - Constraint mask has Inf: {torch.isinf(constraint_mask).any()}")
        if not torch.isnan(constraint_mask).any() and not torch.isinf(constraint_mask).any():
            logger.info(f"  - Constraint mask values: min={constraint_mask.min()}, max={constraint_mask.max()}")
            logger.info(f"  - Unique values: {torch.unique(constraint_mask)}")
        else:
            logger.error("  - Constraint mask has problematic values!")
            if torch.isnan(constraint_mask).any():
                logger.error(f"    - NaN count: {torch.isnan(constraint_mask).sum()}")
            if torch.isinf(constraint_mask).any():
                logger.error(f"    - Inf count: {torch.isinf(constraint_mask).sum()}")
    
    # Step 4: Check all model parameters for NaN/Inf
    logger.info("Step 4: Checking all model parameters for NaN/Inf...")
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

def test_forward_pass_with_constraints(model):
    """Test forward pass after adding constraints."""
    logger.info("Testing forward pass with constraints...")
    
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
    logger.info("Starting CRF constraint loading investigation...")
    
    try:
        model = test_pretrained_model_with_constraints()
        test_forward_pass_with_constraints(model)
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Investigation complete.")
