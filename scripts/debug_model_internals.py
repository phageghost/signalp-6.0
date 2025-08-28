#!/usr/bin/env python3
"""
Debug script to trace through the model's internal computations.
This will help identify exactly where the NaN is coming from.
"""

import os
import torch
import numpy as np
import logging
from signalp6.models import ProteinBertTokenizer, BertSequenceTaggingCRF
from transformers import BertConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_model_forward_step_by_step():
    """Debug the model forward pass step by step."""
    
    logger.info("Setting up model for debugging...")
    
    # Create configuration
    config = BertConfig.from_pretrained("Rostlab/prot_bert")
    config.num_labels = 37
    config.num_global_labels = 6
    config.use_large_crf = True
    config.use_region_labels = True
    config.use_kingdom_id = True
    config.kingdom_embed_size = 32  # Add missing attribute
    config.crf_reduction_mode = "mean"
    config.crf_ignore_index = -100
    config.crf_constraint_type = "BIO"
    
    # Initialize model
    model = BertSequenceTaggingCRF(config)
    model.eval()
    
    # Create realistic input data (matching your training data)
    batch_size = 4
    sequence_length = 70
    
    # Input data with realistic token IDs (0-24 range)
    input_ids = torch.randint(0, 25, (batch_size, sequence_length), dtype=torch.long)
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    token_type_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    
    # Targets with -1 for padding (matching your data)
    # Note: The model trims [CLS] and [SEP] tokens, so targets should be 68 positions
    targets = torch.randint(0, 2, (batch_size, 68, 37), dtype=torch.long)
    targets[targets == 0] = -1  # Set some to -1 for padding
    
    # Global targets
    global_targets = torch.randint(0, 3, (batch_size,), dtype=torch.long)
    
    # Kingdom IDs
    kingdom_ids = torch.randint(0, 3, (batch_size,), dtype=torch.long)
    
    logger.info("Input data created:")
    logger.info(f"  input_ids: shape={input_ids.shape}, range=[{input_ids.min()}, {input_ids.max()}]")
    logger.info(f"  targets: shape={targets.shape}, range=[{targets.min()}, {targets.max()}]")
    logger.info(f"  global_targets: shape={global_targets.shape}, range=[{global_targets.min()}, {global_targets.max()}]")
    logger.info(f"  kingdom_ids: shape={kingdom_ids.shape}, range=[{kingdom_ids.min()}, {kingdom_ids.max()}]")
    
    # Test 1: Just the BERT encoder
    logger.info("\n=== Testing BERT Encoder ===")
    try:
        with torch.no_grad():
            # Get BERT outputs
            bert_outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            last_hidden_state = bert_outputs.last_hidden_state
            logger.info(f"BERT output shape: {last_hidden_state.shape}")
            logger.info(f"BERT output range: [{last_hidden_state.min():.6f}, {last_hidden_state.max():.6f}]")
            
            if torch.isnan(last_hidden_state).any():
                logger.error("NaN detected in BERT output!")
            else:
                logger.info("BERT encoder working correctly")
                
    except Exception as e:
        logger.error(f"BERT encoder failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: BERT + classification heads
    logger.info("\n=== Testing Classification Heads ===")
    try:
        with torch.no_grad():
            # Get BERT outputs
            bert_outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            last_hidden_state = bert_outputs.last_hidden_state
            
            # Test global classification head
            if hasattr(model, 'global_classifier'):
                global_logits = model.global_classifier(last_hidden_state[:, 0, :])  # [CLS] token
                logger.info(f"Global logits shape: {global_logits.shape}")
                logger.info(f"Global logits range: [{global_logits.min():.6f}, {global_logits.max():.6f}]")
                
                if torch.isnan(global_logits).any():
                    logger.error("NaN detected in global logits!")
                else:
                    logger.info("Global classifier working correctly")
            
            # Test position classification head
            if hasattr(model, 'position_classifier'):
                position_logits = model.position_classifier(last_hidden_state)
                logger.info(f"Position logits shape: {position_logits.shape}")
                logger.info(f"Position logits range: [{position_logits.min():.6f}, {position_logits.max():.6f}]")
                
                if torch.isnan(position_logits).any():
                    logger.error("NaN detected in position logits!")
                else:
                    logger.info("Position classifier working correctly")
                    
    except Exception as e:
        logger.error(f"Classification heads failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Full forward pass with targets (using targets_bitmap for region labels)
    logger.info("\n=== Testing Full Forward Pass ===")
    try:
        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(
                input_ids,
                global_targets=global_targets,
                targets=None,  # Don't specify targets when using targets_bitmap
                targets_bitmap=targets,
                input_mask=attention_mask,
                sample_weights=None,
                kingdom_ids=kingdom_ids,
            )
            
            logger.info("Full forward pass completed!")
            logger.info(f"Loss: {loss}")
            logger.info(f"Global probs shape: {global_probs.shape}")
            logger.info(f"Position probs shape: {pos_probs.shape}")
            logger.info(f"Position predictions shape: {pos_preds.shape}")
            
            # Check for NaNs
            if torch.isnan(loss).any():
                logger.error("NaN detected in loss!")
            if torch.isnan(global_probs).any():
                logger.error("NaN detected in global probabilities!")
            if torch.isnan(pos_probs).any():
                logger.error("NaN detected in position probabilities!")
                
    except Exception as e:
        logger.error(f"Full forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Check model parameters
    logger.info("\n=== Checking Model Parameters ===")
    try:
        total_params = 0
        nan_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if torch.isnan(param).any():
                nan_params += param.numel()
                logger.error(f"NaN detected in parameter: {name}")
                logger.error(f"  Shape: {param.shape}")
                logger.error(f"  Range: [{param.min()}, {param.max()}]")
        
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Parameters with NaN: {nan_params}")
        
        if nan_params == 0:
            logger.info("All model parameters are valid")
            
    except Exception as e:
        logger.error(f"Parameter check failed: {e}")
        import traceback
        traceback.print_exc()

def debug_crf_specifically():
    """Debug the CRF component specifically."""
    
    logger.info("\n=== Testing CRF Component ===")
    
    try:
        # Create a minimal CRF test
        batch_size = 2
        seq_len = 10
        num_tags = 37
        
        # Create logits
        logits = torch.randn(batch_size, seq_len, num_tags) * 0.1
        
        # Create targets (with -1 for padding)
        targets = torch.randint(0, num_tags, (batch_size, seq_len))
        targets[0, 5:] = -1  # Add some padding
        
        logger.info(f"CRF test - Logits shape: {logits.shape}")
        logger.info(f"CRF test - Targets shape: {targets.shape}")
        logger.info(f"CRF test - Targets range: [{targets.min()}, {targets.max()}]")
        
        # Test if the issue is with -1 targets
        if -1 in targets:
            logger.info("Testing with -1 targets (padding)")
            # Replace -1 with 0 for CRF computation
            targets_for_crf = targets.clone()
            targets_for_crf[targets_for_crf == -1] = 0
            
            logger.info(f"Targets for CRF range: [{targets_for_crf.min()}, {targets_for_crf.max()}]")
            
    except Exception as e:
        logger.error(f"CRF test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Starting model internal debugging...")
    
    # Test 1: Step-by-step model debugging
    debug_model_forward_step_by_step()
    
    # Test 2: CRF-specific debugging
    debug_crf_specifically()
    
    logger.info("Model internal debugging complete.")
