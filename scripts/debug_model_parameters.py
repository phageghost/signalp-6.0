#!/usr/bin/env python3
"""
Debug script to check if the model parameters contain NaN or Inf values.
This will help identify if the issue is in the model initialization.
"""

import os
import torch
import logging
from signalp6.models import ProteinBertTokenizer, BertSequenceTaggingCRF
from transformers import BertConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_parameters():
    """Check if the model parameters contain NaN or Inf values."""
    
    logger.info("Checking model parameters for NaN/Inf values...")
    
    try:
        # Create the same model configuration as in training
        config = BertConfig.from_pretrained("Rostlab/prot_bert")
        config.num_labels = 37
        config.num_global_labels = 6
        config.use_large_crf = True
        config.use_region_labels = True
        config.use_kingdom_id = True
        config.kingdom_embed_size = 32
        
        logger.info("Creating model...")
        model = BertSequenceTaggingCRF(config)
        
        # Check all parameters for NaN/Inf
        total_params = 0
        nan_params = 0
        inf_params = 0
        zero_params = 0
        
        logger.info("Checking model parameters...")
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            if torch.isnan(param).any():
                nan_params += param.numel()
                logger.error(f"❌ NaN found in {name}: {torch.isnan(param).sum()} values")
                
            if torch.isinf(param).any():
                inf_params += param.numel()
                logger.error(f"❌ Inf found in {name}: {torch.isinf(param).sum()} values")
                
            if (param == 0).all():
                zero_params += param.numel()
                logger.warning(f"⚠️  All zeros in {name}")
            
            # Check parameter statistics
            param_min = param.min().item()
            param_max = param.max().item()
            param_mean = param.mean().item()
            param_std = param.std().item()
            
            if abs(param_mean) > 1e3 or abs(param_std) > 1e3:
                logger.warning(f"⚠️  Large values in {name}: mean={param_mean:.2e}, std={param_std:.2e}")
            
            logger.debug(f"  {name}: shape={param.shape}, range=[{param_min:.6f}, {param_max:.6f}], mean={param_mean:.6f}, std={param_std:.6f}")
        
        logger.info(f"Parameter summary:")
        logger.info(f"  Total parameters: {total_params}")
        logger.info(f"  NaN parameters: {nan_params}")
        logger.info(f"  Inf parameters: {inf_params}")
        logger.info(f"  Zero parameters: {zero_params}")
        
        if nan_params > 0 or inf_params > 0:
            logger.error("❌ Model contains NaN or Inf parameters!")
            return False
        else:
            logger.info("✅ All model parameters are valid")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to check model parameters: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_forward_pass():
    """Check if the model can perform a forward pass without NaN."""
    
    logger.info("\n--- Testing Model Forward Pass ---")
    
    try:
        # Create model
        config = BertConfig.from_pretrained("Rostlab/prot_bert")
        config.num_labels = 37
        config.num_global_labels = 6
        config.use_large_crf = True
        config.use_region_labels = True
        config.use_kingdom_id = True
        config.kingdom_embed_size = 32
        
        model = BertSequenceTaggingCRF(config)
        model.eval()
        
        # Create dummy input data
        batch_size = 2
        seq_len = 72
        
        # Create realistic input data
        input_ids = torch.randint(0, 25, (batch_size, seq_len), dtype=torch.long)
        input_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        global_targets = torch.randint(0, 6, (batch_size,), dtype=torch.long)
        kingdom_ids = torch.randint(0, 4, (batch_size,), dtype=torch.long)
        
        # Create targets with -1 for padding (matching your data)
        targets = torch.zeros((batch_size, 70, 37), dtype=torch.float)
        targets[:, :, 0] = 1  # Set some default values
        targets[targets == 0] = -1  # Set padding to -1
        
        logger.info(f"Input shapes:")
        logger.info(f"  input_ids: {input_ids.shape}")
        logger.info(f"  input_mask: {input_mask.shape}")
        logger.info(f"  global_targets: {global_targets.shape}")
        logger.info(f"  kingdom_ids: {kingdom_ids.shape}")
        logger.info(f"  targets: {targets.shape}")
        
        # Test forward pass
        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(
                input_ids,
                global_targets=global_targets,
                targets=None,
                targets_bitmap=targets,
                input_mask=input_mask,
                sample_weights=None,
                kingdom_ids=kingdom_ids,
            )
            
            logger.info("Forward pass successful!")
            logger.info(f"Loss: {loss}")
            logger.info(f"Global probs shape: {global_probs.shape}")
            logger.info(f"Position probs shape: {pos_probs.shape}")
            logger.info(f"Position predictions shape: {pos_preds.shape}")
            
            # Check for NaNs
            if torch.isnan(loss).any():
                logger.error("❌ NaN detected in loss!")
                return False
            if torch.isnan(global_probs).any():
                logger.error("❌ NaN detected in global probabilities!")
                return False
            if torch.isnan(pos_probs).any():
                logger.error("❌ NaN detected in position probabilities!")
                return False
                
            logger.info("✅ No NaNs detected in forward pass!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_initialization():
    """Check if there are issues with model initialization."""
    
    logger.info("\n--- Checking Model Initialization ---")
    
    try:
        # Check if the model can be created multiple times
        for i in range(3):
            logger.info(f"Creating model {i+1}...")
            
            config = BertConfig.from_pretrained("Rostlab/prot_bert")
            config.num_labels = 37
            config.num_global_labels = 6
            config.use_large_crf = True
            config.use_region_labels = True
            config.use_kingdom_id = True
            config.kingdom_embed_size = 32
            
            model = BertSequenceTaggingCRF(config)
            
            # Check if model has the expected attributes
            if not hasattr(model, 'bert'):
                logger.error(f"❌ Model {i+1} missing 'bert' attribute")
                return False
            if not hasattr(model, 'kingdom_embedding'):
                logger.error(f"❌ Model {i+1} missing 'kingdom_embedding' attribute")
                return False
            if not hasattr(model, 'crf'):
                logger.error(f"❌ Model {i+1} missing 'crf' attribute")
                return False
                
            logger.info(f"✅ Model {i+1} created successfully")
            
            # Check parameter count
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model {i+1} has {param_count} parameters")
            
            del model  # Clean up
            
        logger.info("✅ All models created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting model parameter debugging...")
    
    # Check model parameters
    params_ok = check_model_parameters()
    
    # Check model initialization
    init_ok = check_model_initialization()
    
    # Check forward pass
    forward_ok = check_model_forward_pass()
    
    # Summary
    logger.info("\n--- Summary ---")
    if params_ok and init_ok and forward_ok:
        logger.info("🎉 All checks passed! Model should work correctly.")
    else:
        logger.error("💥 Some checks failed! Model has issues.")
        
        if not params_ok:
            logger.error("  - Model parameters contain NaN/Inf values")
        if not init_ok:
            logger.error("  - Model initialization failed")
        if not forward_ok:
            logger.error("  - Model forward pass failed")
