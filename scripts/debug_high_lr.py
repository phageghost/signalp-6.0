#!/usr/bin/env python3
"""
Debug script to test if the high learning rate (10) is causing the NaN issue.
"""

import os
import torch
import logging
from signalp6.models import ProteinBertTokenizer, BertSequenceTaggingCRF
from transformers import BertConfig
from signalp6.training_utils import RegionCRFDataset
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the same device logic as training script
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")

def debug_high_learning_rate():
    """Debug if the high learning rate is causing the NaN issue."""
    
    logger.info("Testing with high learning rate (10)...")
    
    try:
        # Load tokenizer and dataset
        tokenizer = ProteinBertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        
        train_data = RegionCRFDataset(
            "data/train_set.fasta",
            sample_weights_path=None,
            tokenizer=tokenizer,
            partition_id=[0, 2],
            kingdom_id=["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=None,
            make_cs_state=False,
            add_global_label=False,
        )
        
        train_loader = DataLoader(
            train_data,
            batch_size=20,
            collate_fn=train_data.collate_fn,
            shuffle=True,
        )
        
        # Create model
        config = BertConfig.from_pretrained("Rostlab/prot_bert")
        config.num_labels = 37
        config.num_global_labels = 6
        config.use_large_crf = True
        config.use_region_labels = True
        config.use_kingdom_id = True
        config.kingdom_embed_size = 32
        
        model = BertSequenceTaggingCRF(config)
        model.to(device)
        model.train()
        
        # Create optimizer with the SAME learning rate as training script
        optimizer = torch.optim.SGD(model.parameters(), lr=10.0, weight_decay=1.2e-6)
        
        logger.info("Model and optimizer created with lr=10.0")
        
        # Test with first batch
        batch = next(iter(train_loader))
        
        if len(batch) == 7:
            data, targets, input_mask, global_targets, cleavage_sites, sample_weights, kingdom_ids = batch
        else:
            data, targets, input_mask, global_targets, sample_weights, kingdom_ids = batch
        
        # Move to device
        data = data.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        input_mask = input_mask.to(device, dtype=torch.bool)
        global_targets = global_targets.to(device, dtype=torch.long)
        sample_weights = sample_weights.to(device, dtype=torch.float32) if sample_weights is not None else None
        kingdom_ids = kingdom_ids.to(device, dtype=torch.long)
        
        logger.info("Data moved to device")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        loss, global_probs, pos_probs, pos_preds = model(
            data,
            global_targets=global_targets,
            targets=None,
            targets_bitmap=targets,
            input_mask=input_mask,
            sample_weights=sample_weights,
            kingdom_ids=kingdom_ids,
        )
        
        logger.info(f"Forward pass successful: loss = {loss}")
        
        # Test backward pass
        logger.info("Testing backward pass...")
        optimizer.zero_grad()
        
        # Check if loss is NaN before backward
        if torch.isnan(loss).any():
            logger.error("❌ Loss is NaN before backward pass!")
            return False
        
        loss.backward()
        logger.info("Backward pass completed")
        
        # Check gradients for NaN/Inf
        logger.info("Checking gradients...")
        total_norm = 0
        nan_grads = 0
        inf_grads = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads += 1
                    logger.error(f"❌ NaN gradients in {name}")
                    
                if torch.isinf(param.grad).any():
                    inf_grads += 1
                    logger.error(f"❌ Inf gradients in {name}")
                
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Check for extremely large gradients
                if param_norm > 1e6:
                    logger.warning(f"⚠️  Large gradients in {name}: {param_norm:.2e}")
        
        total_norm = total_norm ** (1. / 2)
        logger.info(f"Total gradient norm: {total_norm:.6f}")
        logger.info(f"Parameters with NaN gradients: {nan_grads}")
        logger.info(f"Parameters with Inf gradients: {inf_grads}")
        
        if nan_grads > 0 or inf_grads > 0:
            logger.error("❌ Found NaN or Inf gradients!")
            return False
        
        # Test optimizer step with high learning rate
        logger.info("Testing optimizer step with lr=10.0...")
        optimizer.step()
        logger.info("Optimizer step completed")
        
        # Check if model parameters became NaN/Inf
        logger.info("Checking model parameters after optimizer step...")
        nan_params = 0
        inf_params = 0
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params += 1
                logger.error(f"❌ NaN parameters in {name} after optimizer step!")
                
            if torch.isinf(param).any():
                inf_params += 1
                logger.error(f"❌ Inf parameters in {name} after optimizer step!")
        
        logger.info(f"Parameters with NaN after optimizer step: {nan_params}")
        logger.info(f"Parameters with Inf after optimizer step: {inf_params}")
        
        if nan_params > 0 or inf_params > 0:
            logger.error("❌ Model parameters became NaN/Inf after optimizer step!")
            return False
        
        # Test multiple steps to see if instability develops
        logger.info("Testing multiple optimizer steps...")
        for step in range(5):
            # Forward pass
            loss, global_probs, pos_probs, pos_preds = model(
                data,
                global_targets=global_targets,
                targets=None,
                targets_bitmap=targets,
                input_mask=input_mask,
                sample_weights=sample_weights,
                kingdom_ids=kingdom_ids,
            )
            
            if torch.isnan(loss).any():
                logger.error(f"❌ Loss became NaN at step {step}!")
                return False
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        logger.error(f"❌ NaN gradients at step {step} in {name}!")
                        return False
                    if torch.isinf(param.grad).any():
                        logger.error(f"❌ Inf gradients at step {step} in {name}!")
                        return False
            
            # Optimizer step
            optimizer.step()
            
            # Check parameters
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    logger.error(f"❌ NaN parameters at step {step} in {name}!")
                    return False
                if torch.isinf(param).any():
                    logger.error(f"❌ Inf parameters at step {step} in {name}!")
                    return False
            
            logger.info(f"Step {step} completed successfully, loss: {loss.item():.6f}")
        
        logger.info("✅ All high learning rate tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ High learning rate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_high_learning_rate()
    if success:
        logger.info("🎉 High learning rate test PASSED! Learning rate is not the issue.")
    else:
        logger.error("💥 High learning rate test FAILED! Learning rate might be causing the issue.")
