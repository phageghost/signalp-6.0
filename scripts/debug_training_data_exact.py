#!/usr/bin/env python3
"""
Debug script that exactly replicates the training data loading process.
This will help identify the exact source of the NaN issue.
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

def debug_training_data_exact():
    """Debug the exact training data loading process."""
    
    logger.info("Loading training data with exact same process as training script...")
    
    try:
        # Load tokenizer and dataset exactly as in training
        tokenizer = ProteinBertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        
        # Create dataset with exact same parameters as training
        train_data = RegionCRFDataset(
            "data/train_set.fasta",
            sample_weights_path=None,
            tokenizer=tokenizer,
            partition_id=[0, 2],  # Same as training: train_ids = [0, 1, 2], remove val_id=1
            kingdom_id=["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=None,
            make_cs_state=False,
            add_global_label=False,
        )
        
        logger.info(f"Dataset loaded with {len(train_data)} samples")
        
        # Create data loader exactly as in training
        train_loader = DataLoader(
            train_data,
            batch_size=20,  # Same as your training
            collate_fn=train_data.collate_fn,
            shuffle=True,
        )
        
        logger.info(f"Data loader created with {len(train_loader)} batches")
        
        # Create model exactly as in training
        config = BertConfig.from_pretrained("Rostlab/prot_bert")
        config.num_labels = 37
        config.num_global_labels = 6
        config.use_large_crf = True
        config.use_region_labels = True
        config.use_kingdom_id = True
        config.kingdom_embed_size = 32
        
        model = BertSequenceTaggingCRF(config)
        model.to(device)
        model.train()  # Set to training mode
        
        logger.info(f"Model created and moved to {device}")
        
        # Test with the first few batches exactly as in training
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Only test first 3 batches
                break
                
            logger.info(f"\n--- Testing Batch {i} ---")
            
            # Unpack batch exactly as in training
            if len(batch) == 7:  # With cleavage sites
                data, targets, input_mask, global_targets, cleavage_sites, sample_weights, kingdom_ids = batch
            else:  # Without cleavage sites
                data, targets, input_mask, global_targets, sample_weights, kingdom_ids = batch
                cleavage_sites = None
            
            logger.info(f"Batch unpacked:")
            logger.info(f"  data: {data.shape}, dtype: {data.dtype}, device: {data.device}")
            logger.info(f"  targets: {targets.shape}, dtype: {targets.dtype}, device: {targets.device}")
            logger.info(f"  input_mask: {input_mask.shape}, dtype: {input_mask.dtype}, device: {input_mask.device}")
            logger.info(f"  global_targets: {global_targets.shape}, dtype: {global_targets.dtype}, device: {global_targets.device}")
            logger.info(f"  kingdom_ids: {kingdom_ids.shape}, dtype: {kingdom_ids.dtype}, device: {kingdom_ids.device}")
            
            # Move to device exactly as in training
            data = data.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)
            input_mask = input_mask.to(device, dtype=torch.bool)
            global_targets = global_targets.to(device, dtype=torch.long)
            sample_weights = sample_weights.to(device, dtype=torch.float32) if sample_weights is not None else None
            kingdom_ids = kingdom_ids.to(device, dtype=torch.long)
            
            logger.info(f"Data moved to device:")
            logger.info(f"  data: {data.device}, dtype: {data.dtype}")
            logger.info(f"  targets: {targets.device}, dtype: {targets.dtype}")
            logger.info(f"  input_mask: {input_mask.device}, dtype: {input_mask.dtype}")
            logger.info(f"  global_targets: {global_targets.device}, dtype: {global_targets.dtype}")
            logger.info(f"  kingdom_ids: {kingdom_ids.device}, dtype: {kingdom_ids.dtype}")
            
            # Check for NaN/Inf in inputs
            if torch.isnan(data).any():
                logger.error(f"❌ NaN detected in input data")
                return False
            if torch.isnan(targets).any():
                logger.error(f"❌ NaN detected in targets")
                return False
            if torch.isnan(global_targets).any():
                logger.error(f"❌ NaN detected in global targets")
                return False
            if torch.isnan(kingdom_ids).any():
                logger.error(f"❌ NaN detected in kingdom ids")
                return False
            
            # Check data ranges
            logger.info(f"Data ranges:")
            logger.info(f"  data: [{data.min()}, {data.max()}]")
            logger.info(f"  targets: [{targets.min()}, {targets.max()}]")
            logger.info(f"  global_targets: [{global_targets.min()}, {global_targets.max()}]")
            logger.info(f"  kingdom_ids: [{kingdom_ids.min()}, {kingdom_ids.max()}]")
            
            # Test model forward pass exactly as in training
            logger.info("Testing model forward pass...")
            
            loss, global_probs, pos_probs, pos_preds = model(
                data,
                global_targets=global_targets,
                targets=None,  # Using targets_bitmap for region labels
                targets_bitmap=targets,
                input_mask=input_mask,
                sample_weights=sample_weights,
                kingdom_ids=kingdom_ids,
            )
            
            logger.info("Forward pass completed!")
            logger.info(f"Loss: {loss}")
            logger.info(f"Loss shape: {loss.shape}")
            logger.info(f"Global probs shape: {global_probs.shape}")
            logger.info(f"Position probs shape: {pos_probs.shape}")
            logger.info(f"Position predictions shape: {pos_preds.shape}")
            
            # Check for NaN/Inf in outputs
            if torch.isnan(loss).any():
                logger.error(f"❌ NaN detected in loss!")
                return False
            if torch.isnan(global_probs).any():
                logger.error(f"❌ NaN detected in global probabilities!")
                return False
            if torch.isnan(pos_probs).any():
                logger.error(f"❌ NaN detected in position probabilities!")
                return False
                
            logger.info(f"✅ Batch {i} completed successfully")
            
        logger.info("🎉 All test batches completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_training_data_exact()
    if success:
        logger.info("🎉 Training data debug PASSED! The issue is not in the data loading.")
    else:
        logger.error("💥 Training data debug FAILED! Found the source of the NaN issue.")
