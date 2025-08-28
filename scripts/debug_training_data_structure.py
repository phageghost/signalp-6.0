#!/usr/bin/env python3
"""
Debug script to examine the actual training data structure.
This will help identify the exact shapes and values causing the NaN issue.
"""

import os
import torch
import numpy as np
import logging
from signalp6.models import ProteinBertTokenizer, BertSequenceTaggingCRF
from transformers import BertConfig
from signalp6.training_utils import RegionCRFDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def examine_training_data():
    """Examine the actual training data structure."""
    
    logger.info("Loading training data...")
    
    try:
        # Load the same dataset that's used in training
        tokenizer = ProteinBertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        
        # Create a small dataset (same as in training)
        dataset = RegionCRFDataset(
            "data/train_set.fasta",
            sample_weights_path=None,
            tokenizer=tokenizer,
            partition_id=[0],  # Just check partition 0
            kingdom_id=["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=None,
            make_cs_state=False,
            add_global_label=False,
        )
        
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Examine a few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            logger.info(f"\n--- Sample {i} ---")
            
            # Unpack the sample
            if len(sample) == 7:  # With cleavage sites
                data, targets, input_mask, global_targets, cleavage_sites, sample_weights, kingdom_ids = sample
            else:  # Without cleavage sites
                data, targets, input_mask, global_targets, sample_weights, kingdom_ids = sample
                cleavage_sites = None
            
            logger.info(f"  data shape: {data.shape}, range: [{data.min()}, {data.max()}]")
            logger.info(f"  targets shape: {targets.shape}, range: [{targets.min()}, {targets.max()}]")
            logger.info(f"  input_mask shape: {input_mask.shape}, range: [{input_mask.min()}, {input_mask.max()}]")
            
            # Handle different types for global_targets and kingdom_ids
            if hasattr(global_targets, 'shape'):
                logger.info(f"  global_targets shape: {global_targets.shape}, range: [{global_targets.min()}, {global_targets.max()}]")
            else:
                logger.info(f"  global_targets: {global_targets} (type: {type(global_targets)})")
                
            if hasattr(kingdom_ids, 'shape'):
                logger.info(f"  kingdom_ids shape: {kingdom_ids.shape}, range: [{kingdom_ids.min()}, {kingdom_ids.max()}]")
            else:
                logger.info(f"  kingdom_ids: {kingdom_ids} (type: {type(kingdom_ids)})")
            
            if cleavage_sites is not None:
                if hasattr(cleavage_sites, 'shape'):
                    logger.info(f"  cleavage_sites shape: {cleavage_sites.shape}, range: [{cleavage_sites.min()}, {cleavage_sites.max()}]")
                else:
                    logger.info(f"  cleavage_sites: {cleavage_sites} (type: {type(cleavage_sites)})")
            
            # Check for special values
            if -1 in targets:
                logger.info(f"  Found -1 in targets at positions: {np.where(targets == -1)}")
            
            # Check sequence lengths
            if input_mask is not None:
                if hasattr(input_mask, 'sum'):
                    seq_lens = input_mask.sum()
                    logger.info(f"  Sequence length: {seq_lens}")
                else:
                    logger.info(f"  input_mask sum: {input_mask}")
        
        # Test the collate function
        logger.info("\n--- Testing Collate Function ---")
        try:
            # Create a small batch
            batch_samples = [dataset[i] for i in range(min(2, len(dataset)))]
            collated = dataset.collate_fn(batch_samples)
            
            logger.info(f"Collated batch has {len(collated)} elements")
            for j, item in enumerate(collated):
                if isinstance(item, torch.Tensor):
                    logger.info(f"  Item {j}: shape={item.shape}, dtype={item.dtype}, range=[{item.min()}, {item.max()}]")
                else:
                    logger.info(f"  Item {j}: type={type(item)}, value={item}")
                    
        except Exception as e:
            logger.error(f"Collate function failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        import traceback
        traceback.print_exc()

def test_model_with_real_data():
    """Test the model with the actual training data structure."""
    
    logger.info("\n--- Testing Model with Real Data ---")
    
    try:
        # Load tokenizer and dataset
        tokenizer = ProteinBertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        
        dataset = RegionCRFDataset(
            "data/train_set.fasta",
            sample_weights_path=None,
            tokenizer=tokenizer,
            partition_id=[0],
            kingdom_id=["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=None,
            make_cs_state=False,
            add_global_label=False,
        )
        
        # Get a sample
        sample = dataset[0]
        if len(sample) == 7:
            data, targets, input_mask, global_targets, cleavage_sites, sample_weights, kingdom_ids = sample
        else:
            data, targets, input_mask, global_targets, sample_weights, kingdom_ids = sample
        
        logger.info(f"Sample data shape: {data.shape}")
        logger.info(f"Sample targets shape: {targets.shape}")
        logger.info(f"Sample input_mask shape: {input_mask.shape}")
        logger.info(f"Sample global_targets: {global_targets}")
        logger.info(f"Sample kingdom_ids: {kingdom_ids}")
        
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
        
        # Test forward pass
        with torch.no_grad():
            # Convert numpy arrays to tensors and add batch dimension
            data_tensor = torch.unsqueeze(torch.tensor(data, dtype=torch.long), 0)
            targets_tensor = torch.unsqueeze(torch.tensor(targets, dtype=torch.float), 0)
            input_mask_tensor = torch.unsqueeze(torch.tensor(input_mask, dtype=torch.bool), 0)
            global_targets_tensor = torch.tensor([global_targets], dtype=torch.long)
            kingdom_ids_tensor = torch.tensor([kingdom_ids], dtype=torch.long)
            
            logger.info(f"Input tensor shapes:")
            logger.info(f"  data: {data_tensor.shape}")
            logger.info(f"  targets: {targets_tensor.shape}")
            logger.info(f"  input_mask: {input_mask_tensor.shape}")
            logger.info(f"  global_targets: {global_targets_tensor.shape}")
            logger.info(f"  kingdom_ids: {kingdom_ids_tensor.shape}")
            
            # Test forward pass
            loss, global_probs, pos_probs, pos_preds = model(
                data_tensor,
                global_targets=global_targets_tensor,
                targets=None,
                targets_bitmap=targets_tensor,
                input_mask=input_mask_tensor,
                sample_weights=None,
                kingdom_ids=kingdom_ids_tensor,
            )
            
            logger.info("Forward pass successful!")
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
        logger.error(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Starting training data structure examination...")
    
    # Examine the training data
    examine_training_data()
    
    # Test the model with real data
    test_model_with_real_data()
    
    logger.info("Training data structure examination complete.")
