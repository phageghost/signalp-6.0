#!/usr/bin/env python3
"""
Debug script to investigate model loading and see where CRF parameters get corrupted.
"""

import os
import sys
import torch
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signalp6.models.bert_crf import BertSequenceTaggingCRF
from transformers import BertConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_model_loading():
    """Debug model loading to see where CRF parameters get corrupted."""
    
    logger.info("Debugging model loading process...")
    
    # Create a minimal config
    config = BertConfig(
        vocab_size=30,  # Default ProtBERT vocab size
        hidden_size=1024,
        num_hidden_layers=30,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        num_labels=37,  # Number of sequence labels
        num_global_labels=6,  # Number of global labels (note: not global_labels)
        kingdom_embed_size=0,  # No kingdom embedding for this test
        use_region_labels=True,  # Enable region labels
    )
    
    logger.info("Created config with num_labels=37, global_labels=6")
    
    # Test 1: Create model from scratch
    logger.info("Test 1: Creating model from scratch...")
    model_scratch = BertSequenceTaggingCRF(config)
    
    logger.info(f"Model scratch CRF transitions: shape={model_scratch.crf.transitions.shape}")
    logger.info(f"Model scratch CRF transitions: has NaN={torch.isnan(model_scratch.crf.transitions).any()}")
    logger.info(f"Model scratch CRF transitions: has Inf={torch.isinf(model_scratch.crf.transitions).any()}")
    if not torch.isnan(model_scratch.crf.transitions).any() and not torch.isinf(model_scratch.crf.transitions).any():
        logger.info(f"Model scratch CRF transitions: min={model_scratch.crf.transitions.min():.6f}, max={model_scratch.crf.transitions.max():.6f}")
    
    logger.info(f"Model scratch CRF constraint mask: shape={model_scratch.crf._constraint_mask.shape}")
    logger.info(f"Model scratch CRF constraint mask: has NaN={torch.isnan(model_scratch.crf._constraint_mask).any()}")
    logger.info(f"Model scratch CRF constraint mask: has Inf={torch.isinf(model_scratch.crf._constraint_mask).any()}")
    if not torch.isnan(model_scratch.crf._constraint_mask).any() and not torch.isinf(model_scratch.crf._constraint_mask).any():
        logger.info(f"Model scratch CRF constraint mask: min={model_scratch.crf._constraint_mask.min():.6f}, max={model_scratch.crf._constraint_mask.max():.6f}")
    
    # Test 2: Load model from pretrained checkpoint
    logger.info("Test 2: Loading model from pretrained checkpoint...")
    try:
        model_pretrained = BertSequenceTaggingCRF.from_pretrained("Rostlab/prot_bert", config=config)
        
        logger.info(f"Model pretrained CRF transitions: shape={model_pretrained.crf.transitions.shape}")
        logger.info(f"Model pretrained CRF transitions: has NaN={torch.isnan(model_pretrained.crf.transitions).any()}")
        logger.info(f"Model pretrained CRF transitions: has Inf={torch.isinf(model_pretrained.crf.transitions).any()}")
        if not torch.isnan(model_pretrained.crf.transitions).any() and not torch.isinf(model_pretrained.crf.transitions).any():
            logger.info(f"Model pretrained CRF transitions: min={model_pretrained.crf.transitions.min():.6f}, max={model_pretrained.crf.transitions.max():.6f}")
        
        logger.info(f"Model pretrained CRF constraint mask: shape={model_pretrained.crf._constraint_mask.shape}")
        logger.info(f"Model pretrained CRF constraint mask: has NaN={torch.isnan(model_pretrained.crf._constraint_mask).any()}")
        logger.info(f"Model pretrained CRF constraint mask: has Inf={torch.isinf(model_pretrained.crf._constraint_mask).any()}")
        if not torch.isnan(model_pretrained.crf._constraint_mask).any() and not torch.isinf(model_pretrained.crf._constraint_mask).any():
            logger.info(f"Model pretrained CRF constraint mask: min={model_pretrained.crf._constraint_mask.min():.6f}, max={model_pretrained.crf._constraint_mask.max():.6f}")
        
    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
    
    # Test 3: Move model to device
    logger.info("Test 3: Moving model to device...")
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info(f"Moving model to {device}")
        
        model_scratch = model_scratch.to(device)
        
        logger.info(f"Model scratch on {device} CRF transitions: has NaN={torch.isnan(model_scratch.crf.transitions).any()}")
        logger.info(f"Model scratch on {device} CRF constraint mask: has NaN={torch.isnan(model_scratch.crf._constraint_mask).any()}")
        
    else:
        logger.info("MPS not available, skipping device test")
    
    logger.info("Model loading debugging complete.")

if __name__ == "__main__":
    debug_model_loading()
