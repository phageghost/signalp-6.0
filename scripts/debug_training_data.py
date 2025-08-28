#!/usr/bin/env python3
"""
Debug script to check the actual training data for token ID issues.
This will help identify if the NaN problem is caused by invalid token IDs.
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

def check_tokenizer_vocab():
    """Check the tokenizer vocabulary size and range."""
    
    logger.info("Checking tokenizer vocabulary...")
    
    try:
        tokenizer = ProteinBertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        
        vocab_size = tokenizer.tokenizer.vocab_size
        logger.info(f"Tokenizer vocabulary size: {vocab_size}")
        
        # Check what tokens are available
        special_tokens = tokenizer.tokenizer.special_tokens_map
        logger.info(f"Special tokens: {special_tokens}")
        
        # Check token ID ranges
        all_token_ids = list(range(vocab_size))
        logger.info(f"Token ID range: [{min(all_token_ids)}, {max(all_token_ids)}]")
        
        return tokenizer, vocab_size
        
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return None, None

def check_training_data_tokens(data_path):
    """Check the actual training data for token ID issues."""
    
    logger.info(f"Checking training data at: {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        return
    
    try:
        # Try to load a small sample of the data
        from signalp6.training_utils import RegionCRFDataset
        
        # Create a minimal dataset to check token IDs
        tokenizer, vocab_size = check_tokenizer_vocab()
        if tokenizer is None:
            return
        
        # Create a small dataset
        dataset = RegionCRFDataset(
            data_path,
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
        
        logger.info(f"Dataset size: {len(dataset)}")
        
        # Check first few samples
        for i in range(min(5, len(dataset))):
            try:
                sample = dataset[i]
                logger.info(f"Sample {i}: {sample}")
                
                # Check token IDs if available
                if len(sample) >= 1:
                    data = sample[0]  # First element should be token IDs
                    if isinstance(data, torch.Tensor):
                        logger.info(f"Sample {i} token IDs shape: {data.shape}")
                        logger.info(f"Sample {i} token ID range: [{data.min()}, {data.max()}]")
                        
                        # Check for invalid token IDs
                        if data.max() >= vocab_size:
                            logger.error(f"Sample {i} has invalid token ID: {data.max()} >= {vocab_size}")
                        
                        # Check for NaN/Inf
                        if torch.isnan(data).any():
                            logger.error(f"Sample {i} has NaN in token IDs")
                        if torch.isinf(data).any():
                            logger.error(f"Sample {i} has Inf in token IDs")
                        
                        # Check for negative token IDs
                        if data.min() < 0:
                            logger.error(f"Sample {i} has negative token ID: {data.min()}")
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Failed to check training data: {e}")
        import traceback
        traceback.print_exc()

def check_model_embedding_layer():
    """Check if the model's embedding layer can handle the token IDs."""
    
    logger.info("Checking model embedding layer...")
    
    try:
        # Create model config
        config = BertConfig.from_pretrained("Rostlab/prot_bert")
        config.num_labels = 37
        config.num_global_labels = 6
        config.use_large_crf = True
        config.use_region_labels = True
        config.use_kingdom_id = True
        config.kingdom_embed_size = 32
        
        # Create model
        model = BertSequenceTaggingCRF.from_pretrained("Rostlab/prot_bert", config=config)
        
        # Check embedding layer
        embedding_layer = model.bert.embeddings.word_embeddings
        logger.info(f"Embedding layer weight shape: {embedding_layer.weight.shape}")
        logger.info(f"Embedding layer num_embeddings: {embedding_layer.num_embeddings}")
        logger.info(f"Embedding layer embedding_dim: {embedding_layer.embedding_dim}")
        
        # Check if weights have NaN/Inf
        if torch.isnan(embedding_layer.weight).any():
            logger.error("NaN detected in embedding layer weights!")
        if torch.isinf(embedding_layer.weight).any():
            logger.error("Inf detected in embedding layer weights!")
        
        logger.info(f"Embedding weights range: [{embedding_layer.weight.min():.6f}, {embedding_layer.weight.max():.6f}]")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to check model embedding layer: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_token_id_limits():
    """Test what happens when we use token IDs at the limit of the vocabulary."""
    
    logger.info("Testing token ID limits...")
    
    model = check_model_embedding_layer()
    if model is None:
        return
    
    try:
        # Test with token IDs at the edge of vocabulary
        vocab_size = model.bert.embeddings.word_embeddings.num_embeddings
        
        # Test with valid token IDs
        valid_tokens = torch.randint(0, vocab_size, (2, 10))
        logger.info(f"Testing with valid token IDs: {valid_tokens}")
        
        # Test with invalid token IDs (beyond vocabulary)
        invalid_tokens = torch.randint(vocab_size, vocab_size + 10, (2, 10))
        logger.info(f"Testing with invalid token IDs: {invalid_tokens}")
        
        # Test embedding lookup with valid tokens
        try:
            valid_embeddings = model.bert.embeddings.word_embeddings(valid_tokens)
            logger.info(f"Valid token embeddings shape: {valid_embeddings.shape}")
            logger.info(f"Valid token embeddings range: [{valid_embeddings.min():.6f}, {valid_embeddings.max():.6f}]")
        except Exception as e:
            logger.error(f"Failed to get valid token embeddings: {e}")
        
        # Test embedding lookup with invalid tokens
        try:
            invalid_embeddings = model.bert.embeddings.word_embeddings(invalid_tokens)
            logger.info(f"Invalid token embeddings shape: {invalid_embeddings.shape}")
            logger.info(f"Invalid token embeddings range: [{invalid_embeddings.min():.6f}, {invalid_embeddings.max():.6f}]")
        except Exception as e:
            logger.error(f"Failed to get invalid token embeddings: {e}")
            logger.info("This is expected - invalid token IDs should cause errors")
        
    except Exception as e:
        logger.error(f"Failed to test token ID limits: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Starting training data debugging...")
    
    # Check 1: Tokenizer vocabulary
    check_tokenizer_vocab()
    
    # Check 2: Model embedding layer
    check_model_embedding_layer()
    
    # Check 3: Token ID limits
    test_token_id_limits()
    
    # Check 4: Actual training data (if available)
    data_path = "data/train_set.fasta"
    if os.path.exists(data_path):
        check_training_data_tokens(data_path)
    else:
        logger.warning(f"Training data not found at {data_path}")
    
    logger.info("Training data debugging complete.")
