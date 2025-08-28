#!/usr/bin/env python3
"""
Debug script to identify the source of NaN values in SignalP 6.0 training.
This script will help isolate where the NaN values are coming from.
"""

import torch
import numpy as np
import logging
from signalp6.models import ProteinBertTokenizer, BertSequenceTaggingCRF
from transformers import BertConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_model_forward_pass():
    """Test a simple forward pass to identify where NaN values appear."""
    
    # Create a minimal configuration
    config = BertConfig.from_pretrained("Rostlab/prot_bert")
    config.num_labels = 37
    config.num_global_labels = 6
    config.use_large_crf = True
    config.use_region_labels = True
    config.use_kingdom_id = True
    config.kingdom_embed_size = 32
    
    # Create model
    model = BertSequenceTaggingCRF.from_pretrained("Rostlab/prot_bert", config=config)
    model.eval()
    
    # Create dummy input data
    batch_size = 4
    seq_len = 50
    
    # Input data (token IDs)
    data = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    
    # Input mask
    input_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    
    # Global targets (6 classes)
    global_targets = torch.randint(0, 6, (batch_size,), dtype=torch.long)
    
    # Kingdom IDs
    kingdom_ids = torch.randint(0, 4, (batch_size,), dtype=torch.long)
    
    # Position targets (37 classes, one-hot encoded)
    targets = torch.zeros((batch_size, seq_len, 37), dtype=torch.long)
    targets[:, :, 0] = 1  # Set all to class 0 initially
    
    logger.info("Testing model forward pass...")
    logger.info(f"Input shapes: data={data.shape}, targets={targets.shape}, "
                f"global_targets={global_targets.shape}, kingdom_ids={kingdom_ids.shape}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(
                data,
                global_targets=None,
                targets=None,  # Don't specify targets for inference
                targets_bitmap=None,  # Don't specify targets_bitmap for inference
                input_mask=input_mask,
                sample_weights=None,
                kingdom_ids=kingdom_ids,
            )
        
        logger.info("Forward pass successful!")
        logger.info(f"Loss: {loss}")
        logger.info(f"Global probs shape: {global_probs.shape}")
        logger.info(f"Position probs shape: {pos_probs.shape}")
        logger.info(f"Position predictions shape: {pos_preds.shape}")
        
        # Check for NaN/Inf
        if torch.isnan(loss).any():
            logger.error("NaN detected in loss!")
        if torch.isinf(loss).any():
            logger.error("Inf detected in loss!")
        if torch.isnan(global_probs).any():
            logger.error("NaN detected in global probabilities!")
        if torch.isnan(pos_probs).any():
            logger.error("NaN detected in position probabilities!")
            
        # Log value ranges
        logger.info(f"Loss range: [{loss.min()}, {loss.max()}]")
        logger.info(f"Global probs shape: {global_probs.shape}")
        logger.info(f"Position probs shape: {pos_probs.shape}")
        
        # Test training mode forward pass (this is what causes NaN in training)
        logger.info("Testing training mode forward pass...")
        model.train()
        try:
            loss_train, global_probs_train, pos_probs_train, pos_preds_train = model(
                data,
                global_targets=global_targets,
                targets=targets,
                targets_bitmap=None,
                input_mask=input_mask,
                sample_weights=None,
                kingdom_ids=kingdom_ids,
            )
            
            logger.info("Training forward pass successful!")
            logger.info(f"Training loss: {loss_train}")
            logger.info(f"Training loss shape: {loss_train.shape}")
            
            # Check for NaN/Inf in training mode
            if torch.isnan(loss_train).any():
                logger.error("NaN detected in training loss!")
            if torch.isinf(loss_train).any():
                logger.error("Inf detected in training loss!")
                
            logger.info(f"Training loss range: [{loss_train.min()}, {loss_train.max()}]")
            
        except Exception as e:
            logger.error(f"Training forward pass failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_loss_computation():
    """Test the loss computation specifically."""
    
    # Create dummy logits and targets
    batch_size = 4
    seq_len = 50
    num_classes = 37
    
    # Create logits with reasonable values
    logits = torch.randn(batch_size, seq_len, num_classes) * 0.1  # Small values
    
    # Create targets
    targets = torch.randint(0, num_classes, (batch_size, seq_len))
    
    # Test CRF loss computation
    try:
        from torchcrf import CRF
        
        crf = CRF(num_classes, batch_first=True)
        
        # Compute loss
        loss = -crf(logits, targets)
        
        logger.info(f"CRF loss: {loss}")
        logger.info(f"Loss shape: {loss.shape}")
        
        if torch.isnan(loss).any():
            logger.error("NaN detected in CRF loss!")
        if torch.isinf(loss).any():
            logger.error("Inf detected in CRF loss!")
            
    except ImportError:
        logger.warning("torchcrf not available, skipping CRF loss test")
        logger.info("This is expected if you don't have torchcrf installed")
    except Exception as e:
        logger.error(f"CRF loss computation failed: {e}")
        import traceback
        traceback.print_exc()

def test_hyperparameter_sensitivity():
    """Test if the issue is related to hyperparameters."""
    
    logger.info("Testing hyperparameter sensitivity...")
    
    # Test different learning rates
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    
    for lr in learning_rates:
        logger.info(f"Testing learning rate: {lr}")
        
        # Create a simple model for testing
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        # Create dummy data
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        
        # Forward pass
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        
        # Check for NaN/Inf
        if torch.isnan(loss).any():
            logger.error(f"NaN detected with learning rate {lr}!")
        if torch.isinf(loss).any():
            logger.error(f"Inf detected with learning rate {lr}!")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    logger.error(f"NaN gradient detected in {name} with learning rate {lr}!")
                if torch.isinf(param.grad).any():
                    logger.error(f"Inf gradient detected in {name} with learning rate {lr}!")
        
        optimizer.step()
        
        logger.info(f"Learning rate {lr}: Loss = {loss.item():.6f}")
    
    logger.info("Hyperparameter sensitivity test complete.")

def test_device_and_dtype():
    """Test if the issue is related to device placement or data types."""
    
    logger.info("Testing device and data type compatibility...")
    
    # Test different devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    for device_name in devices:
        logger.info(f"Testing on device: {device_name}")
        device = torch.device(device_name)
        
        # Test different data types
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            logger.info(f"Testing with dtype: {dtype}")
            
            try:
                # Create model and move to device
                model = torch.nn.Linear(10, 1).to(device).to(dtype)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                
                # Create data with specific dtype and move to device
                x = torch.randn(4, 10, dtype=dtype, device=device)
                y = torch.randn(4, 1, dtype=dtype, device=device)
                
                # Forward pass
                pred = model(x)
                loss = torch.nn.functional.mse_loss(pred, y)
                
                # Check for NaN/Inf
                if torch.isnan(loss).any():
                    logger.error(f"NaN detected on {device_name} with {dtype}!")
                if torch.isinf(loss).any():
                    logger.error(f"Inf detected on {device_name} with {dtype}!")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Check gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            logger.error(f"NaN gradient detected in {name} on {device_name} with {dtype}!")
                        if torch.isinf(param.grad).any():
                            logger.error(f"Inf gradient detected in {name} on {device_name} with {dtype}!")
                
                optimizer.step()
                
                logger.info(f"Device {device_name}, dtype {dtype}: Loss = {loss.item():.6f}")
                
            except Exception as e:
                logger.error(f"Error on {device_name} with {dtype}: {e}")
    
    logger.info("Device and data type test complete.")

if __name__ == "__main__":
    logger.info("Starting NaN debugging...")
    
    # Test 1: Model forward pass
    test_model_forward_pass()
    
    # Test 2: Loss computation
    test_loss_computation()
    
    # Test 3: Hyperparameter sensitivity
    test_hyperparameter_sensitivity()
    
    # Test 4: Device and data type compatibility
    test_device_and_dtype()
    
    logger.info("NaN debugging complete.")
