#!/usr/bin/env python3
"""
Test script to verify that the training now works with the corrected learning rate.
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


def test_fixed_training():
    """Test if the training now works with the corrected learning rate."""

    logger.info("Testing training with corrected learning rate (0.001)...")

    try:
        # Load tokenizer and dataset
        tokenizer = ProteinBertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False
        )

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

        # Create optimizer with the CORRECTED learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1.2e-6)

        logger.info("Model and optimizer created with lr=0.001")

        # Test with first batch
        batch = next(iter(train_loader))

        if len(batch) == 7:
            (
                data,
                targets,
                input_mask,
                global_targets,
                cleavage_sites,
                sample_weights,
                kingdom_ids,
            ) = batch
        else:
            data, targets, input_mask, global_targets, sample_weights, kingdom_ids = (
                batch
            )

        # Move to device
        data = data.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        input_mask = input_mask.to(device, dtype=torch.bool)
        global_targets = global_targets.to(device, dtype=torch.long)
        sample_weights = (
            sample_weights.to(device, dtype=torch.float32)
            if sample_weights is not None
            else None
        )
        kingdom_ids = kingdom_ids.to(device, dtype=torch.long)

        logger.info("Data moved to device")

        # Test multiple training steps
        logger.info("Testing multiple training steps...")
        for step in range(10):  # Test 10 steps
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
            nan_grads = 0
            inf_grads = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        nan_grads += 1
                        logger.error(f"❌ NaN gradients at step {step} in {name}!")
                        return False
                    if torch.isinf(param.grad).any():
                        inf_grads += 1
                        logger.error(f"❌ Inf gradients at step {step} in {name}!")
                        return False

            # Optimizer step
            optimizer.step()

            # Check parameters
            nan_params = 0
            inf_params = 0
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params += 1
                    logger.error(f"❌ NaN parameters at step {step} in {name}!")
                    return False
                if torch.isinf(param).any():
                    inf_params += 1
                    logger.error(f"❌ Inf parameters at step {step} in {name}!")
                    return False

            logger.info(f"Step {step} completed successfully, loss: {loss.item():.6f}")

        logger.info("✅ All training steps completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Fixed training test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fixed_training()
    if success:
        logger.info("🎉 Fixed training test PASSED! The NaN issue is resolved.")
        logger.info(
            "You can now run your training script with the corrected learning rate."
        )
    else:
        logger.error("💥 Fixed training test FAILED! The issue persists.")
