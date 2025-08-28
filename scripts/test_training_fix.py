#!/usr/bin/env python3
"""
Test script to verify that the training fix resolves the NaN issue.
This will run a minimal training loop to check if the model forward pass works correctly.
"""

import os
import torch
import logging
from signalp6.models import ProteinBertTokenizer, BertSequenceTaggingCRF
from transformers import BertConfig
from signalp6.training_utils import RegionCRFDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_training_fix():
    """Test if the training fix resolves the NaN issue."""

    logger.info("Testing training fix...")

    try:
        # Load tokenizer and dataset
        tokenizer = ProteinBertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False
        )

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

        # Create model
        config = BertConfig.from_pretrained("Rostlab/prot_bert")
        config.num_labels = 37
        config.num_global_labels = 6
        config.use_large_crf = True
        config.use_region_labels = True
        config.use_kingdom_id = True
        config.kingdom_embed_size = 32

        model = BertSequenceTaggingCRF(config)
        model.train()  # Set to training mode

        # Test with a small batch
        batch_samples = [dataset[i] for i in range(min(2, len(dataset)))]
        collated = dataset.collate_fn(batch_samples)

        # Unpack the collated batch
        (
            data,
            targets,
            input_mask,
            global_targets,
            cleavage_sites,
            sample_weights,
            kingdom_ids,
        ) = collated

        logger.info(f"Batch shapes:")
        logger.info(f"  data: {data.shape}")
        logger.info(f"  targets: {targets.shape}")
        logger.info(f"  input_mask: {input_mask.shape}")
        logger.info(f"  global_targets: {global_targets.shape}")
        logger.info(f"  kingdom_ids: {kingdom_ids.shape}")

        # Test the model forward pass (same as in training)
        loss, global_probs, pos_probs, pos_preds = model(
            data,
            global_targets=global_targets,  # This was the fix - passing global_targets
            targets=None,  # Using targets_bitmap for region labels
            targets_bitmap=targets,
            input_mask=input_mask,
            sample_weights=sample_weights,
            kingdom_ids=kingdom_ids,
        )

        logger.info("Forward pass successful!")
        logger.info(f"Loss: {loss}")
        logger.info(f"Loss shape: {loss.shape}")
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

        logger.info("✅ No NaNs detected! Training fix is working.")

        # Test backward pass
        loss.backward()
        logger.info("✅ Backward pass successful!")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training_fix()
    if success:
        logger.info("🎉 Training fix test PASSED! The NaN issue should be resolved.")
    else:
        logger.error("💥 Training fix test FAILED! The NaN issue persists.")
