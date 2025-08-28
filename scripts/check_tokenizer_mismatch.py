#!/usr/bin/env python3
"""
Check for tokenizer mismatches between data creation and training.
This will help identify if different tokenizers were used.
"""

import os
import torch
import logging
from signalp6.models import ProteinBertTokenizer
from transformers import BertTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_different_tokenizers():
    """Check different tokenizer implementations."""

    logger.info("Checking different tokenizer implementations...")

    # Check 1: ProteinBertTokenizer (what you're using)
    try:
        logger.info("=== ProteinBertTokenizer ===")
        tokenizer1 = ProteinBertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False
        )
        logger.info(f"Vocabulary size: {tokenizer1.tokenizer.vocab_size}")
        logger.info(f"Special tokens: {tokenizer1.tokenizer.special_tokens_map}")

        # Check what happens when we tokenize a simple sequence
        test_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        tokens1 = tokenizer1.tokenize(test_seq)
        logger.info(f"Tokenized sequence: {tokens1[:10]}...")  # First 10 tokens

        # Convert to IDs
        ids1 = tokenizer1.convert_tokens_to_ids(tokens1)
        logger.info(f"Token IDs: {ids1[:10]}...")  # First 10 IDs
        logger.info(f"ID range: [{min(ids1)}, {max(ids1)}]")

    except Exception as e:
        logger.error(f"ProteinBertTokenizer failed: {e}")

    # Check 2: Standard BertTokenizer
    try:
        logger.info("\n=== Standard BertTokenizer ===")
        tokenizer2 = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False
        )
        logger.info(f"Vocabulary size: {tokenizer2.vocab_size}")
        logger.info(f"Special tokens: {tokenizer2.special_tokens_map}")

        # Check what happens when we tokenize the same sequence
        tokens2 = tokenizer2.tokenize(test_seq)
        logger.info(f"Tokenized sequence: {tokens2[:10]}...")

        # Convert to IDs
        ids2 = tokenizer2.convert_tokens_to_ids(tokens2)
        logger.info(f"Token IDs: {ids2[:10]}...")
        logger.info(f"ID range: [{min(ids2)}, {max(ids2)}]")

    except Exception as e:
        logger.error(f"Standard BertTokenizer failed: {e}")

    # Check 3: Check if there's a custom tokenizer in your data directory
    logger.info("\n=== Checking for custom tokenizer ===")
    custom_tokenizer_path = "data/tokenizer"
    if os.path.exists(custom_tokenizer_path):
        logger.info(f"Found custom tokenizer at: {custom_tokenizer_path}")
        try:
            # Try to load the custom tokenizer
            custom_tokenizer = ProteinBertTokenizer.from_pretrained(
                custom_tokenizer_path, do_lower_case=False
            )
            logger.info(
                f"Custom tokenizer vocabulary size: {custom_tokenizer.tokenizer.vocab_size}"
            )
            logger.info(
                f"Custom tokenizer special tokens: {custom_tokenizer.tokenizer.special_tokens_map}"
            )

            # Test with the same sequence
            tokens_custom = custom_tokenizer.tokenize(test_seq)
            ids_custom = custom_tokenizer.convert_tokens_to_ids(tokens_custom)
            logger.info(
                f"Custom tokenizer ID range: [{min(ids_custom)}, {max(ids_custom)}]"
            )

        except Exception as e:
            logger.error(f"Custom tokenizer failed: {e}")
    else:
        logger.info("No custom tokenizer found")


def check_training_script_tokenizer():
    """Check what tokenizer the training script is actually using."""

    logger.info("\n=== Checking training script tokenizer usage ===")

    # Look at the training script to see what tokenizer it loads
    training_script_path = "scripts/train_model.py"

    if os.path.exists(training_script_path):
        with open(training_script_path, "r") as f:
            content = f.read()

        # Look for tokenizer loading patterns
        if "TOKENIZER_DICT" in content:
            logger.info("Found TOKENIZER_DICT in training script")
            # Extract the tokenizer dictionary
            start = content.find("TOKENIZER_DICT = {")
            if start != -1:
                end = content.find("}", start)
                tokenizer_dict = content[start : end + 1]
                logger.info(f"Tokenizer dictionary: {tokenizer_dict}")

        # Look for kingdom_as_token usage
        if "kingdom_as_token" in content:
            logger.info("Training script uses kingdom_as_token feature")
            if "data/tokenizer" in content:
                logger.info(
                    "Training script loads custom tokenizer from data/tokenizer"
                )

        # Look for tokenizer loading in the main loop
        if "from_pretrained" in content and "tokenizer" in content:
            logger.info("Training script loads tokenizer with from_pretrained")
    else:
        logger.error("Training script not found")


def check_data_creation_script():
    """Check if there's a data creation script that might use a different tokenizer."""

    logger.info("\n=== Checking for data creation scripts ===")

    # Look for scripts that might create the training data
    scripts_dir = "scripts"
    if os.path.exists(scripts_dir):
        for script in os.listdir(scripts_dir):
            if script.endswith(".py") and "data" in script.lower():
                logger.info(f"Found potential data script: {script}")

                script_path = os.path.join(scripts_dir, script)
                try:
                    with open(script_path, "r") as f:
                        content = f.read()

                    # Look for tokenizer usage
                    if "tokenizer" in content:
                        logger.info(f"  {script} contains tokenizer references")
                    if "from_pretrained" in content:
                        logger.info(f"  {script} uses from_pretrained")

                except Exception as e:
                    logger.error(f"  Error reading {script}: {e}")


if __name__ == "__main__":
    logger.info("Starting tokenizer mismatch investigation...")

    # Check 1: Different tokenizer implementations
    check_different_tokenizers()

    # Check 2: Training script tokenizer usage
    check_training_script_tokenizer()

    # Check 3: Data creation scripts
    check_data_creation_script()

    logger.info("Tokenizer mismatch investigation complete.")
