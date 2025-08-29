# © Copyright Technical University of Denmark
"""
Script to train Bert-CRF (SignalP 6.0).

Loas a pretrained Bert checkpoint into the custom `BertSequenceTaggingCRF` model.
Sets config as specified by CLI. Consult help string of args for info.

Uses wandb for logging. There's a dirty fix in place to prevent wandb from logging, when wandb was
initialized already. This is necessary when spawning multiple train loops from the same script, because
wandb reinit is still bugged.
"""
import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    recall_score,
    precision_score,
)
import torch
from transformers import BertConfig
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb

from signalp6.models import ProteinBertTokenizer, BertSequenceTaggingCRF
from signalp6.utils import get_metrics_multistate
from signalp6.utils import class_aware_cosine_similarities, get_region_lengths
from signalp6.training_utils import (
    LargeCRFPartitionDataset,
    SIGNALP_KINGDOM_DICT,
    RegionCRFDataset,
    compute_cosine_region_regularization,
    Adamax,
)


# get the git hash - and log it
# wandb does that automatically - but only when in the correct directory when launching the job.
# by also doing it manually, force to launch from the correct directory, because otherwise this command will fail.
GIT_HASH = (
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()
)
MODEL_DICT = {
    "bert_prottrans": (BertConfig, BertSequenceTaggingCRF),
}
TOKENIZER_DICT = {"bert_prottrans": (ProteinBertTokenizer, "Rostlab/prot_bert")}


# Fix OpenMP conflict on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set default tensor type to float32 for Metal compatibility
torch.set_default_dtype(torch.float32)


# Use Metal Performance Shaders (MPS) on Apple Silicon, CUDA on NVIDIA, or CPU as fallback
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def log_metrics(metrics_dict, split: str, step: int):
    """Convenience function to add prefix to all metrics before logging."""
    wandb.log(
        {
            f"{split.capitalize()} {name.capitalize()}": value
            for name, value in metrics_dict.items()
        },
        step=step,
    )


def setup_logger(log_fpath: str, verbosity: int = logging.INFO) -> logging.Logger:
    """
    Set up the logger.

    Args:
        log_fpath (str): The path to the log file.
        verbosity (int, optional): The level of logging. Defaults to logging.INFO.
    """
    # global logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(verbosity)

    # create file handler
    fh = logging.FileHandler(log_fpath, mode="w")
    fh.setLevel(verbosity)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("Logger set up at level %s and writing to %s.", verbosity, log_fpath)

    return logger


# This is a quick fix for hyperparameter search.
# wandb reinit does not work on scientific linux yet, so use
# a pseudo-wandb instead of the actual wandb library
class DecoyConfig:
    def update(self, *args, **kwargs):
        pass


class DecoyWandb:
    config = DecoyConfig()

    def init(self, *args, **kwargs):
        print(
            "Decoy Wandb initiated, override wandb with no-op logging to prevent errors."
        )
        pass

    def log(self, value_dict, *args, **kwargs):
        # filter for train logs here, don't want to print at every step
        if list(value_dict.keys())[0].startswith("Train"):
            pass
        else:
            print(value_dict)
            print(args)
            print(kwargs)

    def watch(self, *args, **kwargs):
        pass


def tagged_seq_to_cs_multiclass(tagged_seqs: np.ndarray, sp_tokens=(0, 4, 5)):
    """Convert a sequences of tokens to the index of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels
        sp_tokens: label tokens that indicate a signal peptide
    Returns:
        cs_sites: (batch_size) integer array of last position that is a SP. NaN if no SP present in sequence.
    """

    def get_last_sp_idx(x: np.ndarray) -> int:
        """Func1d to get the last index that is tagged as SP. use with np.apply_along_axis."""
        sp_idx = np.where(np.isin(x, sp_tokens))[0]
        if len(sp_idx) > 0:
            max_idx = sp_idx.max() + 1
        else:
            max_idx = np.nan
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites


def report_metrics(
    true_global_labels: np.ndarray,
    pred_global_labels: np.ndarray,
    true_sequence_labels: np.ndarray,
    pred_sequence_labels: np.ndarray,
    use_cs_tag=False,
) -> Dict[str, float]:
    """Utility function to get metrics from model output"""
    true_cs = tagged_seq_to_cs_multiclass(
        true_sequence_labels, sp_tokens=[4, 9, 14] if use_cs_tag else [3, 7, 11]
    )
    pred_cs = tagged_seq_to_cs_multiclass(
        pred_sequence_labels, sp_tokens=[4, 9, 14] if use_cs_tag else [3, 7, 11]
    )
    # Ensure both arrays are 1D and have the same shape before creating the mask
    if true_cs.ndim != 1 or pred_cs.ndim != 1:
        # Flatten arrays if needed
        true_cs = true_cs.flatten()
        pred_cs = pred_cs.flatten()

    # Ensure both arrays have the same length
    if len(true_cs) != len(pred_cs):
        min_len = min(len(true_cs), len(pred_cs))
        true_cs = true_cs[:min_len]
        pred_cs = pred_cs[:min_len]

    # Now create the mask and filter out NaN values
    valid_mask = ~np.isnan(true_cs)

    # Apply the mask
    pred_cs = pred_cs[valid_mask]
    true_cs = true_cs[valid_mask]

    # Replace any remaining NaN values with -1
    true_cs = np.where(np.isnan(true_cs), -1, true_cs)
    pred_cs = np.where(np.isnan(pred_cs), -1, pred_cs)

    # applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent to the argmax.
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)
    metrics_dict = {}
    metrics_dict["CS Recall"] = recall_score(true_cs, pred_cs, average="micro")
    metrics_dict["CS Precision"] = precision_score(true_cs, pred_cs, average="micro")
    metrics_dict["CS MCC"] = matthews_corrcoef(true_cs, pred_cs)
    metrics_dict["Detection MCC"] = matthews_corrcoef(
        true_global_labels, pred_global_labels_thresholded
    )

    return metrics_dict


def report_metrics_kingdom_averaged(
    true_global_labels: np.ndarray,
    pred_global_labels: np.ndarray,
    true_sequence_labels: np.ndarray,
    pred_sequence_labels: np.ndarray,
    kingdom_ids: np.ndarray,
    input_token_ids: np.ndarray,
    cleavage_sites: np.ndarray = None,
    use_cs_tag=False,
) -> Dict[str, float]:
    """Utility function to get metrics from model output"""

    sp_tokens = [3, 7, 11, 15, 19]
    if use_cs_tag:
        sp_tokens = [4, 9, 14]
    if (
        cleavage_sites is not None
    ):  # implicit: when cleavage sites are provided, am using region states
        sp_tokens = [5, 11, 19, 26, 31]
        true_cs = cleavage_sites.astype(float)
        # need to convert so np.isnan works
        true_cs[true_cs == -1] = np.nan
    else:
        true_cs = tagged_seq_to_cs_multiclass(true_sequence_labels, sp_tokens=sp_tokens)

    pred_cs = tagged_seq_to_cs_multiclass(pred_sequence_labels, sp_tokens=sp_tokens)

    # Ensure both arrays are 1D and have the same shape before creating the mask
    if true_cs.ndim != 1 or pred_cs.ndim != 1:
        # Flatten arrays if needed
        true_cs = true_cs.flatten()
        pred_cs = pred_cs.flatten()

    # Ensure both arrays have the same length
    if len(true_cs) != len(pred_cs):
        min_len = min(len(true_cs), len(pred_cs))
        true_cs = true_cs[:min_len]
        pred_cs = pred_cs[:min_len]

    # Now create the mask and filter out NaN values
    valid_mask = ~np.isnan(true_cs)

    # Apply the mask
    cs_kingdom = kingdom_ids[valid_mask]
    pred_cs = pred_cs[valid_mask]
    true_cs = true_cs[valid_mask]

    # Replace any remaining NaN values with -1
    true_cs = np.where(np.isnan(true_cs), -1, true_cs)
    pred_cs = np.where(np.isnan(pred_cs), -1, pred_cs)

    # applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent to the argmax.
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)

    # compute metrics for each kingdom
    rev_kingdom_dict = dict(
        zip(SIGNALP_KINGDOM_DICT.values(), SIGNALP_KINGDOM_DICT.keys())
    )
    all_cs_mcc = []
    all_detection_mcc = []
    metrics_dict = {}
    for kingdom in np.unique(kingdom_ids):
        kingdom_global_labels = true_global_labels[kingdom_ids == kingdom]
        kingdom_pred_global_labels_thresholded = pred_global_labels_thresholded[
            kingdom_ids == kingdom
        ]
        kingdom_true_cs = true_cs[cs_kingdom == kingdom]
        kingdom_pred_cs = pred_cs[cs_kingdom == kingdom]

        metrics_dict[f"CS Recall {rev_kingdom_dict[kingdom]}"] = recall_score(
            kingdom_true_cs, kingdom_pred_cs, average="micro"
        )
        metrics_dict[f"CS Precision {rev_kingdom_dict[kingdom]}"] = precision_score(
            kingdom_true_cs, kingdom_pred_cs, average="micro"
        )
        metrics_dict[f"CS MCC {rev_kingdom_dict[kingdom]}"] = matthews_corrcoef(
            kingdom_true_cs, kingdom_pred_cs
        )
        metrics_dict[f"Detection MCC {rev_kingdom_dict[kingdom]}"] = matthews_corrcoef(
            kingdom_global_labels, kingdom_pred_global_labels_thresholded
        )

        all_cs_mcc.append(metrics_dict[f"CS MCC {rev_kingdom_dict[kingdom]}"])
        all_detection_mcc.append(
            metrics_dict[f"Detection MCC {rev_kingdom_dict[kingdom]}"]
        )

    if (
        cleavage_sites is not None
    ):  # implicit: when cleavage sites are provided, am using region states
        n_h, h_c = class_aware_cosine_similarities(
            pred_sequence_labels,
            input_token_ids,
            true_global_labels,
            replace_value=np.nan,
            op_mode="numpy",
        )
        n_lengths, h_lengths, c_lengths = get_region_lengths(
            pred_sequence_labels, true_global_labels, agg_fn="none"
        )
        for label in np.unique(true_global_labels):
            if label == 0 or label == 5:
                continue

            metrics_dict[f"Cosine similarity nh {label}"] = np.nanmean(
                n_h[true_global_labels == label]
            )
            metrics_dict[f"Cosine similarity hc {label}"] = np.nanmean(
                h_c[true_global_labels == label]
            )
            metrics_dict[f"Average length n {label}"] = n_lengths[
                true_global_labels == label
            ].mean()
            metrics_dict[f"Average length h {label}"] = h_lengths[
                true_global_labels == label
            ].mean()
            metrics_dict[f"Average length c {label}"] = c_lengths[
                true_global_labels == label
            ].mean()
            # w&b can plot histogram heatmaps over time when logging sequences
            metrics_dict[f"Lengths n {label}"] = n_lengths[true_global_labels == label]
            metrics_dict[f"Lengths h {label}"] = h_lengths[true_global_labels == label]
            metrics_dict[f"Lengths c {label}"] = c_lengths[true_global_labels == label]

    metrics_dict["CS MCC"] = sum(all_cs_mcc) / len(all_cs_mcc)
    metrics_dict["Detection MCC"] = sum(all_detection_mcc) / len(all_detection_mcc)

    return metrics_dict


def check_model_parameters(model: torch.nn.Module, stage: str = "training"):
    """Check if the model parameters contain NaN or Inf values."""
    logger.info(f"Checking model parameters for NaN/Inf values at {stage} start...")

    has_nan = False
    has_inf = False
    problematic_params = []

    # Check parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"NaN detected in model parameter '{name}'!")
            has_nan = True
            problematic_params.append(
                (name, "NaN", param.shape, param.min(), param.max())
            )
        if torch.isinf(param).any():
            logger.error(f"Inf detected in model parameter '{name}'!")
            has_inf = True
            problematic_params.append(
                (name, "Inf", param.shape, param.min(), param.max())
            )

    # Check buffers (like running statistics in BatchNorm, etc.)
    for name, buffer in model.named_buffers():
        if torch.isnan(buffer).any():
            logger.error(f"NaN detected in model buffer '{name}'!")
            has_nan = True
            problematic_params.append(
                (name, "NaN (buffer)", buffer.shape, buffer.min(), buffer.max())
            )
        if torch.isinf(buffer).any():
            logger.error(f"Inf detected in model buffer '{name}'!")
            has_inf = True
            problematic_params.append(
                (name, "Inf (buffer)", buffer.shape, buffer.min(), buffer.max())
            )

    if has_nan or has_inf:
        logger.error(
            f"Model has {len(problematic_params)} problematic parameters/buffers at {stage} start!"
        )
        for name, problem_type, shape, min_val, max_val in problematic_params[
            :5
        ]:  # Show first 5
            logger.error(
                f"  - {name}: {problem_type}, shape {shape}, range [{min_val:.6f}, {max_val:.6f}]"
            )
        return False
    else:
        logger.info(f"All model parameters and buffers are valid at {stage} start")
        return True


def train_epoch(
    model: torch.nn.Module,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.ArgumentParser,
    global_step: int,
) -> Tuple[float, int]:
    """Predict one minibatch and performs update step.
    Returns:
        loss: loss value of the minibatch
    """

    model.train()
    optimizer.zero_grad()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []
    all_kingdom_ids = []  # gather ids for kingdom-averaged metrics
    all_token_ids = []
    all_cs = []
    total_loss = 0

    for i, batch in enumerate(train_data):
        logger.info(f"Processing batch {i} ...")

        logger.debug(f"Batch {i}: Starting forward pass")

        # Check model state


        if args.sp_region_labels:
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
            (
                data,
                targets,
                input_mask,
                global_targets,
                sample_weights,
                kingdom_ids,
            ) = batch

        # input_ids must stay as long integers for embedding layers
        data = data.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        input_mask = input_mask.to(device, dtype=torch.bool)
        global_targets = global_targets.to(device, dtype=torch.long)
        sample_weights = (
            sample_weights.to(device, dtype=torch.float32)
            if args.use_sample_weights
            else None
        )
        kingdom_ids = kingdom_ids.to(device, dtype=torch.long)

        if args.check_nans_and_infs:
            # Check for NaN/Inf in inputs
            if torch.isnan(data).any():
                logger.warning(f"Batch {i}: NaN detected in input data")
            if torch.isnan(targets).any():
                logger.warning(f"Batch {i}: NaN detected in targets")
            if torch.isnan(global_targets).any():
                logger.warning(f"Batch {i}: NaN detected in global targets")
            if sample_weights is not None and torch.isnan(sample_weights).any():
                logger.warning(f"Batch {i}: NaN detected in sample weights")

            # Check for extreme values
            if torch.isinf(data).any():
                logger.warning(f"Batch {i}: Inf detected in input data")
            if torch.isinf(targets).any():
                logger.warning(f"Batch {i}: Inf detected in targets")
            if torch.isinf(global_targets).any():
                logger.warning(f"Batch {i}: Inf detected in global targets")

        # Log input statistics
        logger.debug(f"Batch {i}: Input data range: [{data.min()}, {data.max()}]")
        logger.debug(f"Batch {i}: Targets range: [{targets.min()}, {targets.max()}]")
        logger.debug(
            f"Batch {i}: Global targets range: [{global_targets.min()}, {global_targets.max()}]"
        )
        if sample_weights is not None:
            logger.debug(
                f"Batch {i}: Sample weights range: [{sample_weights.min():.6f}, {sample_weights.max():.6f}]"
            )

        optimizer.zero_grad()

        loss, global_probs, pos_probs, pos_preds = model(
            data,
            global_targets=global_targets,
            targets=targets if not args.sp_region_labels else None,
            targets_bitmap=targets if args.sp_region_labels else None,
            input_mask=input_mask,
            sample_weights=sample_weights,
            kingdom_ids=kingdom_ids if args.kingdom_embed_size > 0 else None,
        )

        # Check for NaN/Inf in raw model outputs
        if args.check_nans_and_infs:
            if torch.isnan(loss).any():
                logger.error(f"Batch {i}: NaN detected in RAW loss from model!")
                logger.error(f"  - Loss shape: {loss.shape}, values: {loss}")
            if torch.isinf(loss).any():
                logger.error(f"Batch {i}: Inf detected in RAW loss from model!")
                logger.error(f"  - Loss shape: {loss.shape}, values: {loss}")
            if torch.isnan(global_probs).any():
                logger.error(f"Batch {i}: NaN detected in global_probs from model!")
            if torch.isnan(pos_probs).any():
                logger.error(f"Batch {i}: NaN detected in pos_probs from model!")

        loss = (
            loss.mean()
        )  # if DataParallel because loss is a vector, if not doesn't matter

        # Check if loss.mean() introduced NaN/Inf
        if args.check_nans_and_infs:
            if torch.isnan(loss).any():
                logger.error(f"Batch {i}: NaN introduced by loss.mean()!")
                logger.error(f"  - Original loss had shape: {loss.shape}")
            if torch.isinf(loss).any():
                logger.error(f"Batch {i}: Inf introduced by loss.mean()!")
                logger.error(f"  - Original loss had shape: {loss.shape}")

        logger.debug(
            f"Batch {i}: Raw loss shape: {loss.shape}, "
            f"Loss value: {loss.item():.6f}"
        )
        logger.debug(
            f"Batch {i}: Global probs shape: {global_probs.shape}, "
            f"Position probs shape: {pos_probs.shape}"
        )
        logger.debug(f"Batch {i}: Position predictions shape: {pos_preds.shape}")
        logger.debug(
            f"Batch {i}: Targets shape: {targets.shape}, "
            f"Global targets shape: {global_targets.shape}"
        )

        # Check for NaN/Inf in model outputs
        if args.check_nans_and_infs:
            if torch.isnan(loss).any():
                logger.warning(f"Batch {i}: NaN detected in loss")
            if torch.isnan(global_probs).any():
                logger.warning(f"Batch {i}: NaN detected in global probabilities")
            if torch.isnan(pos_probs).any():
                logger.warning(f"Batch {i}: NaN detected in position probabilities")
            if torch.isnan(pos_preds).any():
                logger.warning(f"Batch {i}: NaN detected in position predictions")

            if torch.isinf(loss).any():
                logger.warning(f"Batch {i}: Inf detected in loss")
            if torch.isinf(global_probs).any():
                logger.warning(f"Batch {i}: Inf detected in global probabilities")
            if torch.isinf(pos_probs).any():
                logger.warning(f"Batch {i}: Inf detected in position probabilities")

        # Log output statistics
        logger.debug(
            f"Batch {i}: Global probs range: [{global_probs.min():.6f}, {global_probs.max():.6f}]"
        )
        logger.debug(
            f"Batch {i}: Position probs range: [{pos_probs.min():.6f}, {pos_probs.max():.6f}]"
        )
        logger.debug(
            f"Batch {i}: Position predictions range: [{pos_preds.min()}, {pos_preds.max()}]"
        )

        # Check if loss is NaN or Inf before regularization and handle accordingly
        if args.check_nans_and_infs:
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.error(
                    f"Batch {i}: Loss is NaN or Inf BEFORE regularization, skipping batch"
                )
                logger.debug(f"Batch {i}: Skipping backward pass and optimizer step")
                continue

        # Check for extreme loss values that might cause numerical instability
        if loss.item() > 1e6:
            logger.warning(
                f"Batch {i}: Loss value {loss.item():.2e} is extremely high, skipping batch"
            )
            logger.debug(f"Batch {i}: Skipping backward pass and optimizer step")
            continue

        total_loss += loss.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())
        all_token_ids.append(data.detach().cpu().numpy())
        if args.sp_region_labels:
            all_cs.append(cleavage_sites)
        else:
            all_cs.append(None)

        # if args.region_regularization_alpha >0:
        # removing special tokens by indexing should be sufficient.
        # remaining SEP tokens (when sequence was padded) are ignored in aggregation.
        if args.sp_region_labels and args.region_regularization_alpha > 0:

            # Ensure all tensors are on the same device for regularization
            # Make slicing adaptive to handle different sequence lengths
            # We need to ensure pos_probs and data have matching sequence dimensions

            logger.debug(f"Batch {i}: Original pos_probs shape: {pos_probs.shape}")
            logger.debug(f"Batch {i}: Original data shape: {data.shape}")

            # Determine the actual sequence length from pos_probs (model output)
            seq_len = pos_probs.shape[1]

            # Slice data to match the sequence length of pos_probs
            # Remove special tokens from data to match pos_probs length
            if data.shape[1] > seq_len:
                # Data is longer than model output, remove tokens from both ends
                tokens_to_remove = data.shape[1] - seq_len
                start_idx = tokens_to_remove // 2
                end_idx = data.shape[1] - (tokens_to_remove - start_idx)
                pos_probs_device = pos_probs.to(device)
                data_device = data[:, start_idx:end_idx].to(device)
                input_mask_device = input_mask[:, start_idx:end_idx].to(device)
            else:
                # Standard case: remove 1 token from each end
                pos_probs_device = pos_probs[:, 1:-1, :].to(device)
                data_device = data[:, 1:-1].to(device)
                input_mask_device = input_mask[:, 1:-1].to(device)

            global_targets_device = global_targets.to(device)

            logger.debug(
                f"Batch {i}: Adaptive slicing - seq_len: {seq_len}, data.shape[1]: {data.shape[1]}"
            )
            logger.debug(
                f"Batch {i}: Final pos_probs_device shape: {pos_probs_device.shape}"
            )
            logger.debug(f"Batch {i}: Final data_device shape: {data_device.shape}")
            logger.debug(
                f"Batch {i}: pos_probs_device[:2] = {pos_probs_device.shape[:2]}"
            )
            logger.debug(f"Batch {i}: data_device.shape = {data_device.shape}")

            nh, hc = compute_cosine_region_regularization(
                pos_probs_device, data_device, global_targets_device, input_mask_device
            )

            # Check regularization outputs
            if args.check_nans_and_infs:
                if torch.isnan(nh).any():
                    logger.warning(f"Batch {i}: NaN detected in n-h regularization")
                if torch.isnan(hc).any():
                    logger.warning(f"Batch {i}: NaN detected in h-c regularization")
                if torch.isinf(nh).any():
                    logger.warning(f"Batch {i}: Inf detected in n-h regularization")
                if torch.isinf(hc).any():
                    logger.warning(f"Batch {i}: Inf detected in h-c regularization")

            logger.debug(
                f"Batch {i}: n-h regularization range: [{nh.min():.6f}, {nh.max():.6f}], mean: {nh.mean():.6f}"
            )
            logger.debug(
                f"Batch {i}: h-c regularization range: [{hc.min():.6f}, {hc.max():.6f}], mean: {hc.mean():.6f}"
            )
            logger.debug(
                f"Batch {i}: Regularization alpha: {args.region_regularization_alpha}"
            )
            logger.debug(f"Batch {i}: Loss before regularization: {loss.item():.6f}")

            loss = loss + nh.mean() * args.region_regularization_alpha
            loss = loss + hc.mean() * args.region_regularization_alpha

            logger.debug(f"Batch {i}: Loss after regularization: {loss.item():.6f}")

        # Check if loss became NaN or Inf after regularization
        if args.check_nans_and_infs:
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.error(
                    f"Batch {i}: Loss became NaN or Inf AFTER regularization, skipping batch"
                )
            logger.debug(f"Batch {i}: Skipping backward pass and optimizer step")
            continue

        # Check for Inf values in model outputs that might cause issues
        if args.check_nans_and_infs:
            if torch.isinf(global_probs).any() or torch.isinf(pos_probs).any():
                logger.error(
                    f"Batch {i}: Inf values detected in model outputs, skipping batch"
                )
                logger.debug(f"Batch {i}: Skipping backward pass and optimizer step")
                continue

            log_metrics(
                {
                    "n-h regularization": nh.mean().detach().cpu().numpy(),
                    "h-c regularization": hc.mean().detach().cpu().numpy(),
                },
                "train",
                global_step,
            )

        loss.backward()

        # Check for NaN/Inf in gradients
        has_nan_grad = False
        has_inf_grad = False
        if args.check_nans_and_infs:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        logger.error(
                            f"Batch {i}: NaN detected in gradients for parameter '{name}'!"
                        )
                        has_nan_grad = True
                    if torch.isinf(param.grad).any():
                        logger.error(
                            f"Batch {i}: Inf detected in gradients for parameter '{name}'!"
                        )
                        has_inf_grad = True

            if has_nan_grad or has_inf_grad:
                logger.error(
                    f"Batch {i}: Model has NaN/Inf gradients after backward pass!"
                )

        # Log gradient information
        total_norm = 0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** (1.0 / 2)
        logger.debug(
            f"Batch {i}: Total gradient norm: {total_norm:.6f}, "
            f"Parameters with gradients: {param_count}"
        )

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            # More aggressive gradient clipping for numerical stability
            clip_norm = min(
                args.clip, 0.1
            )  # Use smaller clipping norm if gradients are large
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            logger.debug(f"Batch {i}: Applied gradient clipping with norm {clip_norm}")

            # Check if gradients are still too large after clipping
            total_norm_after = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_after += param_norm.item() ** 2
            total_norm_after = total_norm_after ** (1.0 / 2)

            if total_norm_after > 1.0:
                logger.warning(
                    f"Batch {i}: Gradients still large after clipping: {total_norm_after:.2f}"
                )
                # Skip this batch if gradients are still too large
                continue

        # from IPython import embed; embed()
        optimizer.step()

        if args.check_nans_and_infs:
            # Check if optimizer step introduced NaN/Inf in parameters
            has_nan_param = False
            has_inf_param = False
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    logger.error(
                        f"Batch {i}: NaN detected in parameter '{name}' after optimizer step!"
                    )
                    has_nan_param = True
                if torch.isinf(param).any():
                    logger.error(
                        f"Batch {i}: Inf detected in parameter '{name}' after optimizer step!"
                    )
                    has_inf_param = True

            if has_nan_param or has_inf_param:
                logger.error(
                    f"Batch {i}: Model has NaN/Inf parameters after optimizer step!"
                )

        log_metrics({"loss": loss.item()}, "train", global_step)

        if args.optimizer == "smart_adamax":
            current_lr = optimizer.get_lr()[0]
            log_metrics({"Learning rate": current_lr}, "train", global_step)
        else:
            current_lr = optimizer.param_groups[0]["lr"]
            log_metrics({"Learning Rate": current_lr}, "train", global_step)

        # Reduce learning rate if we're experiencing instability
        if loss.item() > 1e4 and current_lr > 1e-5:
            reduction_factor = 0.5
            new_lr = current_lr * reduction_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            logger.warning(
                f"Batch {i}: Reducing learning rate from {current_lr:.2e} to {new_lr:.2e} due to high loss"
            )
            current_lr = new_lr

        logger.debug(f"Batch {i}: Learning rate: {current_lr:.8f}")

        global_step += 1

    # Check if we have any valid batches
    if len(all_targets) == 0:
        logger.error("No valid batches were processed - all batches had Inf/NaN loss")
        return float("inf"), global_step

    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)
    all_token_ids = np.concatenate(all_token_ids)
    if args.sp_region_labels:
        all_cs = np.concatenate(all_cs)
    else:
        all_cs = None

    logger.debug(
        f"Epoch summary: Processed {len(train_data)} batches, "
        f"Total loss: {total_loss:.6f}, "
        f"Average loss per batch: {total_loss/len(train_data):.6f}"
    )
    logger.debug(
        f"Epoch summary: Targets shape: {all_targets.shape}, "
        f"Global targets shape: {all_global_targets.shape}"
    )
    logger.debug(
        f"Epoch summary: Global probs shape: {all_global_probs.shape}, "
        f"Position predictions shape: {all_pos_preds.shape}"
    )

    if args.average_per_kingdom:
        metrics = report_metrics_kingdom_averaged(
            all_global_targets,
            all_global_probs,
            all_targets,
            all_pos_preds,
            all_kingdom_ids,
            all_token_ids,
            all_cs,
            args.use_cs_tag,
        )
    else:
        metrics = report_metrics(
            all_global_targets,
            all_global_probs,
            all_targets,
            all_pos_preds,
            args.use_cs_tag,
        )
    log_metrics(metrics, "train", global_step)

    return total_loss / len(train_data), global_step


def validate_epoch(model: torch.nn.Module, val_loader: DataLoader, args) -> float:
    """Run over the validation data. Average loss over the full set."""
    model.eval()

    all_targets = []
    all_global_targets = []
    all_global_probs = []
    all_pos_preds = []
    all_kingdom_ids = []
    all_token_ids = []
    all_cs = []

    total_loss = 0
    for i, batch in enumerate(val_loader):
        if args.sp_region_labels:
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
            (
                data,
                targets,
                input_mask,
                global_targets,
                sample_weights,
                kingdom_ids,
            ) = batch
        # input_ids must stay as long integers for embedding layers
        data = data.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        input_mask = input_mask.to(device, dtype=torch.bool)
        global_targets = global_targets.to(device, dtype=torch.long)
        sample_weights = (
            sample_weights.to(device, dtype=torch.float32)
            if args.use_sample_weights
            else None
        )
        kingdom_ids = kingdom_ids.to(device, dtype=torch.long)

        with torch.no_grad():
            loss, global_probs, pos_probs, pos_preds = model(
                data,
                global_targets=global_targets,
                targets=targets if not args.sp_region_labels else None,
                targets_bitmap=targets if args.sp_region_labels else None,
                sample_weights=sample_weights,
                input_mask=input_mask,
                kingdom_ids=kingdom_ids if args.kingdom_embed_size > 0 else None,
            )

        total_loss += loss.mean().item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())
        all_token_ids.append(data.detach().cpu().numpy())
        if args.sp_region_labels:
            all_cs.append(cleavage_sites)
        else:
            all_cs.append(None)

    # Check if we have any valid batches
    if len(all_targets) == 0:
        logger.error("No valid batches were processed in validation")
        return float("inf"), {}

    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)
    all_token_ids = np.concatenate(all_token_ids)
    if args.sp_region_labels:
        all_cs = np.concatenate(all_cs)
    else:
        all_cs = None

    logger.debug(
        f"Validation summary: Processed {len(val_loader)} batches, "
        f"Total loss: {total_loss:.6f}, "
        f"Average loss per batch: {total_loss/len(val_loader):.6f}"
    )
    logger.debug(
        f"Validation summary: Targets shape: {all_targets.shape}, "
        f"Global targets shape: {all_global_targets.shape}"
    )

    if args.average_per_kingdom:
        metrics = report_metrics_kingdom_averaged(
            all_global_targets,
            all_global_probs,
            all_targets,
            all_pos_preds,
            all_kingdom_ids,
            all_token_ids,
            all_cs,
            args.use_cs_tag,
        )
    else:
        metrics = report_metrics(
            all_global_targets,
            all_global_probs,
            all_targets,
            all_pos_preds,
            args.use_cs_tag,
        )

    val_metrics = {"loss": total_loss / len(val_loader), **metrics}
    return (total_loss / len(val_loader)), val_metrics


def train_model(args: argparse.ArgumentParser):
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{args.experiment_name}_{args.test_partition}_{args.validation_partition}_{time_stamp}"

    # TODO get rid of this dirty fix once wandb works again
    global wandb
    import wandb

    if (
        wandb.run is None and not args.crossval_run
    ):  # Only initialize when there is no run yet (when importing main_training_loop to other scripts)
        wandb.init(dir=args.output_dir, name=experiment_name)
    else:
        wandb = DecoyWandb()

    # Set seed
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        seed = args.random_seed
    else:
        seed = torch.seed()

    logger.info(f"torch seed: {seed}")
    wandb.config.update({"seed": seed})

    logger.info(f"Saving to {args.output_dir}")

    # Setup Model
    logger.info(f"Loading pretrained model in {args.resume}")
    config = MODEL_DICT[args.model_architecture][0].from_pretrained(args.resume)

    # Handle xla_device attribute that may not exist in newer transformers versions
    if hasattr(config, "xla_device") and config.xla_device:
        setattr(config, "xla_device", False)

    setattr(config, "num_labels", args.num_seq_labels)
    setattr(config, "num_global_labels", args.num_global_labels)

    setattr(config, "lm_output_dropout", args.lm_output_dropout)
    setattr(config, "lm_output_position_dropout", args.lm_output_position_dropout)
    setattr(config, "crf_scaling_factor", args.crf_scaling_factor)
    setattr(
        config, "use_large_crf", True
    )  # legacy, parameter is used in evaluation scripts. Ensures choice of right CS states.

    if args.sp_region_labels:
        setattr(config, "use_region_labels", True)

    if args.kingdom_embed_size > 0:
        setattr(config, "use_kingdom_id", True)
        setattr(config, "kingdom_embed_size", args.kingdom_embed_size)

    # Note: CRF constraints were an experimental feature that is not part of the
    # intended SignalP 6.0 architecture described in the paper. The model should
    # learn valid transitions naturally from the training data.

    # setattr(config, 'gradient_checkpointing', True) #hardcoded when working with 256aa data
    if args.kingdom_as_token:
        setattr(
            config, "kingdom_id_as_token", True
        )  # model needs to know that token at pos 1 needs to be removed for CRF

    if args.global_label_as_input:
        setattr(config, "type_id_as_token", True)

    if args.remove_top_layers > 0:
        # num_hidden_layers if bert
        n_layers = (
            config.num_hidden_layers
            if args.model_architecture == "bert_prottrans"
            else config.n_layer
        )
        if args.remove_top_layers > n_layers:
            logger.warning(f"Trying to remove more layers than there are: {n_layers}")
            args.remove_top_layers = n_layers

        setattr(
            config,
            (
                "num_hidden_layers"
                if args.model_architecture == "bert_prottrans"
                else "n_layer"
            ),
            n_layers - args.remove_top_layers,
        )

    # Load tokenizer first to determine vocabulary size
    if args.kingdom_as_token:
        logger.info(
            "Using kingdom IDs as word in sequence, extending embedding layer of pretrained model."
        )
        tokenizer = TOKENIZER_DICT[args.model_architecture][0].from_pretrained(
            "data/tokenizer", do_lower_case=False
        )
        # Update config to match the custom tokenizer vocabulary size BEFORE loading the model
        setattr(config, "vocab_size", tokenizer.tokenizer.vocab_size)
        logger.info(f"Updated config vocab_size to: {config.vocab_size}")
    else:
        tokenizer = TOKENIZER_DICT[args.model_architecture][0].from_pretrained(
            TOKENIZER_DICT[args.model_architecture][1], do_lower_case=False
        )

    # Now load the model with the correct configuration
    # Create the model from scratch first to avoid CRF initialization issues
    logger.info("Creating model from scratch to avoid CRF initialization issues...")
    model = MODEL_DICT[args.model_architecture][1](config)

    # Then load only the BERT weights from the pretrained checkpoint
    logger.info(f"Loading BERT weights from {args.resume}...")
    try:
        # Load the pretrained model temporarily to get the state dict
        temp_model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.resume)

        # Get the state dict and filter out non-BERT parameters
        state_dict = temp_model.state_dict()
        bert_state_dict = {}
        for key, value in state_dict.items():
            # Only load BERT-related weights, skip CRF and other newly initialized layers
            if key.startswith("bert.") or key.startswith("embeddings."):
                bert_state_dict[key] = value

        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(
            bert_state_dict, strict=False
        )
        logger.info(f"Loaded BERT weights successfully")
        if missing_keys:
            logger.info(f"Missing keys (expected): {missing_keys}")
        if unexpected_keys:
            logger.info(f"Unexpected keys (ignored): {unexpected_keys}")

        # Clean up the temporary model
        del temp_model

    except Exception as e:
        logger.warning(f"Failed to load BERT weights: {e}")
        logger.info("Continuing with random initialization...")

    # Note: CRF constraints have been removed as they are not part of the intended
    # SignalP 6.0 architecture. The model learns valid transitions naturally.

    # Resize embeddings if using custom tokenizer
    if args.kingdom_as_token:
        model.resize_token_embeddings(tokenizer.tokenizer.vocab_size)
        logger.info(
            f"Resized model embeddings to match tokenizer vocab size: {tokenizer.tokenizer.vocab_size}"
        )

    # setup data
    val_id = args.validation_partition
    test_id = args.test_partition
    train_ids = [0, 1, 2]  # ,3,4]
    train_ids.remove(val_id)
    train_ids.remove(test_id)
    logger.info(f"Training on {train_ids}, validating on {val_id}")

    kingdoms = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"]

    if args.sp_region_labels:
        train_data = RegionCRFDataset(
            args.data,
            args.sample_weights,
            tokenizer=tokenizer,
            partition_id=train_ids,
            kingdom_id=kingdoms,
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=args.positive_samples_weight,
            make_cs_state=args.use_cs_tag,
            add_global_label=args.global_label_as_input,
            augment_data_paths=[args.additional_train_set],
        )
        val_data = RegionCRFDataset(
            args.data,
            args.sample_weights,
            tokenizer=tokenizer,
            partition_id=[val_id],
            kingdom_id=kingdoms,
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=args.positive_samples_weight,
            make_cs_state=args.use_cs_tag,
            add_global_label=args.global_label_as_input,
        )
        logger.info("Using labels for SP region prediction.")
    else:
        train_data = LargeCRFPartitionDataset(
            args.data,
            args.sample_weights,
            tokenizer=tokenizer,
            partition_id=train_ids,
            kingdom_id=kingdoms,
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=args.positive_samples_weight,
            make_cs_state=args.use_cs_tag,
        )
        val_data = LargeCRFPartitionDataset(
            args.data,
            args.sample_weights,
            tokenizer=tokenizer,
            partition_id=[val_id],
            kingdom_id=kingdoms,
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=args.positive_samples_weight,
            make_cs_state=args.use_cs_tag,
        )
    logger.info(
        f"{len(train_data)} training sequences, {len(val_data)} validation sequences."
    )

    logger.debug(f"Training data type: {type(train_data).__name__}")
    logger.debug(f"Validation data type: {type(val_data).__name__}")
    if hasattr(train_data, "sample_weights"):
        logger.debug(
            f"Training data has sample weights: {train_data.sample_weights is not None}"
        )
    if hasattr(train_data, "balanced_sampling_weights"):
        logger.debug(
            f"Training data has balanced sampling weights: {train_data.balanced_sampling_weights is not None}"
        )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=train_data.collate_fn,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, collate_fn=train_data.collate_fn
    )

    if args.use_random_weighted_sampling:
        sampler = WeightedRandomSampler(
            train_data.sample_weights, len(train_data), replacement=False
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            collate_fn=train_data.collate_fn,
            sampler=sampler,
        )
    elif args.use_weighted_kingdom_sampling:
        sampler = WeightedRandomSampler(
            train_data.balanced_sampling_weights, len(train_data), replacement=False
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            collate_fn=train_data.collate_fn,
            sampler=sampler,
        )
        logger.info(
            f"Using kingdom-balanced oversampling. Sum of all sampling weights = {sum(train_data.balanced_sampling_weights)}"
        )

    logger.info(f"Data loaded. One epoch = {len(train_loader)} batches.")

    logger.debug(f"Training data loader batch size: {args.batch_size}")
    logger.debug(f"Training data loader shuffle: True")
    logger.debug(f"Validation data loader batch size: {args.batch_size}")
    if args.use_random_weighted_sampling:
        logger.debug("Using random weighted sampling")
    elif args.use_weighted_kingdom_sampling:
        logger.debug("Using weighted kingdom sampling")
    else:
        logger.debug("Using standard sampling")

    # set up wandb logging, login and project id from commandline vars
    wandb.config.update(args)
    wandb.config.update({"git commit ID": GIT_HASH})
    wandb.config.update(model.config.to_dict())
    # TODO uncomment as soon as w&b fixes the bug on their end.
    # wandb.watch(model)
    logger.info(f"Logging experiment as {experiment_name} to wandb/tensorboard")
    logger.info(f"Saving checkpoints at {args.output_dir}")

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    elif args.optimizer == "adamax":
        optimizer = torch.optim.Adamax(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    elif args.optimizer == "smart_adamax":
        t_total = len(train_loader) * args.epochs
        optimizer = Adamax(
            model.parameters(),
            lr=args.lr,
            warmup=0.1,
            t_total=t_total,
            schedule="warmup_linear",
            betas=(0.9, 0.999),
            weight_decay=args.wdecay,
            max_grad_norm=1,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    logger.debug(f"Optimizer: {args.optimizer}")
    logger.debug(f"Learning rate: {args.lr}")
    logger.debug(f"Weight decay: {args.wdecay}")

    # Check if learning rate might be too high
    if args.lr > 0.01:
        logger.warning(
            f"Learning rate {args.lr} might be too high and could cause instability"
        )
    if args.lr > 0.1:
        logger.error(
            f"Learning rate {args.lr} is very high and likely to cause NaN loss"
        )

    if args.optimizer == "smart_adamax":
        logger.debug(f"Smart Adamax warmup: 0.1, t_total: {t_total}")

    model.to(device)
    logger.info("Model set up!")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} trainable parameters")

    # Check model parameters for NaN/Inf before starting training
    if args.check_nans_and_infs:
        check_model_parameters(model=model, stage="training")

    # Additional model diagnostics
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(
        f"Model base model prefix: {getattr(model, 'base_model_prefix', 'N/A')}"
    )
    logger.info(f"Model config: {type(model.config).__name__}")
    logger.info(f"Model has kingdom embedding: {hasattr(model, 'kingdom_embedding')}")
    if hasattr(model, "kingdom_embedding"):
        logger.info(f"Kingdom embedding shape: {model.kingdom_embedding.weight.shape}")
        logger.info(
            f"Kingdom embedding device: {model.kingdom_embedding.weight.device}"
        )

    logger.info(f"Model device: {device}")
    logger.info(f"Model architecture: {args.model_architecture}")
    logger.info(f"Number of sequence labels: {args.num_seq_labels}")
    logger.info(f"Number of global labels: {args.num_global_labels}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient clipping: {args.clip}")

    # Log model configuration details
    logger.info(f"Model config - hidden size: {model.config.hidden_size}")
    logger.info(f"Model config - num hidden layers: {model.config.num_hidden_layers}")
    logger.info(f"Model config - intermediate size: {model.config.intermediate_size}")
    logger.info(
        f"Model config - num attention heads: {model.config.num_attention_heads}"
    )
    logger.info(f"Model config - dropout: {model.config.hidden_dropout_prob}")
    logger.info(
        f"Model config - attention dropout: {model.config.attention_probs_dropout_prob}"
    )

    # Check for potential configuration issues
    if model.config.hidden_dropout_prob > 0.5:
        logger.warning(
            f"High dropout rate {model.config.hidden_dropout_prob} might cause instability"
        )
    if model.config.attention_probs_dropout_prob > 0.5:
        logger.warning(
            f"High attention dropout rate {model.config.attention_probs_dropout_prob} might cause instability"
        )

    logger.debug(f"Model training mode: {model.training}")

    # keep track of best loss
    stored_loss = 100000000
    learning_rate_steps = 0
    num_epochs_no_improvement = 0
    global_step = 0
    best_mcc_sum = 0
    best_mcc_global = 0
    best_mcc_cs = 0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Starting epoch {epoch}")
        logger.debug(f"Epoch {epoch}: Training on {len(train_loader)} batches")
        # Check model parameters at start of each epoch
        if args.check_nans_and_infs:
            check_model_parameters(model, f"epoch {epoch}")

        epoch_train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            args,
            global_step,
        )

        logger.info(
            f"Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps"
        )
        logger.debug(f"Epoch {epoch}: Starting validation on {len(val_loader)} batches")
        val_loss, val_metrics = validate_epoch(model=model, val_loader=val_loader, args=args)
        log_metrics(val_metrics, "val", global_step)
        logger.info(
            f"Validation: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}. Epochs without improvement: {num_epochs_no_improvement}. lr step {learning_rate_steps}"
        )

        mcc_sum = val_metrics["Detection MCC"] + val_metrics["CS MCC"]

        logger.debug(f"Epoch {epoch}: Validation loss: {val_loss:.6f}")
        logger.debug(
            f"Epoch {epoch}: Detection MCC: {val_metrics['Detection MCC']:.6f}"
        )
        logger.debug(f"Epoch {epoch}: CS MCC: {val_metrics['CS MCC']:.6f}")
        logger.debug(f"Epoch {epoch}: MCC sum: {mcc_sum:.6f}")
        logger.debug(f"Epoch {epoch}: Best MCC sum so far: {best_mcc_sum:.6f}")
        log_metrics({"MCC Sum": mcc_sum}, "val", global_step)

        if mcc_sum > best_mcc_sum:
            best_mcc_sum = mcc_sum
            best_mcc_global = val_metrics["Detection MCC"]
            best_mcc_cs = val_metrics["CS MCC"]
            num_epochs_no_improvement = 0

            model.save_pretrained(args.output_dir)
            logger.info(
                f'New best model with loss {val_loss},MCC Sum {mcc_sum} MCC global {val_metrics["Detection MCC"]}, MCC seq {val_metrics["CS MCC"]}, Saving model, training step {global_step}'
            )

            logger.info(f"Epoch {epoch}: Saved new best model to {args.output_dir}")
            logger.info(f"Epoch {epoch}: Previous best MCC sum: {best_mcc_sum:.6f}")
            logger.info(f"Epoch {epoch}: New best MCC sum: {mcc_sum:.6f}")

        else:
            num_epochs_no_improvement += 1

        # when cross-validating, check that the seed is working for region detection
        if args.crossval_run and epoch == 1:
            # small length in first epoch = bad seed.
            if val_metrics["Average length n 1"] <= 4:
                print("Bad seed for region tagging.")
                run_completed = False
                return best_mcc_global, best_mcc_cs, run_completed

    logger.info(f"Epoch {epoch}, epoch limit reached. Training complete")
    logger.info(
        f"Best: MCC Sum {best_mcc_sum}, Detection {best_mcc_global}, CS {best_mcc_cs}"
    )
    log_metrics(
        {
            "Best MCC Detection": best_mcc_global,
            "Best MCC CS": best_mcc_cs,
            "Best MCC sum": best_mcc_sum,
        },
        "val",
        global_step,
    )

    # reload best checkpoint
    model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.output_dir)
    ds = RegionCRFDataset(
        args.data,
        args.sample_weights,
        tokenizer=tokenizer,
        partition_id=[test_id],
        kingdom_id=kingdoms,
        add_special_tokens=True,
        return_kingdom_ids=True,
        positive_samples_weight=args.positive_samples_weight,
        make_cs_state=args.use_cs_tag,
        add_global_label=args.global_label_as_input,
    )
    dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=ds.collate_fn, batch_size=80
    )
    logger.info("Evaluating final metrics on test set")
    metrics = get_metrics_multistate(model, dataloader)
    logger.info("Evaluating final metrics on validation set")
    val_metrics = get_metrics_multistate(model, val_loader)

    if args.crossval_run or args.log_all_final_metrics:
        log_metrics(metrics, "test", global_step)
        log_metrics(val_metrics, "best_val", global_step)
    logger.info(metrics)
    logger.info("Validation set")
    logger.info(val_metrics)

    # Check if metrics are empty
    if not metrics and not val_metrics:
        logger.info(
            "No metrics to display - both test and validation metrics are empty"
        )
    else:
        df = pd.DataFrame.from_dict([metrics, val_metrics]).T
        df.columns = ["test", "val"]
        df.index = df.index.str.split("_", expand=True)
        pd.set_option("display.max_rows", None)
        print(df.sort_index())

    run_completed = True
    return best_mcc_global, best_mcc_cs, run_completed  # best_mcc_sum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Bert-CRF model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/data/train_set.fasta",
        help="location of the data corpus. Expects test, train and valid .fasta",
    )
    parser.add_argument(
        "--sample_weights",
        type=str,
        default=None,
        help="path to .csv file with the weights for each sample",
    )
    parser.add_argument(
        "--test_partition",
        type=int,
        default=0,
        help="partition that will not be used in this training run",
    )
    parser.add_argument(
        "--validation_partition",
        type=int,
        default=1,
        help="partition that will be used for validation in this training run",
    )

    # args relating to training strategy.
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=8000, help="upper epoch limit")

    parser.add_argument(
        "--batch_size", type=int, default=80, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--wdecay",
        type=float,
        default=1.2e-6,
        help="weight decay applied to all weights",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer to use (sgd, adam, adamax)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_run",
        help="path to save logs and trained model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="Rostlab/prot_bert",
        help="path of model to resume (directory containing .bin and config.json, or HF model)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="BERT-CRF",
        help="experiment name for logging",
    )
    parser.add_argument(
        "--crossval_run",
        action="store_true",
        help="override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.",
    )
    parser.add_argument(
        "--log_all_final_metrics",
        action="store_true",
        help="log all final test/val metrics to w&b",
    )
    parser.add_argument(
        "--check_nans_and_infs",
        action="store_true",
        help="check for NaN/Inf values in model inputs,outputs and parameters",
    )

    parser.add_argument("--num_seq_labels", type=int, default=37)
    parser.add_argument("--num_global_labels", type=int, default=6)
    parser.add_argument(
        "--global_label_as_input",
        action="store_true",
        help="Add the global label to the input sequence (only predict CS given a known label)",
    )

    parser.add_argument(
        "--region_regularization_alpha",
        type=float,
        default=0,
        help="multiplication factor for the region similarity regularization term",
    )
    parser.add_argument(
        "--lm_output_dropout",
        type=float,
        default=0.1,
        help="dropout applied to LM output",
    )
    parser.add_argument(
        "--lm_output_position_dropout",
        type=float,
        default=0.1,
        help="dropout applied to LM output, drops full hidden states from sequence",
    )
    parser.add_argument(
        "--use_sample_weights",
        action="store_true",
        help="Use sample weights to rescale loss per sample",
    )
    parser.add_argument(
        "--use_random_weighted_sampling",
        action="store_true",
        help="use sample weights to load random samples as minibatches according to weights",
    )
    parser.add_argument(
        "--positive_samples_weight",
        type=float,
        default=None,
        help="Scaling factor for positive samples loss, e.g. 1.5. Needs --use_sample_weights flag in addition.",
    )
    parser.add_argument(
        "--average_per_kingdom",
        action="store_true",
        help="Average MCCs per kingdom instead of overall computatition",
    )
    parser.add_argument(
        "--crf_scaling_factor",
        type=float,
        default=1.0,
        help="Scale CRF NLL by this before adding to global label loss",
    )
    parser.add_argument(
        "--use_weighted_kingdom_sampling",
        action="store_true",
        help="upsample all kingdoms to equal probabilities",
    )
    parser.add_argument(
        "--random_seed", type=int, default=None, help="random seed for torch."
    )
    parser.add_argument(
        "--additional_train_set",
        type=str,
        default=None,
        help="Additional samples to train on",
    )

    # args for model architecture
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="bert_prottrans",
        help="which model architecture the checkpoint is for",
    )
    parser.add_argument(
        "--remove_top_layers",
        type=int,
        default=0,
        help="How many layers to remove from the top of the LM.",
    )
    parser.add_argument(
        "--kingdom_embed_size",
        type=int,
        default=0,
        help="If >0, embed kingdom ids to N and concatenate with LM hidden states before CRF.",
    )
    parser.add_argument(
        "--use_cs_tag",
        action="store_true",
        help="Replace last token of SP with C for cleavage site",
    )
    parser.add_argument(
        "--kingdom_as_token",
        action="store_true",
        help="Kingdom ID is first token in the sequence",
    )
    parser.add_argument(
        "--sp_region_labels",
        action="store_true",
        help="Use labels for n,h,c regions of SPs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging for detailed training information",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # make unique output dir in output dir
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    full_name = "_".join(
        [
            args.experiment_name,
            "test",
            str(args.test_partition),
            "validation",
            str(args.validation_partition),
            time_stamp,
        ]
    )

    if args.crossval_run == True:
        full_name = "_".join(
            [
                args.experiment_name,
                "test",
                str(args.test_partition),
                "validation",
                str(args.validation_partition),
            ]
        )

    args.output_dir = os.path.join(args.output_dir, full_name)
    os.makedirs(args.output_dir, exist_ok=True)
    global logger
    logger = setup_logger(
        Path(args.output_dir).joinpath("log.txt"),
        verbosity=logging.DEBUG if args.verbose else logging.INFO,
    )

    train_model(args)


if __name__ == "__main__":
    sys.exit(main())
