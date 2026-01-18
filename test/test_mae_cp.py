"""
Test script for evaluating MAE-CP (Masked Autoencoder Continue Pretraining) models.

This script evaluates MAE models (both baseline pretrained and CP-finetuned) using
the same evaluation pipeline as test_dinov3.py for consistency.

Metrics evaluated:
- k-NN classification (accuracy, F1, ROC AUC)
- Linear probe (accuracy, F1, ROC AUC)
- K-Means clustering (ARI, NMI)
- RankMe (effective rank of representations)
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent))

import stable_pretraining as spt
from core.mae_cp_dataset import MAE_CPDataset, create_mae_cp_transforms
from core.load_mae_weights import load_pretrained_mae_weights
from core.cp_datasets import DATASET_STATS
from metrics import zero_shot_eval  # Use shared evaluation functions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MAEWrapper(nn.Module):
    """
    Wrapper around MAE model to provide a consistent interface with DINOWrapper.
    
    This makes MAE compatible with the zero_shot_eval function from metrics.py.
    """
    def __init__(self, mae_model):
        super().__init__()
        self.model = mae_model
    
    def forward(self, x):
        """
        Forward pass that returns CLS token embedding.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            CLS token features [B, hidden_dim]
        """
        # Forward through MAE encoder
        latent, _, _ = self.model.forward_encoder(x, mask_ratio=0.0)
        
        # Return CLS token (first token)
        return latent[:, 0]


def load_mae_model(
    checkpoint_path: Optional[str] = None,
    model_size: str = "base",
    device: str = "cuda",
    img_size: int = 224,
) -> nn.Module:
    """
    Load MAE model from checkpoint or pretrained weights.
    
    Args:
        checkpoint_path: Path to MAE-CP checkpoint (None for baseline)
        model_size: Model size ('base', 'large', 'huge')
        device: Device to load model on
        img_size: Input image size
        
    Returns:
        Wrapped MAE model
    """
    # Create MAE backbone
    if model_size == "base":
        backbone = spt.backbone.mae.vit_base_patch16_dec512d8b(
            img_size=img_size,
            norm_pix_loss=False,
        )
    elif model_size == "large":
        backbone = spt.backbone.mae.vit_large_patch16_dec512d8b(
            img_size=img_size,
            norm_pix_loss=False,
        )
    elif model_size == "huge":
        backbone = spt.backbone.mae.vit_huge_patch14_dec512d8b(
            img_size=img_size,
            norm_pix_loss=False,
        )
    else:
        raise ValueError(f"Invalid model_size: {model_size}")
    
    if checkpoint_path is None:
        # Load baseline pretrained MAE from HuggingFace
        logger.info("Loading baseline pretrained MAE from HuggingFace")
        pretrained_source = f"facebook/vit-mae-{model_size}"
        load_pretrained_mae_weights(backbone, source=pretrained_source, strict=False)
    else:
        # Load MAE-CP checkpoint
        logger.info(f"Loading MAE-CP checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract backbone weights from stable-pretraining checkpoint
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Filter backbone weights
            backbone_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    backbone_state_dict[k.replace("backbone.", "")] = v
            
            # Load weights
            missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
            if missing:
                logger.info(f"Missing keys: {len(missing)} keys")
            if unexpected:
                logger.info(f"Unexpected keys: {len(unexpected)} keys")
        else:
            raise ValueError("Invalid checkpoint format")
    
    # Wrap MAE model
    wrapped_model = MAEWrapper(backbone)
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    
    return wrapped_model


def create_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int = 256,
    limit_data: Optional[int] = None,
    num_workers: int = 4,
):
    """
    Create train and test dataloaders for evaluation.
    
    Args:
        dataset_name: Name of dataset
        data_root: Root directory for data
        batch_size: Batch size
        limit_data: Limit training data size (None = use all)
        num_workers: Number of dataloader workers
        
    Returns:
        train_loader, test_loader, dataset_info
    """
    # Get dataset stats
    dataset_stats = DATASET_STATS.get(dataset_name)
    if dataset_stats is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create transforms
    transform = create_mae_cp_transforms(
        input_size=dataset_stats["input_size"],
        is_rgb=dataset_stats["is_rgb"],
        is_train=False,  # Use test-time transforms for evaluation
    )
    
    # Create datasets
    train_dataset = MAE_CPDataset(
        dataset_name=dataset_name,
        root=data_root,
        split="train",
        limit_data=limit_data if limit_data else -1,
        transform=transform,
    )
    
    test_dataset = MAE_CPDataset(
        dataset_name=dataset_name,
        root=data_root,
        split="test",
        limit_data=-1,  # Always use full test set
        transform=transform,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    dataset_info = {
        "num_classes": dataset_stats["num_classes"],
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
    }
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Train size: {dataset_info['train_size']}")
    logger.info(f"Test size: {dataset_info['test_size']}")
    logger.info(f"Num classes: {dataset_info['num_classes']}")
    
    return train_loader, test_loader, dataset_info


def test_mae_cp(
    checkpoint_path: Optional[str],
    dataset_name: str,
    data_root: str,
    model_size: str = "base",
    batch_size: int = 256,
    limit_data: Optional[int] = None,
    probe_lr: float = 1e-3,
    probe_steps: int = 10000,
    device: str = "cuda",
):
    """
    Test MAE-CP model using the same evaluation pipeline as test_dinov3.py.
    
    Args:
        checkpoint_path: Path to checkpoint (None for baseline MAE)
        dataset_name: Dataset name
        data_root: Data root directory
        model_size: Model size
        batch_size: Batch size for evaluation
        limit_data: Limit training data for probe (None = use all)
        probe_lr: Learning rate for linear probe
        probe_steps: Training steps for linear probe
        device: Device to use
        
    Returns:
        dict: Evaluation results
    """
    logger.info("=" * 70)
    logger.info("MAE-CP MODEL EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path if checkpoint_path else 'Baseline (HuggingFace)'}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model size: {model_size}")
    logger.info(f"Device: {device}")
    
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    
    # Load model
    model = load_mae_model(
        checkpoint_path=checkpoint_path,
        model_size=model_size,
        device=device,
    )
    
    # Create dataloaders
    train_loader, test_loader, dataset_info = create_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=batch_size,
        limit_data=limit_data,
    )
    
    # Run evaluation using shared evaluation function
    logger.info("\n" + "=" * 70)
    logger.info("ZERO-SHOT EVALUATION")
    logger.info("=" * 70)
    
    results = zero_shot_eval(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        probe_lr=probe_lr,
        probe_steps=probe_steps,
        store_embeddings=False,
    )
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"k-NN Accuracy:       {results['knn_acc']:.4f}")
    logger.info(f"k-NN F1:             {results['knn_f1']:.4f}")
    logger.info(f"k-NN ROC AUC:        {results['knn_roc_auc']:.4f}")
    logger.info(f"Linear Accuracy:     {results['linear_acc']:.4f}")
    logger.info(f"Linear F1:           {results['linear_f1']:.4f}")
    logger.info(f"Linear ROC AUC:      {results['linear_roc_auc']:.4f}")
    logger.info(f"K-Means ARI:         {results['kmeans_ari']:.4f}")
    logger.info(f"K-Means NMI:         {results['kmeans_nmi']:.4f}")
    logger.info(f"RankMe:              {results['rankme']:.4f}")
    logger.info("=" * 70)
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test MAE-CP model using consistent evaluation pipeline"
    )
    
    # Model arguments
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to MAE-CP checkpoint (None for baseline MAE)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["base", "large", "huge"],
        help="MAE model size",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., bloodmnist, pathmnist)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--limit-data",
        type=int,
        default=None,
        help="Limit training data for probe (None = use all)",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--probe-lr",
        type=float,
        default=1e-3,
        help="Learning rate for linear probe",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=10000,
        help="Training steps for linear probe",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Run evaluation
    results = test_mae_cp(
        checkpoint_path=args.checkpoint_path,
        dataset_name=args.dataset,
        data_root=args.data_root,
        model_size=args.model_size,
        batch_size=args.batch_size,
        limit_data=args.limit_data,
        probe_lr=args.probe_lr,
        probe_steps=args.probe_steps,
        device=args.device,
    )
    
    return results


if __name__ == "__main__":
    main()
