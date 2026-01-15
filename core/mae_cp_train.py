"""
MAE Continue Pretraining (MAE-CP) training script using stable-pretraining.

This script trains MAE on domain-specific datasets for continue pretraining,
similar to DINOv3-CP but using MAE's reconstruction objective.
"""

import torch
import lightning as pl
import torchmetrics
import logging

import stable_pretraining as spt
from stable_pretraining.data import transforms

from pathlib import Path
from mae_cp_dataset import MAE_CPDataset
from load_mae_weights import load_pretrained_mae_weights
from infinite_sampler import create_infinite_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mae_cp_transforms(dataset_stats):
    """
    Create transforms for MAE-CP training.
    
    Args:
        dataset_stats: Dictionary with 'mean', 'std', 'input_size', 'is_rgb'
        
    Returns:
        train_transform, val_transform
    """
    mean = dataset_stats.get("mean", [0.485, 0.456, 0.406])
    std = dataset_stats.get("std", [0.229, 0.224, 0.225])
    input_size = dataset_stats.get("input_size", 224)
    is_rgb = dataset_stats.get("is_rgb", True)
    
    # For grayscale images, replicate single-channel stats
    if not is_rgb:
        if len(mean) == 1:
            mean = [mean[0]] * 3
        if len(std) == 1:
            std = [std[0]] * 3
    
    # Training transform with augmentation
    train_transform = transforms.Compose(
        transforms.RGB(),  # Ensure 3 channels
        transforms.RandomResizedCrop((input_size, input_size), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToImage(mean=mean, std=std),
    )
    
    # Validation transform without augmentation
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((int(input_size * 1.14), int(input_size * 1.14))),
        transforms.CenterCrop((input_size, input_size)),
        transforms.ToImage(mean=mean, std=std),
    )
    
    return train_transform, val_transform


def mae_cp_forward(self, batch, stage):
    """
    Forward function for MAE-CP.
    
    Args:
        self: Module instance
        batch: Dictionary with 'image' and 'label' keys
        stage: Training stage ('train', 'val', 'test')
        
    Returns:
        Dictionary with 'embedding', 'loss' (if training)
    """
    out = {}
    
    # MAE forward pass
    latent, pred, mask = self.backbone(batch["image"])
    
    # Extract CLS token as embedding for probing
    out["embedding"] = latent[:, 0]  # [B, hidden_dim]
    
    # Add label for probing callbacks
    if "label" in batch:
        out["label"] = batch["label"]
    
    # Compute reconstruction loss during training
    if self.training:
        # Patchify original image
        target = self.backbone.patchify(batch["image"])
        
        # Compute MAE loss on masked patches only
        loss = spt.losses.mae(
            target=target,
            pred=pred,
            mask=mask,
            norm_pix_loss=self.hparams.get("norm_pix_loss", False),
        )
        
        out["loss"] = loss
        self.log(
            f"{stage}/loss",
            out["loss"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
    
    return out


def train_mae_cp(
    # Dataset parameters
    dataset_name: str = "bloodmnist",
    data_root: str = "/root/data",
    limit_data: int = None,
    
    # Model parameters
    model_size: str = "base",  # 'base', 'large', 'huge'
    pretrained: bool = True,
    pretrained_source: str = "facebook/vit-mae-base",
    mask_ratio: float = 0.75,
    norm_pix_loss: bool = False,
    
    # Training parameters
    batch_size: int = 256,
    epochs: int = 100,
    steps_per_epoch: int = 16,  # OFFICIAL_EPOCH_LENGTH (like DINOv3-CP)
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    
    # Hardware parameters
    num_workers: int = 8,
    precision: str = "16-mixed",
    devices: int = 1,
    
    # Output parameters
    output_dir: str = "/root/output/mae_cp",
    exp_name: str = None,
    
    # Monitoring parameters
    use_wandb: bool = False,
    wandb_project: str = "mae-cp",
):
    """
    Train MAE-CP on a specific dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'bloodmnist', 'food101')
        data_root: Root directory containing datasets
        limit_data: Limit number of training samples (for few-shot)
        model_size: MAE model size ('base', 'large', 'huge')
        pretrained: Whether to load pretrained weights
        pretrained_source: Source of pretrained weights (HF model or checkpoint path)
        mask_ratio: Ratio of patches to mask (0.75 = 75%)
        norm_pix_loss: Whether to normalize pixel values in loss
        batch_size: Training batch size
        epochs: Number of training epochs
        steps_per_epoch: Number of steps per epoch (like DINOv3's OFFICIAL_EPOCH_LENGTH)
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Number of warmup epochs
        num_workers: Number of data loading workers
        precision: Training precision ('32', '16-mixed', 'bf16-mixed')
        devices: Number of GPUs
        output_dir: Output directory for checkpoints and logs
        exp_name: Experiment name (default: {dataset_name}_{model_size}_{limit_data})
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
    """
    # Set experiment name
    if exp_name is None:
        limit_str = f"_{limit_data}" if limit_data else "_full"
        exp_name = f"{dataset_name}_{model_size}{limit_str}"
    
    # Calculate training steps (like DINOv3-CP)
    max_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    logger.info(f"Starting MAE-CP training: {exp_name}")
    logger.info(f"Training configuration:")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Steps per epoch: {steps_per_epoch}")
    logger.info(f"  - Total steps: {max_steps}")
    logger.info(f"  - Warmup steps: {warmup_steps}")
    logger.info(f"  - Batch size: {batch_size}")
    
    # Create hierarchical output directory: output_dir/dataset_name/model_size/limit_data
    limit_str = str(limit_data) if limit_data else "full"
    output_path = Path(output_dir) / dataset_name / model_size / limit_str
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure data root directory exists
    Path(data_root).mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = MAE_CPDataset(
        dataset_name=dataset_name,
        root=data_root,
        split="TRAIN",
        limit_data=limit_data,
    )
    
    dataset_stats = dataset.get_stats()
    dataset_size = len(dataset)
    logger.info(f"Dataset: {dataset_name}, Size: {dataset_size}, Stats: {dataset_stats}")
    
    # Check if dataset is smaller than batch size (common in few-shot)
    if dataset_size < batch_size:
        logger.warning(
            f"Dataset size ({dataset_size}) < Batch size ({batch_size})! "
            f"Using InfiniteSampler to repeat samples and fill batches."
        )
        samples_per_epoch = steps_per_epoch * batch_size
        repetitions_per_epoch = samples_per_epoch / dataset_size
        logger.info(
            f"Each sample will be seen ~{repetitions_per_epoch:.1f} times per epoch "
            f"(~{repetitions_per_epoch * epochs:.0f} times total)"
        )
    
    # Create transforms
    train_transform, val_transform = create_mae_cp_transforms(dataset_stats)
    
    # Create training dataset
    train_dataset = MAE_CPDataset(
        dataset_name=dataset_name,
        root=data_root,
        split="TRAIN",
        limit_data=limit_data,
        transform=train_transform,
    )
    
    # Create training dataloader with InfiniteSampler
    # This allows training with fixed steps regardless of dataset size
    train_loader = create_infinite_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=0,  # For reproducibility
        rank=0,  # Single GPU for now (TODO: add distributed support)
        world_size=1,
        drop_last=True,
        pin_memory=True,
    )
    
    logger.info(f"Created infinite dataloader for training")
    
    # Validation loader (if available)
    try:
        val_dataset = MAE_CPDataset(
            dataset_name=dataset_name,
            root=data_root,
            split="VAL",
            transform=val_transform,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    except Exception as e:
        logger.warning(f"No validation set available: {e}")
        val_loader = None
    
    data = spt.data.DataModule(train=train_loader, val=val_loader)
    
    # Create MAE backbone
    if model_size == "base":
        backbone = spt.backbone.mae.vit_base_patch16_dec512d8b(
            img_size=dataset_stats.get("input_size", 224),
            norm_pix_loss=norm_pix_loss,
        )
        hidden_dim = 768
    elif model_size == "large":
        backbone = spt.backbone.mae.vit_large_patch16_dec512d8b(
            img_size=dataset_stats.get("input_size", 224),
            norm_pix_loss=norm_pix_loss,
        )
        hidden_dim = 1024
    elif model_size == "huge":
        backbone = spt.backbone.mae.vit_huge_patch14_dec512d8b(
            img_size=dataset_stats.get("input_size", 224),
            norm_pix_loss=norm_pix_loss,
        )
        hidden_dim = 1280
    else:
        raise ValueError(f"Invalid model_size: {model_size}")
    
    logger.info(f"Created MAE-{model_size.upper()} model")
    
    # Load pretrained weights
    if pretrained:
        logger.info(f"Loading pretrained weights from {pretrained_source}")
        load_pretrained_mae_weights(
            backbone,
            source=pretrained_source,
            strict=False,
        )
    
    # Get number of classes for probing
    sample = train_dataset[0]
    # Convert labels to scalars (handle numpy arrays)
    labels = []
    for i in range(min(1000, len(train_dataset))):
        label = train_dataset[i]["label"]
        # Handle numpy array labels
        if hasattr(label, 'item'):
            label = label.item()
        elif hasattr(label, '__len__') and len(label) == 1:
            label = label[0]
        labels.append(label)
    num_classes = len(set(labels))
    logger.info(f"Detected {num_classes} classes")
    
    # Create module
    module = spt.Module(
        backbone=backbone,
        forward=mae_cp_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.95),
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "warmup_epochs": warmup_epochs,
                "max_epochs": epochs,
            },
            "interval": "step",  # Step-based scheduling (not epoch-based)
        },
        # Store hyperparameters for forward function
        norm_pix_loss=norm_pix_loss,
        mask_ratio=mask_ratio,
    )
    
    # Create online linear probe for monitoring
    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(hidden_dim, num_classes),
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
        },
        optimizer={"type": "SGD", "lr": 0.1, "momentum": 0.9},
    )
    
    # Create callbacks
    callbacks = [
        linear_probe,
        spt.callbacks.RankMe(
            name="rankme", 
            target="embedding",
            queue_length=8192,
            target_shape=hidden_dim,
        ),
    ]
    
    # Create logger
    if use_wandb:
        from lightning.pytorch.loggers import WandbLogger
        pl_logger = WandbLogger(
            project=wandb_project,
            name=exp_name,
            save_dir=str(output_path),
        )
    else:
        from lightning.pytorch.loggers import CSVLogger
        # Use output_path as save_dir, no name to avoid extra nesting
        pl_logger = CSVLogger(save_dir=str(output_path.parent), name=output_path.name)
    
    # Create trainer with step-based training (like DINOv3-CP)
    # Use max_steps instead of max_epochs for fixed iteration count
    
    # Disable validation if no validation loader available
    if val_loader is None:
        num_sanity_val_steps = 0
        limit_val_batches = 0
        check_val_every_n_epoch = None
        val_check_interval = None
        logger.info("No validation set available, disabling validation")
    else:
        num_sanity_val_steps = 1
        limit_val_batches = 1.0
        # Validate every "epoch" (every steps_per_epoch steps)
        check_val_every_n_epoch = None
        val_check_interval = steps_per_epoch
        logger.info(f"Validation every {steps_per_epoch} steps")
    
    trainer = pl.Trainer(
        max_steps=max_steps,  # Train for fixed number of steps (not epochs)
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=pl_logger,
        default_root_dir=str(output_path),
        num_sanity_val_steps=num_sanity_val_steps,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=10,
        enable_checkpointing=True,
    )
    
    logger.info(f"Trainer configured:")
    logger.info(f"  - Max steps: {max_steps}")
    logger.info(f"  - Validation every: {val_check_interval} steps" if val_check_interval else "  - Validation: disabled")
    
    # Train
    logger.info("Starting training...")
    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()
    
    logger.info(f"Training complete! Checkpoints saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MAE Continue Pretraining")
    
    # Dataset parameters
    parser.add_argument("--dataset", dest="dataset_name", type=str, default="bloodmnist",
                       help="Dataset name")
    parser.add_argument("--data_root", type=str, default="/root/data",
                       help="Root directory for datasets")
    parser.add_argument("--limit_data", type=int, default=None,
                       help="Limit number of training samples")
    
    # Model parameters
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["base", "large", "huge"],
                       help="MAE model size")
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="Use pretrained weights")
    parser.add_argument("--pretrained_source", type=str,
                       default="facebook/vit-mae-base",
                       help="Source of pretrained weights")
    parser.add_argument("--mask_ratio", type=float, default=0.75,
                       help="Masking ratio")
    parser.add_argument("--norm_pix_loss", action="store_true",
                       help="Normalize pixels in loss")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=16,
                       help="Steps per epoch (like DINOv3 OFFICIAL_EPOCH_LENGTH)")
    parser.add_argument("--lr", type=float, default=1.5e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                       help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                       help="Warmup epochs")
    
    # Hardware parameters
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loading workers")
    parser.add_argument("--precision", type=str, default="16-mixed",
                       help="Training precision")
    parser.add_argument("--devices", type=int, default=1,
                       help="Number of GPUs")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/root/output/mae_cp",
                       help="Output directory")
    parser.add_argument("--exp_name", type=str, default=None,
                       help="Experiment name")
    
    # Monitoring parameters
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use W&B logging")
    parser.add_argument("--wandb_project", type=str, default="mae-cp",
                       help="W&B project name")
    
    args = parser.parse_args()
    
    train_mae_cp(**vars(args))

