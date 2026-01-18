"""
Test script for evaluating MAE-CP (Masked Autoencoder Continue Pretraining) models.

This script evaluates MAE models (both baseline pretrained and CP-finetuned) using:
- k-NN classification
- Linear probe
- K-Means clustering
- RankMe (effective rank of representations)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from tqdm import tqdm

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

import stable_pretraining as spt
from mae_cp_dataset import MAE_CPDataset, create_mae_cp_transforms
from load_mae_weights import load_pretrained_mae_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
        MAE backbone model
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
                logger.warning(f"Missing keys: {missing[:10]}...")  # Show first 10
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected[:10]}...")
        else:
            raise ValueError("Invalid checkpoint format")
    
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def extract_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from MAE model.
    
    Args:
        model: MAE backbone
        dataloader: Data loader
        device: Device
        
    Returns:
        features: [N, D] array
        labels: [N] array
    """
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"].to(device)
            labels = batch["label"]
            
            # Forward pass through MAE encoder
            latent, _, _ = model(images)
            
            # Extract CLS token as feature
            feats = latent[:, 0].cpu().numpy()  # [B, hidden_dim]
            
            # Handle labels
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            else:
                labels = np.array(labels)
            
            # Flatten labels if needed
            if labels.ndim > 1:
                labels = labels.squeeze()
            
            features_list.append(feats)
            labels_list.append(labels)
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Ensure labels are integers
    labels = labels.astype(np.int64)
    
    return features, labels


def compute_rankme(features: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Compute RankMe score (effective rank of features).
    
    Based on: https://arxiv.org/abs/2210.02885
    
    Args:
        features: [N, D] feature matrix
        epsilon: Small constant for numerical stability
        
    Returns:
        RankMe score (effective rank)
    """
    # Normalize features
    features = features - features.mean(axis=0, keepdims=True)
    
    # Compute covariance matrix
    # Use SVD on features directly (more stable than cov)
    _, s, _ = np.linalg.svd(features, full_matrices=False)
    
    # Normalize singular values to get probabilities
    s = s**2  # Eigenvalues
    p = s / (s.sum() + epsilon)
    
    # Compute entropy
    p = p + epsilon  # Avoid log(0)
    entropy = -np.sum(p * np.log(p))
    
    # RankMe = exp(entropy)
    rankme = np.exp(entropy)
    
    return float(rankme)


def evaluate_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate k-NN classifier.
    
    Returns:
        Dictionary with knn_acc, knn_f1, knn_roc_auc
    """
    # Train k-NN
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(train_features, train_labels)
    
    # Predict
    pred_labels = knn.predict(test_features)
    pred_probs = knn.predict_proba(test_features)
    
    # Compute metrics
    acc = accuracy_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels, average="macro")
    
    # ROC-AUC (multiclass)
    n_classes = len(np.unique(train_labels))
    if n_classes == 2:
        roc_auc = roc_auc_score(test_labels, pred_probs[:, 1])
    else:
        try:
            roc_auc = roc_auc_score(
                test_labels, pred_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            roc_auc = 0.0  # Some classes might be missing
    
    return {
        "knn_acc": acc,
        "knn_f1": f1,
        "knn_roc_auc": roc_auc,
    }


def evaluate_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    max_iter: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate linear probe (logistic regression).
    
    Returns:
        Dictionary with linear_acc, linear_f1, linear_roc_auc
    """
    # Train logistic regression
    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(train_features, train_labels)
    
    # Predict
    pred_labels = clf.predict(test_features)
    pred_probs = clf.predict_proba(test_features)
    
    # Compute metrics
    acc = accuracy_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels, average="macro")
    
    # ROC-AUC (multiclass)
    n_classes = len(np.unique(train_labels))
    if n_classes == 2:
        roc_auc = roc_auc_score(test_labels, pred_probs[:, 1])
    else:
        try:
            roc_auc = roc_auc_score(
                test_labels, pred_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            roc_auc = 0.0
    
    return {
        "linear_acc": acc,
        "linear_f1": f1,
        "linear_roc_auc": roc_auc,
    }


def evaluate_kmeans(
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate K-Means clustering.
    
    Returns:
        Dictionary with kmeans_ari, kmeans_nmi
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    # Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_clusters = kmeans.fit_predict(features)
    
    # Compute metrics
    ari = adjusted_rand_score(labels, pred_clusters)
    nmi = normalized_mutual_info_score(labels, pred_clusters)
    
    return {
        "kmeans_ari": ari,
        "kmeans_nmi": nmi,
    }


def test_mae_cp(
    checkpoint_path: Optional[str],
    dataset_name: str,
    data_root: str,
    probe_data_size: int = 1000,
    model_size: str = "base",
    batch_size: int = 128,
    num_workers: int = 8,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Test MAE-CP model on a dataset.
    
    Args:
        checkpoint_path: Path to checkpoint (None for baseline)
        dataset_name: Dataset name
        data_root: Data root directory
        probe_data_size: Number of samples for probe training
        model_size: Model size
        batch_size: Batch size
        num_workers: Number of workers
        device: Device
        seed: Random seed
        
    Returns:
        Dictionary with all metrics
    """
    set_seed(seed)
    
    # Create test dataset (full)
    test_dataset = MAE_CPDataset(
        dataset_name=dataset_name,
        root=data_root,
        split="TEST",
        limit_data=None,
    )
    
    dataset_stats = test_dataset.get_stats()
    img_size = dataset_stats.get("input_size", 224)
    
    # Create transforms
    _, val_transform = create_mae_cp_transforms(dataset_stats)
    
    # Apply transform
    test_dataset.transform = val_transform
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Create probe training dataset (limited)
    probe_train_dataset = MAE_CPDataset(
        dataset_name=dataset_name,
        root=data_root,
        split="TRAIN",
        limit_data=probe_data_size,
        transform=val_transform,
    )
    
    probe_train_loader = torch.utils.data.DataLoader(
        probe_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Load model
    model = load_mae_model(
        checkpoint_path=checkpoint_path,
        model_size=model_size,
        device=device,
        img_size=img_size,
    )
    
    # Extract features
    logger.info("Extracting features from probe training set...")
    probe_train_feats, probe_train_labels = extract_features(
        model, probe_train_loader, device
    )
    
    logger.info("Extracting features from test set...")
    test_feats, test_labels = extract_features(model, test_loader, device)
    
    # Evaluate
    results = {}
    
    # k-NN
    logger.info("Evaluating k-NN...")
    knn_results = evaluate_knn(
        probe_train_feats, probe_train_labels, test_feats, test_labels
    )
    results.update(knn_results)
    
    # Linear probe
    logger.info("Evaluating linear probe...")
    linear_results = evaluate_linear_probe(
        probe_train_feats, probe_train_labels, test_feats, test_labels
    )
    results.update(linear_results)
    
    # K-Means
    logger.info("Evaluating K-Means...")
    kmeans_results = evaluate_kmeans(test_feats, test_labels)
    results.update(kmeans_results)
    
    # RankMe
    logger.info("Computing RankMe...")
    rankme_score = compute_rankme(test_feats)
    results["rankme"] = rankme_score
    
    return results


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MAE-CP models")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to MAE-CP checkpoint (None for baseline)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/data",
        help="Data root directory",
    )
    parser.add_argument(
        "--probe_data_size",
        type=int,
        default=1000,
        help="Number of samples for probe training",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=["base", "large", "huge"],
        help="Model size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    results = test_mae_cp(
        checkpoint_path=args.checkpoint_path,
        dataset_name=args.dataset,
        data_root=args.data_root,
        probe_data_size=args.probe_data_size,
        model_size=args.model_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )
    
    print("\nResults:")
    print("=" * 60)
    for metric, value in results.items():
        print(f"{metric:20s}: {value:.4f}")
    print("=" * 60)

