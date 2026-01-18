"""
Evaluation metrics for MAE-CP models.

This module is adapted from DIET_Tuning/evaluation/metrics.py with RankMe added.
"""

import os
import numpy as np
import torch
import time
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    roc_auc_score,
)


def compute_rankme(features):
    """
    Compute RankMe (effective rank) of feature representations.
    
    RankMe = exp(entropy of singular values)
    
    Args:
        features: numpy array of shape (N, D) where N is number of samples
        
    Returns:
        float: RankMe score
    """
    # Center the features
    features_centered = features - features.mean(axis=0, keepdims=True)
    
    # SVD decomposition
    _, s, _ = np.linalg.svd(features_centered, full_matrices=False)
    
    # Normalize singular values to get probability distribution
    p = (s ** 2) / np.sum(s ** 2)
    
    # Compute entropy
    entropy = -np.sum(p * np.log(p + 1e-12))
    
    # RankMe = exp(entropy)
    rankme = np.exp(entropy)
    
    return rankme


def zero_shot_eval(
    model,
    train_loader,
    test_loader,
    device,
    probe_lr=1e-3,
    probe_steps=10000,
    store_embeddings=False,
):
    """Evaluate model using zero-shot methods with proper train/test split.

    Trains on train_loader features and evaluates on test_loader features.
    Single-label datasets only.

    Args:
        model: Model to evaluate
        train_loader: DataLoader for training features
        test_loader: DataLoader for test features
        device: Device to run evaluation on
        probe_lr: Learning rate for linear probe (default: 1e-3)
        probe_steps: Number of training steps for linear probe (default: 10000)
        store_embeddings: Whether to store embeddings to disk

    Returns:
        dict: Dictionary of evaluation metrics including:
            - knn_acc, knn_f1, knn_roc_auc
            - linear_acc, linear_f1, linear_roc_auc
            - kmeans_ari, kmeans_nmi
            - rankme (effective rank)
    """
    start_time = time.time()
    print("Extracting features for zero-shot evaluation with train/test split...")

    def extract_features(loader, split_name):
        """Extract features from a data loader."""
        features, labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting {split_name} features"):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y, _ = batch
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                elif isinstance(batch, dict):
                    # Handle dict batch (from MAE_CPDataset)
                    x = batch["image"]
                    y = batch["label"]
                else:
                    raise ValueError(f"Unexpected batch structure: {type(batch)}")

                x = x.to(device)
                feat = model(x)

                if isinstance(feat, list):
                    features.extend([f.detach().cpu().numpy() for f in feat])
                else:
                    features.append(feat.detach().cpu().numpy())

                # Handle label conversion
                if isinstance(y, torch.Tensor):
                    labels.append(y.detach().cpu().numpy())
                elif isinstance(y, np.ndarray):
                    labels.append(y)
                else:
                    labels.append(np.array(y))

        features = np.vstack(features)
        labels = np.concatenate(labels, axis=0)

        # Normalize labels to 1D
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        return features, labels

    # Extract train and test features separately
    train_features, train_labels = extract_features(train_loader, "train")
    test_features, test_labels = extract_features(test_loader, "test")

    if store_embeddings:
        store_path = f"data/embeddings/mae_{time.strftime('%Y%m%d-%H%M%S')}/"
        os.makedirs(store_path, exist_ok=True)
        np.save(f"{store_path}train_features.npy", train_features)
        np.save(f"{store_path}train_labels.npy", train_labels)
        print(f"Stored train features and labels in {store_path}")

    print(f"Train features: {train_features.shape}, Train labels: {train_labels.shape}")
    print(f"Test features: {test_features.shape}, Test labels: {test_labels.shape}")
    print(f"Feature extraction time: {time.time() - start_time:.2f}s")

    results = {}

    # ---------- k-NN ----------
    print("\nRunning k-NN evaluation...")
    t0 = time.time()

    # Dynamically set n_neighbors to avoid error when training set is small
    n_train_samples = train_features.shape[0]
    n_neighbors = min(20, n_train_samples)
    print(
        f"Using {n_neighbors} neighbors (max of 20 or available training samples: {n_train_samples})"
    )

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors, metric="cosine", weights="distance"
    )
    train_features_norm = normalize(train_features, norm="l2")
    test_features_norm = normalize(test_features, norm="l2")
    knn.fit(train_features_norm, train_labels)
    knn_pred = knn.predict(test_features_norm)
    knn_proba = knn.predict_proba(test_features_norm)

    results["knn_acc"] = accuracy_score(test_labels, knn_pred)
    results["knn_f1"] = f1_score(
        test_labels, knn_pred, average="macro", zero_division=0
    )

    # ROC AUC calculation - handle binary vs multiclass
    num_classes = len(np.unique(train_labels))
    try:
        if num_classes == 2:
            # For binary classification, use positive class probabilities
            positive_proba = np.array(knn_proba)[:, 1]
            results["knn_roc_auc"] = roc_auc_score(test_labels, positive_proba)
        else:
            results["knn_roc_auc"] = roc_auc_score(
                test_labels, knn_proba, multi_class="ovr", average="macro"
            )
    except ValueError as e:
        print(f"Warning: Could not compute ROC AUC for k-NN: {e}")
        results["knn_roc_auc"] = 0.0

    print(f"k-NN accuracy: {results['knn_acc']:.4f}, time: {time.time() - t0:.2f}s")
    print(f"k-NN F1 (macro): {results['knn_f1']:.4f}")
    print(f"k-NN ROC AUC: {results['knn_roc_auc']:.4f}")

    # ---------- Linear probe ----------
    print("\nRunning linear probe evaluation (train on train, test on test)...")
    t0 = time.time()

    out_dim = int(num_classes)
    clf = torch.nn.Linear(train_features.shape[1], out_dim).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=probe_lr)
    crit = torch.nn.CrossEntropyLoss()

    clf.train()
    train_features_norm = normalize(train_features, norm="l2")
    test_features_norm = normalize(test_features, norm="l2")

    Xtr = torch.as_tensor(train_features_norm, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    Xte = torch.as_tensor(test_features_norm, dtype=torch.float32, device=device)

    batch_size = 512
    n_samples = Xtr.shape[0]
    n_batches = max(1, (n_samples + batch_size - 1) // batch_size)
    
    for step in range(probe_steps):
        # Shuffle indices for each epoch
        if step % n_batches == 0:
            indices = torch.randperm(n_samples, device=device)

        # Get current batch
        batch_idx = step % n_batches
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        X_batch = Xtr[batch_indices]
        y_batch = ytr[batch_indices]

        opt.zero_grad()
        loss = crit(clf(X_batch), y_batch)
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        logits = clf(Xte)
        pred = logits.argmax(dim=1).cpu().numpy()
        # Get probabilities for ROC AUC calculation
        probe_proba = torch.softmax(logits, dim=1).cpu().numpy()

    results["linear_acc"] = accuracy_score(test_labels, pred)
    results["linear_f1"] = f1_score(test_labels, pred, average="macro", zero_division=0)

    # ROC AUC calculation for linear probe
    try:
        if num_classes == 2:
            # For binary classification, use positive class probabilities
            positive_proba = probe_proba[:, 1]
            results["linear_roc_auc"] = roc_auc_score(test_labels, positive_proba)
        else:
            results["linear_roc_auc"] = roc_auc_score(
                test_labels, probe_proba, multi_class="ovr", average="macro"
            )
    except ValueError as e:
        print(f"Warning: Could not compute ROC AUC for linear probe: {e}")
        results["linear_roc_auc"] = 0.0

    print(
        f"Linear probe accuracy: {results['linear_acc']:.4f}, "
        f"time: {time.time() - t0:.2f}s"
    )
    print(f"Linear probe F1 (macro): {results['linear_f1']:.4f}")
    print(f"Linear probe ROC AUC: {results['linear_roc_auc']:.4f}")

    # ---------- k-means clustering ----------
    print("\nRunning k-means clustering evaluation...")
    t0 = time.time()

    # Use normalized test features for clustering
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    kmeans_pred = kmeans.fit_predict(test_features_norm)

    # Calculate k-means metrics
    results["kmeans_ari"] = adjusted_rand_score(test_labels, kmeans_pred)
    results["kmeans_nmi"] = normalized_mutual_info_score(test_labels, kmeans_pred)

    print(f"k-means ARI: {results['kmeans_ari']:.4f}")
    print(f"k-means NMI: {results['kmeans_nmi']:.4f}")
    print(f"k-means time: {time.time() - t0:.2f}s")

    # ---------- RankMe ----------
    print("\nComputing RankMe (effective rank)...")
    t0 = time.time()
    
    # Compute RankMe on test features
    results["rankme"] = compute_rankme(test_features)
    
    print(f"RankMe: {results['rankme']:.4f}")
    print(f"RankMe time: {time.time() - t0:.2f}s")

    print(f"\nTotal zero-shot evaluation time: {time.time() - start_time:.2f}s")
    return results

