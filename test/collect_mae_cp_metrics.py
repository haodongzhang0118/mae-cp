"""
Collect metrics for MAE-CP experiments across all datasets and CP sizes.

This script:
1. Scans the output directory for all trained models
2. Evaluates each model (baseline + CP) on test sets
3. Computes delta metrics (CP - baseline)
4. Saves results to CSV

Directory structure expected:
  output_root/
    ├── dataset1/
    │   └── base/
    │       ├── 10/
    │       │   ├── best_f1.ckpt
    │       │   └── last.ckpt
    │       ├── 100/
    │       └── ...
    ├── dataset2/
    └── ...

Output CSV columns:
  - dataset: Dataset name
  - cp_dataset_size: CP data size (0 for baseline)
  - probe_dataset_size: Probe training data size
  - knn_acc, knn_f1, knn_roc_auc
  - linear_acc, linear_f1, linear_roc_auc
  - kmeans_ari, kmeans_nmi
  - rankme
  - delta_knn_acc, delta_knn_f1, delta_knn_roc_auc
  - delta_linear_acc, delta_linear_f1, delta_linear_roc_auc
  - delta_kmeans_ari, delta_kmeans_nmi
  - delta_rankme
"""

import os
import sys
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from test_mae_cp import test_mae_cp, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_checkpoints(
    output_root: str,
    model_size: str = "base",
    checkpoint_type: str = "best_f1",
) -> Dict[str, Dict[int, str]]:
    """
    Discover all available checkpoints in the output directory.
    
    Args:
        output_root: Root output directory
        model_size: Model size (base, large, huge)
        checkpoint_type: Which checkpoint to use ('best_f1' or 'last')
        
    Returns:
        Dictionary mapping dataset -> {cp_size -> checkpoint_path}
        
    Example:
        {
            'bloodmnist': {
                10: '/root/output/bloodmnist/base/10/best_f1.ckpt',
                100: '/root/output/bloodmnist/base/100/best_f1.ckpt',
                ...
            },
            ...
        }
    """
    output_root = Path(output_root)
    checkpoints = {}
    
    if not output_root.exists():
        logger.warning(f"Output root does not exist: {output_root}")
        return checkpoints
    
    # Scan for datasets
    for dataset_dir in sorted(output_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        model_dir = dataset_dir / model_size
        
        if not model_dir.exists():
            continue
        
        # Scan for CP sizes
        dataset_checkpoints = {}
        for cp_size_dir in sorted(model_dir.iterdir()):
            if not cp_size_dir.is_dir():
                continue
            
            try:
                cp_size = int(cp_size_dir.name)
            except ValueError:
                continue  # Skip non-numeric directories
            
            # Look for checkpoint
            ckpt_path = cp_size_dir / f"{checkpoint_type}.ckpt"
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint not found: {ckpt_path}")
                continue
            
            dataset_checkpoints[cp_size] = str(ckpt_path)
        
        if dataset_checkpoints:
            checkpoints[dataset_name] = dataset_checkpoints
            logger.info(
                f"Found {len(dataset_checkpoints)} checkpoints for {dataset_name}"
            )
    
    return checkpoints


def collect_metrics(
    output_root: str,
    data_root: str,
    probe_data_size: int = 1000,
    model_size: str = "base",
    checkpoint_type: str = "best_f1",
    batch_size: int = 128,
    num_workers: int = 8,
    device: str = "cuda",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Collect metrics for all datasets and CP sizes.
    
    Args:
        output_root: Root directory containing checkpoints
        data_root: Root directory containing datasets
        probe_data_size: Number of samples for probe training
        model_size: Model size
        checkpoint_type: Which checkpoint to use ('best_f1' or 'last')
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        device: Device
        seed: Random seed
        
    Returns:
        DataFrame with all metrics
    """
    # Discover checkpoints
    logger.info(f"Scanning {output_root} for checkpoints...")
    checkpoints = discover_checkpoints(output_root, model_size, checkpoint_type)
    
    if not checkpoints:
        logger.error("No checkpoints found!")
        return pd.DataFrame()
    
    logger.info(f"Found checkpoints for {len(checkpoints)} datasets")
    
    # Collect metrics
    records = []
    
    for dataset_name in sorted(checkpoints.keys()):
        dataset_ckpts = checkpoints[dataset_name]
        
        # Get baseline results (pretrained MAE without CP)
        logger.info("=" * 80)
        logger.info(
            f"Dataset: {dataset_name} | CP size: 0 (baseline) | "
            f"Probe size: {probe_data_size}"
        )
        logger.info("=" * 80)
        
        set_seed(seed)
        
        try:
            baseline_results = test_mae_cp(
                checkpoint_path=None,  # Baseline pretrained MAE
                dataset_name=dataset_name,
                data_root=data_root,
                probe_data_size=probe_data_size,
                model_size=model_size,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                seed=seed,
            )
            
            # Store baseline
            baseline_record = {
                "dataset": dataset_name,
                "cp_dataset_size": 0,
                "probe_dataset_size": probe_data_size,
            }
            baseline_record.update(baseline_results)
            
            # Add zero deltas for baseline
            for metric in baseline_results.keys():
                baseline_record[f"delta_{metric}"] = 0.0
            
            records.append(baseline_record)
            
        except Exception as e:
            logger.error(f"Error evaluating baseline for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Evaluate CP models
        for cp_size in sorted(dataset_ckpts.keys()):
            ckpt_path = dataset_ckpts[cp_size]
            
            logger.info("=" * 80)
            logger.info(
                f"Dataset: {dataset_name} | CP size: {cp_size} | "
                f"Probe size: {probe_data_size}"
            )
            logger.info(f"Checkpoint: {ckpt_path}")
            logger.info("=" * 80)
            
            set_seed(seed)
            
            try:
                cp_results = test_mae_cp(
                    checkpoint_path=ckpt_path,
                    dataset_name=dataset_name,
                    data_root=data_root,
                    probe_data_size=probe_data_size,
                    model_size=model_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device,
                    seed=seed,
                )
                
                # Store CP results
                cp_record = {
                    "dataset": dataset_name,
                    "cp_dataset_size": cp_size,
                    "probe_dataset_size": probe_data_size,
                }
                cp_record.update(cp_results)
                
                # Compute deltas (CP - baseline)
                for metric in cp_results.keys():
                    if metric in baseline_results:
                        delta = cp_results[metric] - baseline_results[metric]
                        cp_record[f"delta_{metric}"] = delta
                    else:
                        cp_record[f"delta_{metric}"] = 0.0
                
                records.append(cp_record)
                
            except Exception as e:
                logger.error(
                    f"Error evaluating {dataset_name} with CP size {cp_size}: {e}"
                )
                import traceback
                traceback.print_exc()
                continue
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Reorder columns
    base_cols = ["dataset", "cp_dataset_size", "probe_dataset_size"]
    metric_cols = [
        "knn_acc",
        "knn_f1",
        "knn_roc_auc",
        "linear_acc",
        "linear_f1",
        "linear_roc_auc",
        "kmeans_ari",
        "kmeans_nmi",
        "rankme",
    ]
    delta_cols = [f"delta_{m}" for m in metric_cols]
    
    # Only include columns that exist
    all_cols = base_cols + metric_cols + delta_cols
    existing_cols = [c for c in all_cols if c in df.columns]
    df = df[existing_cols]
    
    return df


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect MAE-CP metrics across all datasets"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory containing checkpoints (e.g., /root/output)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/data",
        help="Root directory containing datasets",
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
        "--checkpoint_type",
        type=str,
        default="best_f1",
        choices=["best_f1", "last"],
        help="Which checkpoint to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation",
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
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="mae_cp_metrics.csv",
        help="Output CSV file",
    )
    
    args = parser.parse_args()
    
    # Collect metrics
    logger.info("Starting metric collection...")
    df = collect_metrics(
        output_root=args.output_root,
        data_root=args.data_root,
        probe_data_size=args.probe_data_size,
        model_size=args.model_size,
        checkpoint_type=args.checkpoint_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )
    
    if df.empty:
        logger.error("No metrics collected!")
        return
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved metrics to {args.output_csv}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Collected metrics for {df['dataset'].nunique()} datasets")
    print(f"Total experiments: {len(df)}")
    print("=" * 80)
    print("\nFirst few rows:")
    print(df.head(10).to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()

