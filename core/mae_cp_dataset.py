"""
Adapter to integrate DINOv3 CPDataset with stable-pretraining.

This wraps the existing CPDataset to return dictionary-structured data
compatible with stable-pretraining's data format.
"""

import sys
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Callable
from cp_datasets import CPDataset, DATASET_STATS

class MAE_CPDataset(Dataset):
    """
    CPDataset for MAE-CP training.
    
    Returns dictionary with keys:
        - 'image': Tensor of shape (C, H, W)
        - 'label': Integer label (if available)
    """
    
    def __init__(
        self,
        dataset_name: str,
        root: str,
        split: str = "TRAIN",
        limit_data: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            dataset_name: Name of dataset (e.g., 'food101', 'bloodmnist')
            root: Root directory for data
            split: Dataset split ('TRAIN', 'VAL', 'TEST')
            limit_data: Limit number of samples (for few-shot experiments)
            transform: Transform to apply to images
        """
        self.dataset_name = dataset_name
        self.root = root
        self.split = split
        self.limit_data = limit_data
        self.transform = transform
        
        # Create underlying CPDataset (without transform, we'll apply it here)
        self.cp_dataset = CPDataset(
            dataset_name=dataset_name,
            root=root,
            split=split,
            limit_data=limit_data,
            transform=None,  # We'll apply transform manually
            target_transform=None,
        )
        
        # Get dataset statistics
        self.stats = DATASET_STATS.get(dataset_name.lower(), {})
        
    def __len__(self) -> int:
        return len(self.cp_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get image and label from CPDataset
        image, label = self.cp_dataset[idx]
        
        # Create dictionary
        sample = {
            "image": image,
            "label": label if label is not None else -1,
        }
        
        # Apply transform if provided
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics (mean, std, input_size)."""
        return self.stats


def create_mae_cp_dataloader(
    dataset_name: str,
    root: str,
    split: str = "TRAIN",
    limit_data: Optional[int] = None,
    transform: Optional[Callable] = None,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """
    Create a DataLoader for MAE-CP training.
    
    Args:
        dataset_name: Name of dataset
        root: Root directory for data
        split: Dataset split
        limit_data: Limit number of samples
        transform: Transform to apply
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader instance
    """
    import torch.utils.data
    
    dataset = MAE_CPDataset(
        dataset_name=dataset_name,
        root=root,
        split=split,
        limit_data=limit_data,
        transform=transform,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )
    
    return dataloader


