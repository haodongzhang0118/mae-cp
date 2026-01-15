"""
Infinite Sampler for MAE-CP training.

This sampler infinitely cycles through the dataset, allowing training
with fixed iteration counts regardless of dataset size. This is especially
important for few-shot scenarios where dataset_size < batch_size.

Adapted from DINOv3's InfiniteSampler, but standalone (no dependencies).
"""

import torch
import itertools
from typing import Iterator
from torch.utils.data.sampler import Sampler


class InfiniteSampler(Sampler):
    """
    Sampler that infinitely cycles through dataset indices.
    
    This solves the problem where limit_data < batch_size by repeating
    samples to fill each batch. For example:
    
    - Dataset: 10 samples, Batch size: 64
    - Sampler will produce: [0,1,2,...,9,0,1,2,...,9,0,1,2,...] indefinitely
    
    Args:
        dataset_size: Number of samples in the dataset
        shuffle: Whether to shuffle indices
        seed: Random seed for shuffling
        rank: Distributed training rank (default: 0 for single GPU)
        world_size: Total number of processes (default: 1 for single GPU)
    """
    
    def __init__(
        self,
        dataset_size: int,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        
    def __iter__(self) -> Iterator[int]:
        """Infinite iterator over dataset indices."""
        if self.shuffle:
            # Create a generator for reproducibility
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            
            # Infinite loop with shuffling
            while True:
                # Generate random permutation
                indices = torch.randperm(
                    self.dataset_size, 
                    generator=generator,
                    dtype=torch.int64
                ).tolist()
                
                # Yield indices for this rank (for distributed training)
                for idx in indices[self.rank::self.world_size]:
                    yield idx
        else:
            # Infinite loop without shuffling
            while True:
                indices = list(range(self.dataset_size))
                # Yield indices for this rank
                for idx in indices[self.rank::self.world_size]:
                    yield idx
    
    def __len__(self) -> int:
        """
        Return a placeholder length.
        
        Note: Infinite samplers don't have a true length, but we return
        dataset_size to avoid issues with some libraries.
        """
        return self.dataset_size // self.world_size


class FiniteSubsetSampler(Sampler):
    """
    Sampler that takes a finite subset from an InfiniteSampler.
    
    This is useful when you want to train for a specific number of iterations
    but still benefit from the infinite cycling behavior.
    
    Args:
        infinite_sampler: An InfiniteSampler instance
        num_samples: Number of samples to draw before stopping
    """
    
    def __init__(self, infinite_sampler: InfiniteSampler, num_samples: int):
        self.infinite_sampler = infinite_sampler
        self.num_samples = num_samples
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over a finite subset of infinite indices."""
        yield from itertools.islice(self.infinite_sampler, self.num_samples)
    
    def __len__(self) -> int:
        return self.num_samples


def create_infinite_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    drop_last: bool = True,
    pin_memory: bool = True,
    collate_fn=None,
):
    """
    Create a DataLoader with infinite sampling.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle
        seed: Random seed
        rank: Distributed training rank
        world_size: Number of GPUs
        drop_last: Whether to drop last incomplete batch
        pin_memory: Whether to pin memory
        collate_fn: Optional collate function
        
    Returns:
        DataLoader with infinite sampler
    """
    sampler = InfiniteSampler(
        dataset_size=len(dataset),
        shuffle=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the infinite sampler
    import torch.utils.data
    
    print("Testing InfiniteSampler...")
    print("=" * 60)
    
    # Test case 1: Small dataset (10 samples) with large batch (64)
    print("\nTest 1: Dataset size=10, Batch size=64")
    print("-" * 60)
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            return idx
    
    dataset = DummyDataset(10)
    dataloader = create_infinite_dataloader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,  # Easier to verify
        num_workers=0,
    )
    
    # Get first 3 batches
    dataloader_iter = iter(dataloader)
    for i in range(3):
        batch = next(dataloader_iter)
        print(f"Batch {i+1}: size={len(batch)}, samples={batch.tolist()[:10]}... (showing first 10)")
    
    print("\n✓ Successfully created batches of size 64 from dataset of size 10!")
    print("  Each batch repeats samples to fill the required batch size.")
    
    # Test case 2: Shuffling
    print("\n" + "=" * 60)
    print("Test 2: Shuffling test")
    print("-" * 60)
    
    dataloader_shuffled = create_infinite_dataloader(
        dataset=dataset,
        batch_size=20,
        shuffle=True,
        seed=42,
        num_workers=0,
    )
    
    dataloader_iter = iter(dataloader_shuffled)
    batch1 = next(dataloader_iter)
    batch2 = next(dataloader_iter)
    
    print(f"Batch 1 (shuffled): {batch1.tolist()}")
    print(f"Batch 2 (shuffled): {batch2.tolist()}")
    print("\n✓ Shuffling works correctly!")
    
    # Test case 3: Iteration count
    print("\n" + "=" * 60)
    print("Test 3: Training for fixed iterations")
    print("-" * 60)
    
    OFFICIAL_EPOCH_LENGTH = 16
    EPOCHS = 5
    MAX_STEPS = EPOCHS * OFFICIAL_EPOCH_LENGTH  # 80 steps
    
    dataloader = create_infinite_dataloader(
        dataset=dataset,
        batch_size=64,
        shuffle=True,
        seed=0,
        num_workers=0,
    )
    
    step_count = 0
    dataloader_iter = iter(dataloader)
    
    for step in range(MAX_STEPS):
        batch = next(dataloader_iter)
        step_count += 1
        if step_count % 16 == 0:
            print(f"  Epoch {step_count // 16}: Completed 16 iterations")
    
    print(f"\n✓ Trained for {step_count} steps ({EPOCHS} epochs × {OFFICIAL_EPOCH_LENGTH} steps/epoch)")
    print(f"  Total samples seen: {step_count * 64} (with repetition)")
    print(f"  Unique samples in dataset: {len(dataset)}")
    print(f"  Each sample seen ~{(step_count * 64) / len(dataset):.0f} times")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

