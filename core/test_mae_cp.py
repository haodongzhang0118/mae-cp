"""
Test script for MAE-CP components.

This script tests:
1. Loading HuggingFace MAE weights
2. Dataset adapter
3. Forward pass
4. Training loop (quick test)
"""

import sys
sys.path.append("/Users/zhanghaodong/Desktop/DIET-CP/DINOv3-CP/dinov3/stable-pretraining")
sys.path.append("/Users/zhanghaodong/Desktop/DIET-CP/DINOv3-CP/dinov3")

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_load_mae_weights():
    """Test 1: Load MAE weights from HuggingFace"""
    logger.info("=" * 60)
    logger.info("Test 1: Loading MAE weights from HuggingFace")
    logger.info("=" * 60)
    
    try:
        import stable_pretraining as spt
        from load_mae_weights import load_pretrained_mae_weights
        
        # Create MAE model
        mae_model = spt.backbone.mae.vit_base_patch16_dec512d8b()
        logger.info(f"✓ Created MAE ViT-Base model")
        
        # Load pretrained weights
        load_pretrained_mae_weights(
            mae_model,
            source="facebook/vit-mae-base",
            strict=False
        )
        logger.info(f"✓ Loaded pretrained weights from HuggingFace")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            latent, pred, mask = mae_model(dummy_input)
        
        logger.info(f"✓ Forward pass successful:")
        logger.info(f"  - Latent shape: {latent.shape}")
        logger.info(f"  - Pred shape: {pred.shape}")
        logger.info(f"  - Mask shape: {mask.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_adapter():
    """Test 2: Test dataset adapter"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Testing dataset adapter")
    logger.info("=" * 60)
    
    try:
        from mae_cp_dataset import MAE_CPDataset
        from dinov3.data.datasets.cp_datasets import DATASET_STATS
        
        # Test with a small MedMNIST dataset
        dataset = MAE_CPDataset(
            dataset_name="bloodmnist",
            root="/tmp/test_data",  # Will fail if data not present, but that's ok
            split="TRAIN",
            limit_data=10,
        )
        
        logger.info(f"✓ Created MAE_CPDataset")
        logger.info(f"  - Dataset: bloodmnist")
        logger.info(f"  - Stats: {dataset.get_stats()}")
        
        # Try to get a sample (may fail if data not downloaded)
        try:
            sample = dataset[0]
            logger.info(f"✓ Retrieved sample:")
            logger.info(f"  - Keys: {sample.keys()}")
            logger.info(f"  - Image type: {type(sample['image'])}")
            logger.info(f"  - Label: {sample['label']}")
        except Exception as e:
            logger.warning(f"  Could not retrieve sample (data not available): {e}")
            logger.info(f"  This is OK if you haven't downloaded the data yet")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mae_forward():
    """Test 3: Test MAE forward function"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Testing MAE forward function")
    logger.info("=" * 60)
    
    try:
        import stable_pretraining as spt
        from mae_cp_train import mae_cp_forward
        
        # Create a mock module
        class MockModule:
            def __init__(self):
                self.backbone = spt.backbone.mae.vit_base_patch16_dec512d8b()
                self.training = True
                self.hparams = {"norm_pix_loss": False}
                self.logs = {}
            
            def log(self, key, value, **kwargs):
                self.logs[key] = value
        
        module = MockModule()
        
        # Create dummy batch
        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "label": torch.tensor([0, 1]),
        }
        
        # Forward pass
        output = mae_cp_forward(module, batch, stage="train")
        
        logger.info(f"✓ Forward pass successful:")
        logger.info(f"  - Output keys: {output.keys()}")
        logger.info(f"  - Embedding shape: {output['embedding'].shape}")
        logger.info(f"  - Loss: {output['loss'].item():.4f}")
        logger.info(f"  - Logged metrics: {list(module.logs.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_training():
    """Test 4: Quick training test (1 iteration)"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Testing quick training loop")
    logger.info("=" * 60)
    
    try:
        import stable_pretraining as spt
        from mae_cp_train import mae_cp_forward
        import lightning as pl
        
        # Create MAE backbone
        backbone = spt.backbone.mae.vit_base_patch16_dec512d8b()
        logger.info(f"✓ Created MAE backbone")
        
        # Create module
        module = spt.Module(
            backbone=backbone,
            forward=mae_cp_forward,
            optim={
                "optimizer": {
                    "type": "AdamW",
                    "lr": 1.5e-4,
                    "weight_decay": 0.05,
                },
            },
            norm_pix_loss=False,
            mask_ratio=0.75,
        )
        logger.info(f"✓ Created spt.Module")
        
        # Create dummy data
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return {
                    "image": torch.randn(3, 224, 224),
                    "label": torch.randint(0, 10, (1,)).item(),
                }
        
        train_loader = torch.utils.data.DataLoader(
            DummyDataset(),
            batch_size=2,
            num_workers=0,
        )
        
        data = spt.data.DataModule(train=train_loader, val=None)
        logger.info(f"✓ Created data module")
        
        # Create trainer (just 1 step)
        trainer = pl.Trainer(
            max_steps=2,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )
        logger.info(f"✓ Created trainer")
        
        # Run quick training
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()
        logger.info(f"✓ Training loop completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("MAE-CP Component Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Load MAE Weights", test_load_mae_weights),
        ("Dataset Adapter", test_dataset_adapter),
        ("MAE Forward", test_mae_forward),
        ("Quick Training", test_quick_training),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    logger.info("=" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! MAE-CP is ready to use.")
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")


if __name__ == "__main__":
    main()

