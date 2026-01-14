# MAE-CP 快速入门指南

## 🚀 5 分钟快速开始

### 1. 安装依赖

```bash
# 进入 stable-pretraining 目录
cd /Users/zhanghaodong/Desktop/DIET-CP/DINOv3-CP/dinov3/stable-pretraining

# 安装 stable-pretraining
pip install -e ".[vision,tracking]"

# 安装其他依赖
pip install transformers datasets==2.20.0
```

### 2. 测试安装

```bash
cd /Users/zhanghaodong/Desktop/DIET-CP/DINOv3-CP/dinov3
python mae_cp/test_mae_cp.py
```

### 3. 运行单个实验

```bash
python mae_cp/mae_cp_train.py \
    --dataset bloodmnist \
    --data_root /root/data \
    --limit_data 100 \
    --model_size base \
    --pretrained \
    --batch_size 64 \
    --epochs 10 \
    --output_dir /root/output/mae_cp_test
```

### 4. 批量实验

```bash
# 编辑配置
vim mae_cp/run_mae_cp_experiments.sh

# 运行
./mae_cp/run_mae_cp_experiments.sh
```

---

## 📁 文件说明

| 文件 | 功能 | 重要性 |
|------|------|--------|
| `load_mae_weights.py` | 从 HuggingFace 加载预训练 MAE 权重 | ⭐⭐⭐ |
| `mae_cp_dataset.py` | 数据集适配器（CPDataset → stable-pretraining） | ⭐⭐⭐ |
| `mae_cp_train.py` | 主训练脚本 | ⭐⭐⭐⭐⭐ |
| `run_mae_cp_experiments.sh` | 批量实验脚本 | ⭐⭐⭐⭐ |
| `test_mae_cp.py` | 组件测试脚本 | ⭐⭐ |
| `README.md` | 详细文档 | ⭐⭐ |
| `QUICKSTART.md` | 本文件 | ⭐⭐ |

---

## 🔑 核心概念

### 1. **HuggingFace 权重加载**

```python
from load_mae_weights import load_pretrained_mae_weights
import stable_pretraining as spt

# 创建 MAE 模型
mae_model = spt.backbone.mae.vit_base_patch16_dec512d8b()

# 从 HuggingFace 加载权重
load_pretrained_mae_weights(
    mae_model,
    source="facebook/vit-mae-base",  # HuggingFace model ID
    strict=False
)
```

**支持的预训练模型：**
- `facebook/vit-mae-base` - ViT-Base (768-dim, 12 layers)
- `facebook/vit-mae-large` - ViT-Large (1024-dim, 24 layers)
- `facebook/vit-mae-huge` - ViT-Huge (1280-dim, 32 layers)

### 2. **数据集适配**

```python
from mae_cp_dataset import MAE_CPDataset

# 创建数据集（自动适配 CPDataset）
dataset = MAE_CPDataset(
    dataset_name="bloodmnist",
    root="/root/data/bloodmnist",
    split="TRAIN",
    limit_data=100,  # Few-shot
    transform=transform,
)

# 返回格式：{"image": Tensor, "label": int}
```

### 3. **训练流程**

```python
from mae_cp_train import mae_cp_forward
import stable_pretraining as spt

# 1. 创建 MAE backbone
backbone = spt.backbone.mae.vit_base_patch16_dec512d8b()

# 2. 加载预训练权重
load_pretrained_mae_weights(backbone, "facebook/vit-mae-base")

# 3. 创建 Module
module = spt.Module(
    backbone=backbone,
    forward=mae_cp_forward,  # 自定义 forward 函数
    optim={...},
)

# 4. 创建 Trainer 并训练
trainer = pl.Trainer(...)
manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
```

### 4. **Forward 函数逻辑**

```python
def mae_cp_forward(self, batch, stage):
    # 1. MAE forward pass
    latent, pred, mask = self.backbone(batch["image"])
    
    # 2. 提取 CLS token 作为 embedding
    out = {"embedding": latent[:, 0]}
    
    # 3. 训练时计算重建损失
    if self.training:
        target = self.backbone.patchify(batch["image"])
        loss = spt.losses.mae(target, pred, mask)
        out["loss"] = loss
    
    return out
```

---

## 🎯 典型使用场景

### 场景 1: Few-shot 实验（少样本）

```bash
for num_samples in 10 50 100 250 500; do
    python mae_cp/mae_cp_train.py \
        --dataset food101 \
        --limit_data $num_samples \
        --epochs 100 \
        --output_dir /root/output/mae_cp/food101_fewshot
done
```

### 场景 2: 多数据集对比

```bash
for dataset in bloodmnist pathmnist chestmnist; do
    python mae_cp/mae_cp_train.py \
        --dataset $dataset \
        --epochs 100 \
        --output_dir /root/output/mae_cp/medmnist_comparison
done
```

### 场景 3: 不同模型规格对比

```bash
for model_size in base large huge; do
    python mae_cp/mae_cp_train.py \
        --dataset food101 \
        --model_size $model_size \
        --output_dir /root/output/mae_cp/model_size_comparison
done
```

---

## 🔧 常见配置

### 小内存配置（<16GB GPU）

```bash
python mae_cp/mae_cp_train.py \
    --dataset bloodmnist \
    --model_size base \
    --batch_size 64 \
    --precision 16-mixed \
    --num_workers 4
```

### 多 GPU 训练

```bash
python mae_cp/mae_cp_train.py \
    --dataset food101 \
    --devices 4 \
    --batch_size 256
```

### 快速测试（验证流程）

```bash
python mae_cp/mae_cp_train.py \
    --dataset bloodmnist \
    --limit_data 100 \
    --epochs 5 \
    --batch_size 32 \
    --num_workers 0
```

---

## 📊 监控训练

### 1. **使用 W&B**

```bash
python mae_cp/mae_cp_train.py \
    --dataset food101 \
    --use_wandb \
    --wandb_project mae-cp-food101
```

然后访问：https://wandb.ai/your-username/mae-cp-food101

### 2. **使用 TensorBoard**

```bash
# 训练会自动保存 CSV logs
tensorboard --logdir /root/output/mae_cp
```

### 3. **查看实时指标**

训练过程中会显示：
- `train/loss`: 重建损失
- `val/linear_probe/top1`: Linear probe 准确率（在线监控）
- `val/rankme/rankme`: 表示质量指标

---

## 🐛 故障排查

### 问题 1: ImportError - transformers

```bash
pip install transformers
```

### 问题 2: Out of Memory

减小 batch size:
```bash
--batch_size 64  # 或 32, 16
```

### 问题 3: 数据集找不到

确保数据路径正确：
```bash
ls /root/data/bloodmnist  # 应该有数据文件
```

或修改 `--data_root`:
```bash
--data_root /path/to/your/data
```

### 问题 4: 预训练权重下载慢

设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或使用本地权重：
```bash
--pretrained_source /path/to/mae_checkpoint.pth
```

### 问题 5: stable-pretraining 找不到

确保添加到 PYTHONPATH:
```bash
export PYTHONPATH="/path/to/stable-pretraining:$PYTHONPATH"
```

---

## 📈 预期结果

### Few-shot Learning（BloodMNIST）

| Samples | Random Init | MAE-CP (Base) | Improvement |
|---------|-------------|---------------|-------------|
| 10 | 20.3% | 42.1% | +21.8% |
| 50 | 45.2% | 68.4% | +23.2% |
| 100 | 58.7% | 76.3% | +17.6% |
| Full | 84.2% | 91.7% | +7.5% |

*以上为示例数据，实际结果可能有所不同*

### 训练时间（ViT-Base, 1x A100）

| Dataset | Samples | Epochs | Time |
|---------|---------|--------|------|
| BloodMNIST | 100 | 100 | ~15 min |
| Food101 | 1000 | 100 | ~2 hours |
| Food101 | Full (75K) | 100 | ~20 hours |

---

## 🔗 相关资源

- **MAE 论文**: https://arxiv.org/abs/2111.06377
- **stable-pretraining 文档**: https://rbalestr-lab.github.io/stable-pretraining/
- **HuggingFace MAE**: https://huggingface.co/facebook/vit-mae-base
- **DINOv3-CP (对比)**: ../dinov3/configs/train/cp/

---

## 💡 最佳实践

1. **先用小数据测试**: 使用 `--limit_data 100 --epochs 5` 验证流程
2. **监控 Linear Probe**: 如果 probe 准确率不提升，说明表示质量没有改善
3. **使用预训练权重**: `--pretrained` 几乎总是能提供更好的起点
4. **调整 learning rate**: 如果 loss 不下降，尝试降低 `--lr`（如 1e-4）
5. **保存 checkpoints**: 定期备份 `/root/output/mae_cp`

---

## 🎓 进阶使用

### 自定义 Forward 函数

编辑 `mae_cp_train.py` 中的 `mae_cp_forward` 函数来实现自定义训练逻辑。

### 添加新数据集

1. 在 `dinov3/data/datasets/cp_datasets.py` 中添加数据集支持
2. 在 `DATASET_STATS` 中添加统计信息
3. 使用 `MAE_CPDataset` 自动适配

### 修改优化器

```python
module = spt.Module(
    ...,
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "CosineAnnealingLR",
            "T_max": epochs,
        },
    },
)
```

---

## ✅ Checklist

开始训练前的检查清单：

- [ ] 安装了 `stable-pretraining` 和依赖
- [ ] 测试脚本 `test_mae_cp.py` 通过
- [ ] 数据集已下载到正确路径
- [ ] GPU 可用（`torch.cuda.is_available()`）
- [ ] 足够的磁盘空间（至少 50GB）
- [ ] 配置了 HuggingFace token（如需下载模型）

---

## 🙋 获取帮助

1. **查看详细文档**: `mae_cp/README.md`
2. **运行测试**: `python mae_cp/test_mae_cp.py`
3. **查看示例**: `mae_cp/mae_cp_train.py` 的 `__main__` 部分
4. **对比 DINOv3-CP**: 参考 `run_all_experiments.txt`

祝训练顺利！🚀

