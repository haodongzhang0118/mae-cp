# MAE Continue Pretraining (MAE-CP)

MAE Continue Pretraining for domain adaptation using the `stable-pretraining` framework.

## 概述

MAE-CP 使用 Masked Autoencoder (MAE) 进行领域适应的持续预训练，类似于 DINOv3-CP，但使用重建目标而非自蒸馏。

### 主要特性

- ✅ **基于 stable-pretraining**: 使用 PyTorch Lightning，易于扩展
- ✅ **预训练权重加载**: 从 HuggingFace 加载官方 MAE 权重
- ✅ **支持多数据集**: Food101, FGVC-Aircraft, Galaxy10, MedMNIST 等
- ✅ **Few-shot 实验**: 支持不同样本量的实验
- ✅ **在线监控**: Linear probe 和 RankMe 实时监控训练质量
- ✅ **灵活配置**: 支持 base/large/huge 模型规格

## 文件结构

```
mae_cp/
├── load_mae_weights.py      # HuggingFace 权重加载和转换
├── mae_cp_dataset.py         # 数据集适配器（CPDataset → stable-pretraining）
├── mae_cp_train.py           # 主训练脚本
├── run_mae_cp_experiments.sh # 批量实验脚本
└── README.md                 # 本文件
```

## 安装

### 1. 安装 stable-pretraining

```bash
cd stable-pretraining
pip install -e ".[vision,tracking]"
```

### 2. 安装其他依赖

```bash
pip install transformers  # For loading HuggingFace MAE weights
pip install datasets==2.20.0  # For CPDataset
```

## 使用方法

### 方法 1: 单个实验

```bash
python mae_cp/mae_cp_train.py \
    --dataset bloodmnist \
    --data_root /root/data \
    --limit_data 100 \
    --model_size base \
    --pretrained \
    --pretrained_source facebook/vit-mae-base \
    --batch_size 256 \
    --epochs 100 \
    --lr 1.5e-4 \
    --output_dir /root/output/mae_cp
```

### 方法 2: 批量实验

```bash
# 编辑 run_mae_cp_experiments.sh 配置数据集和样本量
chmod +x mae_cp/run_mae_cp_experiments.sh
./mae_cp/run_mae_cp_experiments.sh
```

### 方法 3: Python API

```python
from mae_cp_train import train_mae_cp

train_mae_cp(
    dataset_name="bloodmnist",
    data_root="/root/data",
    limit_data=100,
    model_size="base",
    pretrained=True,
    pretrained_source="facebook/vit-mae-base",
    batch_size=256,
    epochs=100,
    output_dir="/root/output/mae_cp",
)
```

## 预训练权重

### HuggingFace 模型

| Model | HuggingFace ID | Hidden Dim | Layers |
|-------|---------------|------------|--------|
| MAE ViT-Base | `facebook/vit-mae-base` | 768 | 12 |
| MAE ViT-Large | `facebook/vit-mae-large` | 1024 | 24 |
| MAE ViT-Huge | `facebook/vit-mae-huge` | 1280 | 32 |

### 加载权重示例

```python
from load_mae_weights import load_pretrained_mae_weights
import stable_pretraining as spt

# Create MAE model
mae_model = spt.backbone.mae.vit_base_patch16_dec512d8b()

# Load pretrained weights from HuggingFace
load_pretrained_mae_weights(
    mae_model,
    source="facebook/vit-mae-base",
    strict=False
)

# Or load from local checkpoint
load_pretrained_mae_weights(
    mae_model,
    source="/path/to/mae_checkpoint.pth",
    strict=False
)
```

## 配置参数

### 数据集参数

- `--dataset`: 数据集名称（例如：`bloodmnist`, `food101`）
- `--data_root`: 数据根目录
- `--limit_data`: 限制样本数量（用于 few-shot 实验）

### 模型参数

- `--model_size`: 模型规格 (`base`, `large`, `huge`)
- `--pretrained`: 是否加载预训练权重
- `--pretrained_source`: 预训练权重来源（HuggingFace ID 或本地路径）
- `--mask_ratio`: Mask 比例（默认 0.75）
- `--norm_pix_loss`: 是否归一化像素值

### 训练参数

- `--batch_size`: 批次大小（默认 256）
- `--epochs`: 训练轮数（默认 100）
- `--lr`: 学习率（默认 1.5e-4）
- `--weight_decay`: 权重衰减（默认 0.05）
- `--warmup_epochs`: Warmup 轮数（默认 10）

### 硬件参数

- `--num_workers`: 数据加载线程数
- `--precision`: 训练精度 (`32`, `16-mixed`, `bf16-mixed`)
- `--devices`: GPU 数量

### 输出参数

- `--output_dir`: 输出目录
- `--exp_name`: 实验名称
- `--use_wandb`: 使用 W&B 记录
- `--wandb_project`: W&B 项目名称

## 监控

训练过程中会自动记录以下指标：

1. **Reconstruction Loss**: MAE 重建损失
2. **Linear Probe Accuracy**: 在线 linear probe 准确率
3. **RankMe**: 表示质量指标

使用 W&B 查看训练曲线：

```bash
python mae_cp/mae_cp_train.py \
    --dataset bloodmnist \
    --use_wandb \
    --wandb_project mae-cp
```

## 与 DINOv3-CP 的对比

| 特性 | DINOv3-CP | MAE-CP |
|------|-----------|---------|
| **训练目标** | Self-distillation (DINO + iBOT) | Reconstruction |
| **模型架构** | Teacher-Student | Encoder-Decoder |
| **Loss** | Cross-entropy + Sinkhorn | MSE on masked patches |
| **预训练来源** | facebook/dinov2-base | facebook/vit-mae-base |
| **框架** | 自定义训练循环 | PyTorch Lightning |
| **配置方式** | YAML + 自定义 config | Python + Hydra |

## 支持的数据集

### 新数据集
- Food101 (101 类, 75K 图像)
- FGVC-Aircraft (100 类, 10K 图像)
- Galaxy10-DECaLS (10 类, 17.7K 图像)

### MedMNIST
- BloodMNIST (8 类)
- PathMNIST (9 类)
- ChestMNIST (14 类)
- DermaMNIST (7 类)
- OCTMNIST (4 类)
- PneumoniaMNIST (2 类)
- RetinaMNIST (5 类)
- BreastMNIST (2 类)
- TissueMNIST (8 类)
- OrganAMNIST (11 类)
- OrganCMNIST (11 类)
- OrganSMNIST (11 类)

## 常见问题

### Q1: 如何修改数据增强？

编辑 `mae_cp_train.py` 中的 `create_mae_cp_transforms` 函数。

### Q2: 如何只加载 Encoder 权重？

```python
load_pretrained_mae_weights(
    mae_model,
    source="facebook/vit-mae-base",
    load_encoder_only=True,
    strict=False
)
```

### Q3: 如何使用自定义数据集？

修改 `mae_cp_dataset.py` 或在 `CPDataset` 中添加新数据集支持。

### Q4: 训练很慢怎么办？

1. 减小 `batch_size`
2. 增加 `devices` 使用多 GPU
3. 使用 `precision="16-mixed"` 混合精度训练
4. 增加 `num_workers` 加快数据加载

### Q5: Out of Memory 错误？

1. 减小 `batch_size`
2. 使用更小的模型 `--model_size base`
3. 使用梯度累积（需修改代码）

## 参考

- **MAE 论文**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- **stable-pretraining**: [GitHub](https://github.com/rbalestr-lab/stable-pretraining)
- **HuggingFace MAE**: [facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base)

## 引用

```bibtex
@article{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={CVPR},
  year={2022}
}
```

