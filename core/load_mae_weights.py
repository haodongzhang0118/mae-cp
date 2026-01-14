"""
Load MAE pretrained weights from HuggingFace into stable-pretraining MAE model.

Supports both official MAE checkpoints and HuggingFace ViTMAE models.
"""

import torch
import logging
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def load_mae_from_huggingface(
    model_name: str = "facebook/vit-mae-base",
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load MAE weights from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier
            - "facebook/vit-mae-base": ViT-Base MAE
            - "facebook/vit-mae-large": ViT-Large MAE
            - "facebook/vit-mae-huge": ViT-Huge MAE
        cache_dir: Directory to cache downloaded models
        
    Returns:
        State dict compatible with stable-pretraining MAE model
    """
    try:
        from transformers import ViTMAEForPreTraining
    except ImportError:
        raise ImportError(
            "transformers is required to load HuggingFace MAE weights. "
            "Install with: pip install transformers"
        )
    
    logger.info(f"Loading MAE weights from {model_name}")
    hf_model = ViTMAEForPreTraining.from_pretrained(
        model_name, 
        cache_dir=cache_dir
    )
    hf_state_dict = hf_model.state_dict()
    
    # Convert HuggingFace state dict to stable-pretraining MAE format
    converted_state_dict = convert_hf_to_mae(hf_state_dict)
    
    logger.info(f"Successfully loaded and converted {len(converted_state_dict)} parameters")
    return converted_state_dict


def convert_hf_to_mae(hf_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace ViTMAE state dict to stable-pretraining MAE format.
    
    Key mappings:
    HF: vit.embeddings.patch_embeddings.projection -> MAE: patch_embed.proj
    HF: vit.embeddings.position_embeddings -> MAE: pos_embed
    HF: vit.encoder.layer.{i}.* -> MAE: blocks.{i}.*
    HF: decoder.* -> MAE: decoder_*
    
    HF uses separate query, key, value weights, but MAE uses fused qkv weights.
    This function handles the merging of q, k, v -> qkv.
    """
    mae_state_dict = {}
    
    # First pass: collect all keys and identify qkv components
    qkv_components = {}  # Store q, k, v separately to merge later
    
    for key, value in hf_state_dict.items():
        new_key = key
        
        # Skip the MAE head (we only want encoder + decoder)
        if "head" in key:
            continue
            
        # Patch embedding conversion
        if "vit.embeddings.patch_embeddings.projection" in key:
            new_key = key.replace(
                "vit.embeddings.patch_embeddings.projection", 
                "patch_embed.proj"
            )
            mae_state_dict[new_key] = value
        
        # Position embedding conversion
        elif "vit.embeddings.position_embeddings" in key:
            new_key = "pos_embed"
            mae_state_dict[new_key] = value
        
        # CLS token conversion
        elif "vit.embeddings.cls_token" in key:
            new_key = "cls_token"
            mae_state_dict[new_key] = value
        
        # Encoder blocks conversion
        elif "vit.encoder.layer" in key:
            # Extract layer number
            parts = key.split(".")
            layer_idx = parts[3]  # vit.encoder.layer.{idx}
            rest = ".".join(parts[4:])
            
            # Handle Q, K, V separately - need to merge into qkv
            if "attention.attention.query" in rest:
                param_type = "weight" if "weight" in rest else "bias"
                block_key = f"blocks.{layer_idx}.attn.qkv.{param_type}"
                if block_key not in qkv_components:
                    qkv_components[block_key] = {}
                qkv_components[block_key]["query"] = value
                continue
            elif "attention.attention.key" in rest:
                param_type = "weight" if "weight" in rest else "bias"
                block_key = f"blocks.{layer_idx}.attn.qkv.{param_type}"
                if block_key not in qkv_components:
                    qkv_components[block_key] = {}
                qkv_components[block_key]["key"] = value
                continue
            elif "attention.attention.value" in rest:
                param_type = "weight" if "weight" in rest else "bias"
                block_key = f"blocks.{layer_idx}.attn.qkv.{param_type}"
                if block_key not in qkv_components:
                    qkv_components[block_key] = {}
                qkv_components[block_key]["value"] = value
                continue
            
            # Convert other attention/mlp naming
            rest = rest.replace("attention.output.dense", "attn.proj")
            rest = rest.replace("intermediate.dense", "mlp.fc1")
            rest = rest.replace("output.dense", "mlp.fc2")
            rest = rest.replace("layernorm_before", "norm1")
            rest = rest.replace("layernorm_after", "norm2")
            
            new_key = f"blocks.{layer_idx}.{rest}"
            mae_state_dict[new_key] = value
        
        # Encoder norm conversion
        elif "vit.layernorm" in key:
            new_key = key.replace("vit.layernorm", "norm")
            mae_state_dict[new_key] = value
        
        # Decoder embedding conversion
        elif "decoder.decoder_embed" in key:
            new_key = key.replace("decoder.decoder_embed", "decoder_embed")
            mae_state_dict[new_key] = value
        
        # Decoder position embedding
        elif "decoder.decoder_pos_embed" in key:
            new_key = "decoder_pos_embed"
            mae_state_dict[new_key] = value
        
        # Mask token conversion
        elif "decoder.mask_token" in key:
            new_key = "mask_token"
            mae_state_dict[new_key] = value
        
        # Decoder blocks conversion
        elif "decoder.decoder_layers" in key:
            parts = key.split(".")
            layer_idx = parts[2]  # decoder.decoder_layers.{idx}
            rest = ".".join(parts[3:])
            
            # Handle Q, K, V separately - need to merge into qkv
            if "attention.attention.query" in rest:
                param_type = "weight" if "weight" in rest else "bias"
                block_key = f"decoder_blocks.{layer_idx}.attn.qkv.{param_type}"
                if block_key not in qkv_components:
                    qkv_components[block_key] = {}
                qkv_components[block_key]["query"] = value
                continue
            elif "attention.attention.key" in rest:
                param_type = "weight" if "weight" in rest else "bias"
                block_key = f"decoder_blocks.{layer_idx}.attn.qkv.{param_type}"
                if block_key not in qkv_components:
                    qkv_components[block_key] = {}
                qkv_components[block_key]["key"] = value
                continue
            elif "attention.attention.value" in rest:
                param_type = "weight" if "weight" in rest else "bias"
                block_key = f"decoder_blocks.{layer_idx}.attn.qkv.{param_type}"
                if block_key not in qkv_components:
                    qkv_components[block_key] = {}
                qkv_components[block_key]["value"] = value
                continue
            
            # Convert other attention/mlp naming
            rest = rest.replace("attention.output.dense", "attn.proj")
            rest = rest.replace("intermediate.dense", "mlp.fc1")
            rest = rest.replace("output.dense", "mlp.fc2")
            rest = rest.replace("layernorm_before", "norm1")
            rest = rest.replace("layernorm_after", "norm2")
            
            new_key = f"decoder_blocks.{layer_idx}.{rest}"
            mae_state_dict[new_key] = value
        
        # Decoder norm conversion
        elif "decoder.decoder_norm" in key:
            new_key = key.replace("decoder.decoder_norm", "decoder_norm")
            mae_state_dict[new_key] = value
        
        # Decoder prediction head
        elif "decoder.decoder_pred" in key:
            new_key = key.replace("decoder.decoder_pred", "decoder_pred")
            mae_state_dict[new_key] = value
    
    # Second pass: merge qkv components
    for qkv_key, components in qkv_components.items():
        if len(components) == 3:  # Should have q, k, v
            # Concatenate q, k, v along dim 0 (output dimension)
            qkv_merged = torch.cat([
                components["query"],
                components["key"],
                components["value"]
            ], dim=0)
            mae_state_dict[qkv_key] = qkv_merged
        else:
            logger.warning(f"Incomplete qkv components for {qkv_key}: {components.keys()}")
    
    return mae_state_dict


def load_mae_from_official_checkpoint(
    checkpoint_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Load MAE weights from official checkpoint.
    
    Download from: https://github.com/facebookresearch/mae
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        
    Returns:
        State dict compatible with stable-pretraining MAE model
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading MAE checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Official MAE checkpoints have 'model' key
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    logger.info(f"Successfully loaded {len(state_dict)} parameters")
    return state_dict


def load_pretrained_mae_weights(
    mae_model,
    source: str = "facebook/vit-mae-base",
    load_encoder_only: bool = False,
    load_decoder_only: bool = False,
    strict: bool = False,
) -> None:
    """
    Load pretrained weights into an MAE model.
    
    Args:
        mae_model: stable-pretraining MAE model instance
        source: Either HuggingFace model name or path to checkpoint file
        load_encoder_only: If True, only load encoder weights
        load_decoder_only: If True, only load decoder weights
        strict: Whether to strictly enforce matching keys
    """
    # Determine source type
    if source.startswith("facebook/") or "/" in source:
        # HuggingFace model
        state_dict = load_mae_from_huggingface(source)
    elif Path(source).exists():
        # Local checkpoint file
        state_dict = load_mae_from_official_checkpoint(source)
    else:
        raise ValueError(
            f"Invalid source: {source}. "
            "Must be a HuggingFace model name (e.g., 'facebook/vit-mae-base') "
            "or path to a checkpoint file."
        )
    
    # Filter weights based on load options
    if load_encoder_only:
        state_dict = {
            k: v for k, v in state_dict.items() 
            if not k.startswith("decoder")
        }
        logger.info("Loading encoder weights only")
    elif load_decoder_only:
        state_dict = {
            k: v for k, v in state_dict.items() 
            if k.startswith("decoder") or k == "mask_token"
        }
        logger.info("Loading decoder weights only")
    
    # Load weights
    missing_keys, unexpected_keys = mae_model.load_state_dict(
        state_dict, strict=strict
    )
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    logger.info("Successfully loaded pretrained MAE weights")


# Available pretrained models
PRETRAINED_MODELS = {
    "mae_vit_base_patch16": "facebook/vit-mae-base",
    "mae_vit_large_patch16": "facebook/vit-mae-large",
    "mae_vit_huge_patch14": "facebook/vit-mae-huge",
}


if __name__ == "__main__":
    # Test loading
    import sys
    sys.path.append("/Users/zhanghaodong/Desktop/DIET-CP/DINOv3-CP/dinov3/stable-pretraining")
    
    import stable_pretraining as spt
    
    # Create MAE model
    mae_model = spt.backbone.mae.vit_base_patch16_dec512d8b()
    
    # Load pretrained weights
    load_pretrained_mae_weights(
        mae_model,
        source="facebook/vit-mae-base",
        strict=False
    )
    
    print("✓ Successfully loaded MAE pretrained weights!")

