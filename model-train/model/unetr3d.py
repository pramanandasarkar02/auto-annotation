import torch
from monai.networks.nets import UNETR

def create_unetr3d(
    img_size=(96, 96, 96),
    in_channels: int = 1,
    out_channels: int = 4,
    feature_size: int = 16,   # 16/24/32; larger = heavier
    hidden_size: int = 768,   # ViT dim
    mlp_dim: int = 3072,
    num_heads: int = 12,
    norm_name: str = "instance",
    dropout_rate: float = 0.0,
):
    return UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        # pos_embed="perceptron",  
        norm_name=norm_name,
        res_block=True,
        dropout_rate=dropout_rate,
    )