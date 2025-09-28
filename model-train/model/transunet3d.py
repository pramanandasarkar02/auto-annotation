import torch
from monai.networks.nets import TransUNet


def create_transunet3d(
    img_size: int = 96,  # must be divisible by patch_size
    in_channels: int = 1,
    out_channels: int = 4,
    feature_size: int = 16,           # base number of filters
    hidden_size: int = 768,           # transformer hidden size
    mlp_dim: int = 3072,              # feed-forward layer size
    num_heads: int = 12,              # number of attention heads
    pos_embed: str = "perceptron",    # or "conv"
    norm_name: str = "instance",
    dropout_rate: float = 0.0,
    spatial_dims: int = 3,
    post_activation: str = "softmax", # or "sigmoid", depending on loss
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> TransUNet:
    model = TransUNet(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        pos_embed=pos_embed,
        norm_name=norm_name,
        dropout_rate=dropout_rate,
        spatial_dims=spatial_dims,
        post_activation=post_activation,
    )
    return model.to(device)
